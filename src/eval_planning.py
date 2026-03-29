"""
src/eval_planning.py — CEM-based planning trong latent space

Usage:
    python src/eval_planning.py \
        --checkpoint outputs/checkpoints/best_weight.ckpt \
        --data_path data/processed/nuscenes_mini_bev.h5 \
        --num_evals 20 \
        --device cuda
"""
import torch
import numpy as np
import h5py
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from encoder_bev import BEVEncoder
from jepa_lidar import Predictor, LeWMLiDAR


class CEMPlanner:
    """
    Cross-Entropy Method planner trong latent space.

    Tối ưu action sequence a_{0:H-1} sao cho:
        minimize ||pred_latent(z_start, a_{0:H-1}) - z_goal||²
    """
    def __init__(self, model: LeWMLiDAR, action_dim: int = 3,
                 horizon: int = 15, num_samples: int = 512,
                 num_elites: int = 64, opt_steps: int = 5,
                 device: str = 'cuda'):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.opt_steps = opt_steps
        self.device = device

    @torch.no_grad()
    def plan(self, z_start: torch.Tensor, z_goal: torch.Tensor,
             action_mean_init: torch.Tensor = None,
             action_std_init: float = 1.0) -> torch.Tensor:
        """
        Tìm action sequence tốt nhất.

        Args:
            z_start: (latent_dim,) — latent hiện tại
            z_goal:  (latent_dim,) — latent mục tiêu

        Returns:
            best_actions: (horizon, action_dim)
        """
        H = self.horizon
        D = self.action_dim

        if action_mean_init is not None:
            mean = action_mean_init.clone()
        else:
            mean = torch.zeros(H, D, device=self.device)
        std = torch.ones(H, D, device=self.device) * action_std_init

        for _ in range(self.opt_steps):
            # Sample action sequences: (num_samples, H, D)
            noise = torch.randn(self.num_samples, H, D, device=self.device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise

            # Rollout each sample through world model
            costs = self._evaluate(z_start, z_goal, actions)

            # Select elites
            elite_idxs = torch.argsort(costs)[:self.num_elites]
            elites = actions[elite_idxs]  # (num_elites, H, D)

            # Update distribution
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=0.01)

        return mean

    def _evaluate(self, z_start: torch.Tensor, z_goal: torch.Tensor,
                  actions: torch.Tensor) -> torch.Tensor:
        """
        Rollout actions và tính cost.

        Args:
            z_start: (latent_dim,)
            z_goal:  (latent_dim,)
            actions: (N, H, D)

        Returns:
            costs: (N,)
        """
        N = actions.shape[0]
        z = z_start.unsqueeze(0).expand(N, -1)  # (N, latent_dim)

        for t in range(self.horizon):
            a_t = actions[:, t, :]  # (N, D)
            z = self.model.predict_next(z, a_t)

        # Cost = distance to goal
        costs = ((z - z_goal.unsqueeze(0)) ** 2).sum(dim=-1)  # (N,)
        return costs


def load_model(checkpoint_path: str, device: str,
               latent_dim: int = 192, action_dim: int = 3,
               base_channels: int = 32, hidden_dim: int = 512,
               pred_layers: int = 3) -> LeWMLiDAR:
    encoder = BEVEncoder(in_channels=4, latent_dim=latent_dim,
                         base_channels=base_channels)
    predictor = Predictor(latent_dim=latent_dim, action_dim=action_dim,
                          hidden_dim=hidden_dim, num_layers=pred_layers)
    model = LeWMLiDAR(encoder, predictor)
    state_dict = torch.load(checkpoint_path, map_location=device,
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_planning(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, str(device),
                       latent_dim=args.latent_dim,
                       action_dim=args.action_dim)
    print(f"Model loaded from: {args.checkpoint}")

    with h5py.File(args.data_path, 'r') as f:
        observations = f['observations'][:]  # (N, T, C, H, W)
        actions_gt = f['actions'][:]          # (N, T-1, action_dim)

    N, T, C, H, W = observations.shape
    print(f"Data: {N} sequences, {T} timesteps")

    planner = CEMPlanner(
        model=model,
        action_dim=args.action_dim,
        horizon=T - 1,
        num_samples=args.cem_samples,
        num_elites=args.cem_elites,
        opt_steps=args.cem_opt_steps,
        device=str(device),
    )

    results = {
        'latent_distance': [],
        'action_mse': [],
        'planning_time_ms': [],
    }

    num_evals = min(args.num_evals, N)
    print(f"\nEvaluating {num_evals} episodes...")

    for i in range(num_evals):
        obs_seq = torch.from_numpy(observations[i]).float().to(device)
        act_gt = torch.from_numpy(actions_gt[i]).float().to(device)

        with torch.no_grad():
            z_start = model.encode(obs_seq[0:1])[0]   # (latent_dim,)
            z_goal  = model.encode(obs_seq[-1:])[0]    # (latent_dim,)

        t0 = time.time()
        planned_actions = planner.plan(z_start, z_goal)  # (H, action_dim)
        plan_time_ms = (time.time() - t0) * 1000

        with torch.no_grad():
            z = z_start.unsqueeze(0)
            for t in range(T - 1):
                z = model.predict_next(z, planned_actions[t:t+1])
            z_final = z[0]

        latent_dist = ((z_final - z_goal) ** 2).sum().sqrt().item()
        act_mse = ((planned_actions - act_gt) ** 2).mean().item()

        results['latent_distance'].append(latent_dist)
        results['action_mse'].append(act_mse)
        results['planning_time_ms'].append(plan_time_ms)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{num_evals}] "
                  f"latent_dist={latent_dist:.4f} "
                  f"act_mse={act_mse:.4f} "
                  f"time={plan_time_ms:.0f}ms")

    print(f"\n{'='*60}")
    print(f"  Planning Results ({num_evals} episodes)")
    print(f"{'='*60}")
    print(f"  Latent distance:  {np.mean(results['latent_distance']):.4f} "
          f"± {np.std(results['latent_distance']):.4f}")
    print(f"  Action MSE:       {np.mean(results['action_mse']):.4f} "
          f"± {np.std(results['action_mse']):.4f}")
    print(f"  Planning time:    {np.mean(results['planning_time_ms']):.0f} "
          f"± {np.std(results['planning_time_ms']):.0f} ms")
    print(f"{'='*60}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'planning_results.npz', **results)
    print(f"Saved to: {out_dir / 'planning_results.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/eval")
    p.add_argument("--num_evals", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--action_dim", type=int, default=3)
    p.add_argument("--cem_samples", type=int, default=512)
    p.add_argument("--cem_elites", type=int, default=64)
    p.add_argument("--cem_opt_steps", type=int, default=5)
    args = p.parse_args()
    evaluate_planning(args)
