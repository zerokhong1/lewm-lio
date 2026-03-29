"""
src/degeneracy_detector.py — Surprise-based degeneracy detection

Ý tưởng: Nếu world model predict ẑ_{t+1} khác xa z_{t+1} thực tế,
đó là dấu hiệu có điều bất thường (degeneracy, drift, hoặc scene change).

Metrics:
    surprise_t = ||ẑ_{t+1} - z_{t+1}||²
    degeneracy = surprise_t > threshold (cho k frames liên tiếp)

Usage:
    python src/degeneracy_detector.py \
        --checkpoint outputs/checkpoints/best_weight.ckpt \
        --data_path data/processed/nuscenes_mini_bev.h5 \
        --device cuda
"""
import torch
import numpy as np
import h5py
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from encoder_bev import BEVEncoder
from jepa_lidar import Predictor, LeWMLiDAR


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


class DegeneracyDetector:
    """
    Detect degeneracy dựa trên world model prediction surprise.

    Surprise score: s_t = ||ẑ_{t+1} - z_{t+1}||²

    Degeneracy flagged khi:
        s_t > mean(s) + k * std(s)   cho window_size frames liên tiếp
    """
    def __init__(self, model: LeWMLiDAR, device: str = 'cuda',
                 k_sigma: float = 2.0, window_size: int = 3):
        self.model = model
        self.device = device
        self.k_sigma = k_sigma
        self.window_size = window_size

    @torch.no_grad()
    def compute_surprise_scores(self, observations: np.ndarray,
                                actions: np.ndarray) -> dict:
        """
        Tính surprise score cho mỗi timestep trong sequence.

        Args:
            observations: (T, C, H, W) — BEV frames
            actions:      (T-1, action_dim) — ego-motion

        Returns:
            dict với:
                surprise:    (T-1,) — surprise score từng step
                threshold:   float  — adaptive threshold
                flags:       (T-1,) — binary degeneracy flags
        """
        obs = torch.from_numpy(observations).float().to(self.device)
        act = torch.from_numpy(actions).float().to(self.device)
        T = obs.shape[0]

        # Encode all frames
        z_all = self.model.encode(obs)  # (T, latent_dim)

        # One-step predictions
        surprise_scores = []
        for t in range(T - 1):
            z_t = z_all[t:t+1]      # (1, latent_dim)
            a_t = act[t:t+1]        # (1, action_dim)
            z_pred = self.model.predict_next(z_t, a_t)  # (1, latent_dim)
            z_actual = z_all[t+1:t+2]

            surprise = ((z_pred - z_actual) ** 2).sum().item()
            surprise_scores.append(surprise)

        surprise = np.array(surprise_scores)

        # Adaptive threshold
        mean_s = surprise.mean()
        std_s = surprise.std()
        threshold = mean_s + self.k_sigma * std_s

        # Flag degeneracy: surprise > threshold cho window_size consecutive
        above_threshold = surprise > threshold
        flags = np.zeros_like(surprise, dtype=bool)
        for i in range(len(above_threshold) - self.window_size + 1):
            window = above_threshold[i:i + self.window_size]
            if window.all():
                flags[i:i + self.window_size] = True

        return {
            'surprise': surprise,
            'threshold': threshold,
            'flags': flags,
            'mean': mean_s,
            'std': std_s,
        }


def evaluate_degeneracy(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = load_model(args.checkpoint, str(device),
                       latent_dim=args.latent_dim,
                       action_dim=args.action_dim)
    print(f"Model loaded from: {args.checkpoint}")

    detector = DegeneracyDetector(
        model=model, device=str(device),
        k_sigma=args.k_sigma, window_size=args.window_size,
    )

    with h5py.File(args.data_path, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]

    N = observations.shape[0]
    print(f"Data: {N} sequences")

    all_surprises = []
    all_flags = []

    for i in range(N):
        result = detector.compute_surprise_scores(observations[i], actions[i])
        all_surprises.append(result['surprise'])
        all_flags.append(result['flags'])

        n_flagged = result['flags'].sum()
        if n_flagged > 0 or (i + 1) % 10 == 0:
            print(f"  Seq {i+1}/{N}: "
                  f"surprise={result['mean']:.4f}±{result['std']:.4f} "
                  f"threshold={result['threshold']:.4f} "
                  f"flagged={n_flagged}/{len(result['flags'])}")

    all_surprises = np.concatenate(all_surprises)
    all_flags = np.concatenate(all_flags)

    print(f"\n{'='*60}")
    print(f"  Degeneracy Detection Summary")
    print(f"{'='*60}")
    print(f"  Total timesteps:    {len(all_surprises)}")
    print(f"  Mean surprise:      {all_surprises.mean():.4f}")
    print(f"  Std surprise:       {all_surprises.std():.4f}")
    print(f"  Flagged steps:      {all_flags.sum()} "
          f"({all_flags.mean()*100:.1f}%)")
    print(f"  k_sigma:            {args.k_sigma}")
    print(f"  window_size:        {args.window_size}")
    print(f"{'='*60}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'degeneracy_results.npz',
             surprise=all_surprises, flags=all_flags)
    print(f"Saved to: {out_dir / 'degeneracy_results.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/degeneracy")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--action_dim", type=int, default=3)
    p.add_argument("--k_sigma", type=float, default=2.0,
                   help="Threshold = mean + k_sigma * std")
    p.add_argument("--window_size", type=int, default=3,
                   help="Consecutive frames above threshold to flag")
    args = p.parse_args()
    evaluate_degeneracy(args)
