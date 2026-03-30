"""
src/collect_ablation.py — Chạy eval trên tất cả ablation checkpoints,
thu thập metrics vào 1 bảng tổng hợp.

Cho mỗi checkpoint:
  1. Latent analysis (z_std, uniformity, effective dims)
  2. Planning eval (latent_dist, planning_time)
  3. Degeneracy eval trên perturbed data (best F1)

Usage:
    python src/collect_ablation.py \
        --ablation_dir outputs/ablation \
        --clean_data data/processed/nuscenes_mini_bev.h5 \
        --perturbed_data data/processed/nuscenes_mini_perturbed.h5 \
        --output_dir outputs/ablation_summary \
        --device cuda
"""
import torch
import numpy as np
import h5py
import json
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from encoder_bev import BEVEncoder
from jepa_lidar import Predictor, LeWMLiDAR
from eval_planning import CEMPlanner


def load_model(ckpt_path, device, latent_dim=192, action_dim=3,
               base_channels=32, hidden_dim=512, pred_layers=3):
    encoder = BEVEncoder(in_channels=4, latent_dim=latent_dim,
                         base_channels=base_channels)
    predictor = Predictor(latent_dim=latent_dim, action_dim=action_dim,
                          hidden_dim=hidden_dim, num_layers=pred_layers)
    model = LeWMLiDAR(encoder, predictor)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def eval_latent(model, observations, device):
    """Latent space quality metrics."""
    all_z = []
    for i in range(len(observations)):
        obs = torch.from_numpy(observations[i]).float().to(device)
        z = model.encode(obs)
        all_z.append(z.cpu().numpy())
    embeddings = np.concatenate(all_z, axis=0)

    z_std = embeddings.std(axis=0)
    return {
        'z_global_std': float(embeddings.std()),
        'z_std_uniformity': float(z_std.std()),
        'z_std_mean': float(z_std.mean()),
        'effective_dims': int((z_std > 0.1).sum()),
        'total_dims': int(len(z_std)),
    }


@torch.no_grad()
def eval_planning_quick(model, observations, actions, device,
                        num_evals=10, cem_samples=256, cem_steps=3):
    """Quick planning eval."""
    N, T = observations.shape[0], observations.shape[1]
    num_evals = min(num_evals, N)

    planner = CEMPlanner(
        model=model, action_dim=actions.shape[2],
        horizon=T - 1, num_samples=cem_samples,
        num_elites=32, opt_steps=cem_steps, device=str(device),
    )

    dists, times = [], []
    for i in range(num_evals):
        obs = torch.from_numpy(observations[i]).float().to(device)
        z_start = model.encode(obs[0:1])[0]
        z_goal = model.encode(obs[-1:])[0]

        t0 = time.time()
        planned = planner.plan(z_start, z_goal)
        t_ms = (time.time() - t0) * 1000

        z = z_start.unsqueeze(0)
        for t in range(T - 1):
            z = model.predict_next(z, planned[t:t+1])
        dist = ((z[0] - z_goal) ** 2).sum().sqrt().item()

        dists.append(dist)
        times.append(t_ms)

    return {
        'latent_dist_mean': float(np.mean(dists)),
        'latent_dist_std': float(np.std(dists)),
        'plan_time_ms': float(np.mean(times)),
    }


@torch.no_grad()
def eval_degeneracy_quick(model, observations, actions, gt_mask, device):
    """Quick degeneracy detection eval — return best F1."""
    N, T = observations.shape[0], observations.shape[1]

    all_surprise = []
    for i in range(N):
        obs = torch.from_numpy(observations[i]).float().to(device)
        act = torch.from_numpy(actions[i]).float().to(device)
        z_all = model.encode(obs)
        for t in range(T - 1):
            z_pred = model.predict_next(z_all[t:t+1], act[t:t+1])
            s = ((z_pred - z_all[t+1:t+2]) ** 2).sum().item()
            all_surprise.append(s)

    surprise_flat = np.array(all_surprise)
    gt_flat = (gt_mask[:, 1:] > 0).flatten().astype(bool)

    best_f1, best_pct = 0.0, 0
    for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
        thr = np.percentile(surprise_flat, pct)
        pred = surprise_flat > thr
        tp = int((pred & gt_flat).sum())
        fp = int((pred & ~gt_flat).sum())
        fn = int((~pred & gt_flat).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_pct = f1, pct

    return {
        'best_f1': float(best_f1),
        'best_threshold_pct': best_pct,
        'mean_surprise': float(surprise_flat.mean()),
    }


def find_checkpoint(run_dir):
    """Tìm best hoặc final checkpoint trong run."""
    ckpt_dir = Path(run_dir) / 'ckpts'
    if not ckpt_dir.exists():
        return None
    for name in ['best_weight.ckpt', 'final_weight.ckpt']:
        p = ckpt_dir / name
        if p.exists():
            return str(p)
    ckpts = sorted(ckpt_dir.glob('*.ckpt'))
    return str(ckpts[-1]) if ckpts else None


def infer_latent_dim(run_name):
    """Đoán latent_dim từ tên run."""
    if 'dim_' in run_name:
        try:
            return int(run_name.split('dim_')[1])
        except ValueError:
            pass
    return 192


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with h5py.File(args.clean_data, 'r') as f:
        clean_obs = f['observations'][:]
        clean_act = f['actions'][:]

    perturbed_obs = perturbed_act = perturbed_mask = None
    if args.perturbed_data and Path(args.perturbed_data).exists():
        with h5py.File(args.perturbed_data, 'r') as f:
            perturbed_obs = f['observations'][:]
            perturbed_act = f['actions'][:]
            perturbed_mask = f['degeneracy_mask'][:]
        print(f"Perturbed data loaded: {perturbed_obs.shape[0]} sequences")

    ablation_dir = Path(args.ablation_dir)
    runs = sorted([d for d in ablation_dir.iterdir() if d.is_dir()])
    print(f"Found {len(runs)} ablation runs in {ablation_dir}")

    results = []

    for run_dir in runs:
        run_name = run_dir.name
        ckpt = find_checkpoint(run_dir)
        if ckpt is None:
            print(f"  SKIP {run_name}: no checkpoint found")
            continue

        print(f"\n{'-'*50}")
        print(f"  Run: {run_name}")

        latent_dim = infer_latent_dim(run_name)

        try:
            model = load_model(ckpt, device, latent_dim=latent_dim)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        row = {'run': run_name, 'latent_dim': latent_dim}

        print("    Latent analysis...", end=' ', flush=True)
        lm = eval_latent(model, clean_obs, device)
        row.update(lm)
        print(f"std={lm['z_global_std']:.3f}, eff_dims={lm['effective_dims']}")

        print("    Planning eval...", end=' ', flush=True)
        pm = eval_planning_quick(model, clean_obs, clean_act, device)
        row.update(pm)
        print(f"dist={pm['latent_dist_mean']:.2f}, time={pm['plan_time_ms']:.0f}ms")

        if perturbed_obs is not None:
            print("    Degeneracy eval...", end=' ', flush=True)
            dm = eval_degeneracy_quick(
                model, perturbed_obs, perturbed_act, perturbed_mask, device)
            row.update(dm)
            print(f"F1={dm['best_f1']:.3f}")

        results.append(row)

    # Summary table
    print(f"\n{'='*85}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*85}")
    has_f1 = perturbed_obs is not None
    header = (f"  {'Run':<22} {'z_std':>6} {'Unif':>7} {'Eff':>7} "
              f"{'Dist':>7} {'ms':>5}")
    if has_f1:
        header += f"  {'F1':>6}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for r in results:
        eff = f"{r['effective_dims']}/{r['total_dims']}"
        line = (f"  {r['run']:<22} "
                f"{r['z_global_std']:>6.3f} "
                f"{r['z_std_uniformity']:>7.4f} "
                f"{eff:>7} "
                f"{r['latent_dist_mean']:>7.2f} "
                f"{r['plan_time_ms']:>5.0f}")
        if has_f1 and 'best_f1' in r:
            line += f"  {r['best_f1']:>6.3f}"
        print(line)
    print(f"{'='*85}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if results:
        keys = list(results[0].keys())
        csv_path = out_dir / 'ablation_results.csv'
        with open(csv_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in results:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')

    print(f"\nSaved to: {out_dir}")
    print(f"  ablation_results.json")
    print(f"  ablation_results.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", type=str, default="outputs/ablation")
    p.add_argument("--clean_data", type=str,
                   default="data/processed/nuscenes_mini_bev.h5")
    p.add_argument("--perturbed_data", type=str,
                   default="data/processed/nuscenes_mini_perturbed.h5")
    p.add_argument("--output_dir", type=str,
                   default="outputs/ablation_summary")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    main(args)
