"""
src/eval_degeneracy_perturbed.py — Evaluate degeneracy detector
trên dataset có ground truth perturbation labels.

Metrics:
    Precision: Trong các steps detector flag, bao nhiêu đúng là perturbed?
    Recall:    Trong các steps thực sự perturbed, bao nhiêu được detect?
    F1:        Harmonic mean

Usage:
    python src/eval_degeneracy_perturbed.py \
        --checkpoint outputs/checkpoints/best_weight.ckpt \
        --data_path data/processed/nuscenes_mini_perturbed.h5 \
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


def load_model(checkpoint_path, device, latent_dim=192, action_dim=3,
               base_channels=32, hidden_dim=512, pred_layers=3):
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


@torch.no_grad()
def compute_all_surprises(model, observations, actions, device):
    """Tính surprise score cho toàn bộ dataset."""
    N, T = observations.shape[0], observations.shape[1]
    all_surprise = np.zeros((N, T - 1))

    for i in range(N):
        obs = torch.from_numpy(observations[i]).float().to(device)
        act = torch.from_numpy(actions[i]).float().to(device)

        z_all = model.encode(obs)  # (T, latent_dim)

        for t in range(T - 1):
            z_pred = model.predict_next(z_all[t:t+1], act[t:t+1])
            z_actual = z_all[t+1:t+2]
            surprise = ((z_pred - z_actual) ** 2).sum().item()
            all_surprise[i, t] = surprise

    return all_surprise


def evaluate_with_gt(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device,
                       latent_dim=args.latent_dim,
                       action_dim=args.action_dim)
    print(f"Model loaded from: {args.checkpoint}")

    with h5py.File(args.data_path, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]
        degeneracy_mask = f['degeneracy_mask'][:]       # (N, T)
        perturbation_types = f['perturbation_types'][:]  # (N,)

    N, T = observations.shape[0], observations.shape[1]
    print(f"Data: {N} sequences, {T} timesteps")
    print(f"  Clean:       {(perturbation_types == 0).sum()}")
    print(f"  Teleport:    {(perturbation_types == 1).sum()}")
    print(f"  Freeze:      {(perturbation_types == 2).sum()}")
    print(f"  Noise burst: {(perturbation_types == 3).sum()}")

    print("\nComputing surprise scores...")
    all_surprise = compute_all_surprises(model, observations, actions, device)

    # GT labels: align surprise[t] with transition t -> t+1,
    # degeneracy_mask[:, t+1] indicates if frame t+1 is perturbed
    gt_mask = degeneracy_mask[:, 1:]  # (N, T-1), aligned with surprise

    # Sweep thresholds by percentile
    surprise_flat = all_surprise.flatten()
    gt_flat = (gt_mask > 0).flatten().astype(bool)

    print(f"\n  GT perturbed steps: {gt_flat.sum()}/{len(gt_flat)} "
          f"({gt_flat.mean()*100:.1f}%)")

    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    best_f1 = 0
    best_result = None
    results_table = []

    for pct in percentiles:
        threshold = np.percentile(surprise_flat, pct)
        pred_flat = surprise_flat > threshold

        tp = int((pred_flat & gt_flat).sum())
        fp = int((pred_flat & ~gt_flat).sum())
        fn = int((~pred_flat & gt_flat).sum())
        tn = int((~pred_flat & ~gt_flat).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        results_table.append({
            'percentile': pct,
            'threshold': float(threshold),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_result = results_table[-1]

    print(f"\n{'='*75}")
    print(f"  {'Pct':>5} {'Threshold':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"{'='*75}")
    for r in results_table:
        marker = " <--" if r['f1'] == best_f1 else ""
        print(f"  {r['percentile']:>5} {r['threshold']:>10.2f} "
              f"{r['precision']:>10.3f} {r['recall']:>10.3f} "
              f"{r['f1']:>10.3f} {r['tp']:>6} {r['fp']:>6} "
              f"{r['fn']:>6}{marker}")
    print(f"{'='*75}")
    if best_result:
        print(f"  Best F1={best_f1:.3f} at {best_result['percentile']}th "
              f"percentile (threshold={best_result['threshold']:.2f})")

    # Per-type surprise analysis
    print(f"\n  Per-type surprise (mean +/- std):")
    for ptype, name in [(0, 'Clean'), (1, 'Teleport'),
                        (2, 'Freeze'), (3, 'Noise burst')]:
        mask = perturbation_types == ptype
        if mask.sum() == 0:
            continue
        s = all_surprise[mask].flatten()
        print(f"    {name:>12}: {s.mean():.2f} +/- {s.std():.2f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'degeneracy_eval.npz',
             surprise=all_surprise,
             gt_mask=gt_mask,
             perturbation_types=perturbation_types)
    print(f"\nSaved to: {out_dir / 'degeneracy_eval.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default="outputs/degeneracy_eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--action_dim", type=int, default=3)
    args = p.parse_args()
    evaluate_with_gt(args)
