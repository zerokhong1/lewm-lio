"""
src/analyze_latent.py — Latent space analysis

1. Encode toàn bộ dataset → latent embeddings
2. Thống kê phân phối latent (mean, std, normality)
3. Kiểm tra SIGReg effectiveness
4. Xuất embeddings cho t-SNE visualization

Usage:
    python src/analyze_latent.py \
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


def analyze(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, str(device))

    with h5py.File(args.data_path, 'r') as f:
        observations = f['observations'][:]  # (N, T, C, H, W)

    N, T, C, H, W = observations.shape
    print(f"Encoding {N*T} frames...")

    all_z = []
    with torch.no_grad():
        for i in range(N):
            obs = torch.from_numpy(observations[i]).float().to(device)
            z = model.encode(obs)  # (T, latent_dim)
            all_z.append(z.cpu().numpy())

    embeddings = np.concatenate(all_z, axis=0)  # (N*T, latent_dim)

    z_mean = embeddings.mean(axis=0)
    z_std = embeddings.std(axis=0)
    z_global_mean = embeddings.mean()
    z_global_std = embeddings.std()

    print(f"\n{'='*60}")
    print(f"  Latent Space Analysis")
    print(f"{'='*60}")
    print(f"  Embeddings shape:       {embeddings.shape}")
    print(f"  Global mean:            {z_global_mean:.4f}")
    print(f"  Global std:             {z_global_std:.4f}")
    print(f"  Per-dim std range:      [{z_std.min():.4f}, {z_std.max():.4f}]")
    print(f"  Per-dim std uniformity: {z_std.std():.4f} (lower=more isotropic)")
    print(f"  Effective dims (std>0.1): {(z_std > 0.1).sum()}/{len(z_std)}")
    print(f"{'='*60}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'latent_analysis.npz',
             embeddings=embeddings,
             z_mean=z_mean,
             z_std=z_std)
    print(f"Saved to: {out_dir}")
    print(f"  - Load in notebook: np.load('latent_analysis.npz')")
    print(f"  - Use embeddings for t-SNE visualization")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/analysis")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    analyze(args)
