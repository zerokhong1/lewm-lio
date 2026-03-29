"""
src/visualize.py — Tạo figures cho paper

Figures:
  1. sigreg_std_hist   — Per-dim std histogram (SIGReg effectiveness)
  2. tsne_latent       — t-SNE latent space (requires scikit-learn)
  3. surprise_timeline — Surprise score: clean vs perturbed sequence
  4. planning_results  — Planning latent distance + time distribution

Usage:
    python src/visualize.py \
        --analysis_dir outputs/analysis \
        --degeneracy_dir outputs/degeneracy_eval \
        --planning_dir outputs/eval \
        --output_dir outputs/figures
"""
import numpy as np
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (headless on TEA)
import matplotlib.pyplot as plt


def set_paper_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def save_fig(fig, out_dir, name):
    fig.savefig(Path(out_dir) / f'{name}.png')
    fig.savefig(Path(out_dir) / f'{name}.pdf')
    plt.close(fig)
    print(f"  Saved: {name}.png/pdf")


def fig_std_histogram(analysis_dir, output_dir):
    """Per-dimension std histogram (SIGReg effectiveness)."""
    fpath = Path(analysis_dir) / 'latent_analysis.npz'
    if not fpath.exists():
        print(f"  SKIP std histogram (no {fpath})")
        return

    data = np.load(fpath)
    z_std = data['z_std']  # (latent_dim,)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(z_std, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
               label='Ideal (std=1.0)')
    ax.axvline(x=float(z_std.mean()), color='orange', linestyle='-',
               linewidth=1.5, label=f'Mean={z_std.mean():.3f}')
    ax.set_xlabel('Standard deviation per dimension')
    ax.set_ylabel('Count')
    ax.set_title('SIGReg: Latent dimension isotropy')
    ax.legend()
    save_fig(fig, output_dir, 'sigreg_std_hist')


def fig_tsne(analysis_dir, output_dir):
    """t-SNE visualization of latent space."""
    fpath = Path(analysis_dir) / 'latent_analysis.npz'
    if not fpath.exists():
        print(f"  SKIP t-SNE (no {fpath})")
        return

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  SKIP t-SNE (sklearn not installed — pip install scikit-learn)")
        return

    data = np.load(fpath)
    embeddings = data['embeddings']  # (N*T, latent_dim)

    # Subsample if too large (t-SNE is O(N^2))
    max_points = 2000
    if len(embeddings) > max_points:
        idx = np.random.default_rng(42).choice(
            len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        print(f"  t-SNE: subsampled to {max_points} points")

    print(f"  Computing t-SNE on {len(embeddings)} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(embeddings)

    T = 16  # frames per sequence
    colors = np.arange(len(embeddings)) % T  # color by timestep within seq

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors,
                    cmap='viridis', s=8, alpha=0.6)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.set_title('Latent Space (colored by timestep)')
    plt.colorbar(sc, ax=ax, label='Timestep within sequence')
    save_fig(fig, output_dir, 'tsne_latent')


def fig_surprise_timeline(degeneracy_dir, output_dir):
    """Surprise score timeline — clean vs perturbed sequence."""
    fpath = Path(degeneracy_dir) / 'degeneracy_eval.npz'
    if not fpath.exists():
        print(f"  SKIP surprise timeline (no {fpath})")
        return

    data = np.load(fpath)
    surprise = data['surprise']           # (N, T-1)
    gt_mask = data['gt_mask']             # (N, T-1)
    ptypes = data['perturbation_types']   # (N,)

    clean_idxs = np.where(ptypes == 0)[0]
    perturbed_idxs = np.where(ptypes > 0)[0]

    if len(clean_idxs) == 0 or len(perturbed_idxs) == 0:
        print("  SKIP surprise timeline (need both clean and perturbed seqs)")
        return

    ptype_name = {1: 'Teleport', 2: 'Freeze', 3: 'Noise burst'}

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    ci = clean_idxs[0]
    axes[0].plot(surprise[ci], 'b-o', markersize=4, label='Surprise')
    axes[0].set_ylabel('Surprise score')
    axes[0].set_title(f'Clean sequence (#{ci})')
    axes[0].legend()

    pi = perturbed_idxs[0]
    axes[1].plot(surprise[pi], 'b-o', markersize=4, label='Surprise')
    perturbed_steps = np.where(gt_mask[pi] > 0)[0]
    if len(perturbed_steps) > 0:
        axes[1].axvspan(perturbed_steps[0] - 0.5,
                        perturbed_steps[-1] + 0.5,
                        alpha=0.2, color='red', label='Perturbed region')
    axes[1].set_ylabel('Surprise score')
    axes[1].set_xlabel('Timestep')
    pname = ptype_name.get(int(ptypes[pi]), 'Unknown')
    axes[1].set_title(f'Perturbed sequence (#{pi}, type={pname})')
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, output_dir, 'surprise_timeline')


def fig_planning_dist(planning_dir, output_dir):
    """Planning latent distance and time distribution."""
    fpath = Path(planning_dir) / 'planning_results.npz'
    if not fpath.exists():
        print(f"  SKIP planning dist (no {fpath})")
        return

    data = np.load(fpath)
    latent_dist = data['latent_distance']
    planning_time = data['planning_time_ms']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1.hist(latent_dist, bins=15, color='steelblue', edgecolor='white')
    ax1.axvline(x=float(np.mean(latent_dist)), color='red', linestyle='--',
                label=f'Mean={np.mean(latent_dist):.2f}')
    ax1.set_xlabel('Latent distance to goal')
    ax1.set_ylabel('Count')
    ax1.set_title('Planning: Goal reaching')
    ax1.legend()

    ax2.hist(planning_time, bins=15, color='coral', edgecolor='white')
    ax2.axvline(x=float(np.mean(planning_time)), color='red', linestyle='--',
                label=f'Mean={np.mean(planning_time):.0f}ms')
    ax2.set_xlabel('Planning time (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Planning: Efficiency')
    ax2.legend()

    fig.tight_layout()
    save_fig(fig, output_dir, 'planning_results')


def main(args):
    set_paper_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")

    if Path(args.analysis_dir).exists():
        fig_std_histogram(args.analysis_dir, out)
        fig_tsne(args.analysis_dir, out)

    if Path(args.degeneracy_dir).exists():
        fig_surprise_timeline(args.degeneracy_dir, out)

    if Path(args.planning_dir).exists():
        fig_planning_dist(args.planning_dir, out)

    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir", type=str, default="outputs/analysis")
    p.add_argument("--degeneracy_dir", type=str,
                   default="outputs/degeneracy_eval")
    p.add_argument("--planning_dir", type=str, default="outputs/eval")
    p.add_argument("--output_dir", type=str, default="outputs/figures")
    args = p.parse_args()
    main(args)
