"""
src/visualize_ablation.py — Figures cho ablation results

Generates 3 figures:
  1. ablation_lambda  — Effect of SIGReg weight λ
  2. ablation_dim     — Planning-detection trade-off vs latent dim
  3. ablation_bev     — Effect of BEV resolution

Usage:
    python src/visualize_ablation.py \
        --ablation outputs/ablation_summary/ablation_results.json \
        --output_dir outputs/figures
"""
import json
import numpy as np
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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


def fig_lambda_ablation(results, output_dir):
    """Bar chart: lambda vs (latent_dist, F1)."""
    rows = sorted(
        [r for r in results if r['run'].startswith('lambda_')],
        key=lambda x: float(x['run'].split('_')[1])
    )
    if not rows:
        print("  SKIP lambda figure (no lambda_ runs)")
        return

    lambdas = [r['run'].split('_')[1] for r in rows]
    dists = [r['latent_dist_mean'] for r in rows]
    f1s = [r.get('best_f1', 0.0) for r in rows]

    # Color: red=worst (0.0), green=best (0.1), blue=others
    best_lambda = min(rows, key=lambda x: x['latent_dist_mean'])['run'].split('_')[1]
    colors = ['#5cb85c' if l == best_lambda else
              '#d9534f' if l == '0.0' else '#5bc0de'
              for l in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    x = np.arange(len(lambdas))
    w = 0.5

    bars1 = ax1.bar(x, dists, w, color=colors, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'λ={l}' for l in lambdas])
    ax1.set_ylabel('Latent distance ↓')
    ax1.set_title('Planning quality')
    # Annotate values
    for bar, v in zip(bars1, dists):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, f1s, w, color=colors, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'λ={l}' for l in lambdas])
    ax2.set_ylabel('F1 score ↑')
    ax2.set_title('Degeneracy detection')
    for bar, v in zip(bars2, f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Effect of SIGReg weight λ', fontsize=13)
    fig.tight_layout()
    save_fig(fig, output_dir, 'ablation_lambda')


def fig_dim_ablation(results, output_dir):
    """Dual-axis line chart: dim vs (latent_dist, F1) — trade-off."""
    rows = sorted(
        [r for r in results if r['run'].startswith('dim_')],
        key=lambda x: x['latent_dim']
    )
    if not rows:
        print("  SKIP dim figure (no dim_ runs)")
        return

    dims = [r['latent_dim'] for r in rows]
    dists = [r['latent_dist_mean'] for r in rows]
    f1s = [r.get('best_f1', 0.0) for r in rows]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(dims, dists, 'b-o', markersize=7, linewidth=1.5,
                   label='Latent dist ↓')
    l2, = ax2.plot(dims, f1s, 'r-s', markersize=7, linewidth=1.5,
                   label='F1 ↑')

    # Annotate points
    for d, v in zip(dims, dists):
        ax1.annotate(f'{v:.1f}', (d, v), textcoords='offset points',
                     xytext=(0, 6), ha='center', fontsize=8, color='b')
    for d, v in zip(dims, f1s):
        ax2.annotate(f'{v:.3f}', (d, v), textcoords='offset points',
                     xytext=(0, -14), ha='center', fontsize=8, color='r')

    ax1.set_xlabel('Latent dimension')
    ax1.set_ylabel('Latent distance ↓', color='b')
    ax2.set_ylabel('F1 score ↑', color='r')
    ax1.set_xticks(dims)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
    ax1.set_title('Planning–Detection trade-off vs dim')

    fig.tight_layout()
    save_fig(fig, output_dir, 'ablation_dim')


def fig_bev_ablation(results, output_dir):
    """Grouped bar: BEV resolution vs (sigma_z, latent_dist, F1)."""
    rows = sorted(
        [r for r in results if r['run'].startswith('bev_')],
        key=lambda x: int(x['run'].split('_')[1])
    )
    if not rows:
        print("  SKIP bev figure (no bev_ runs)")
        return

    labels = [f"{r['run'].split('_')[1]}x{r['run'].split('_')[1]}" for r in rows]
    stds = [r['z_global_std'] for r in rows]
    dists = [r['latent_dist_mean'] for r in rows]
    f1s = [r.get('best_f1', 0.0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    x = np.arange(len(labels))
    w = 0.5

    # Best BEV by F1
    best_idx = int(np.argmax(f1s))
    colors = ['#5cb85c' if i == best_idx else '#5bc0de'
              for i in range(len(rows))]

    axes[0].bar(x, stds, w, color=colors, edgecolor='white')
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7,
                    label='Ideal (σ=1.0)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('σ_z')
    axes[0].set_title('Isotropy (ideal=1.0)')
    axes[0].legend(fontsize=8)
    for xi, v in zip(x, stds):
        axes[0].text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

    axes[1].bar(x, dists, w, color=colors, edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Latent dist ↓')
    axes[1].set_title('Planning')
    for xi, v in zip(x, dists):
        axes[1].text(xi, v + 0.2, f'{v:.1f}', ha='center', fontsize=8)

    axes[2].bar(x, f1s, w, color=colors, edgecolor='white')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('F1 ↑')
    axes[2].set_title('Detection')
    for xi, v in zip(x, f1s):
        axes[2].text(xi, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    fig.suptitle('Effect of BEV resolution', fontsize=13)
    fig.tight_layout()
    save_fig(fig, output_dir, 'ablation_bev')


def main(args):
    set_paper_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.ablation) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} ablation runs from {args.ablation}")
    print("Generating ablation figures...")

    fig_lambda_ablation(results, out)
    fig_dim_ablation(results, out)
    fig_bev_ablation(results, out)

    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", type=str,
                   default="outputs/ablation_summary/ablation_results.json")
    p.add_argument("--output_dir", type=str, default="outputs/figures")
    args = p.parse_args()
    main(args)
