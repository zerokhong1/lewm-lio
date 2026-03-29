"""
src/inject_degeneracy.py — Tạo perturbed BEV dataset

Mô phỏng 3 loại degeneracy:
  1. Teleportation: đột ngột thay đổi BEV frame (mô phỏng LIO drift lớn)
  2. Freeze: lặp lại frame cũ (mô phỏng LiDAR bị kẹt/mất tín hiệu)
  3. Noise burst: thêm noise mạnh vào BEV (mô phỏng scan trong môi trường degenerate)

Output: HDF5 mới với thêm ground truth labels degeneracy_mask

Usage:
    python src/inject_degeneracy.py \
        --input data/processed/nuscenes_mini_bev.h5 \
        --output data/processed/nuscenes_mini_perturbed.h5
"""
import numpy as np
import h5py
import argparse
from pathlib import Path


def inject_teleport(obs_seq, act_seq, t_inject, rng):
    """
    Teleportation: Tại t_inject, thay BEV frame bằng frame từ
    vị trí hoàn toàn khác (lấy random frame khác trong sequence).
    """
    obs = obs_seq.copy()
    T = obs.shape[0]

    candidates = [t for t in range(T) if abs(t - t_inject) > T // 3]
    if not candidates:
        candidates = [0 if t_inject > T // 2 else T - 1]
    src = rng.choice(candidates)

    obs[t_inject] = obs[src]
    return obs, act_seq


def inject_freeze(obs_seq, act_seq, t_inject, duration=3, rng=None):
    """
    Freeze: Từ t_inject, lặp lại cùng 1 frame cho duration steps.
    """
    obs = obs_seq.copy()
    T = obs.shape[0]
    end = min(t_inject + duration, T)
    for t in range(t_inject + 1, end):
        obs[t] = obs[t_inject]
    return obs, act_seq


def inject_noise_burst(obs_seq, act_seq, t_inject, duration=3,
                       noise_scale=2.0, rng=None):
    """
    Noise burst: Thêm Gaussian noise mạnh vào BEV frames.
    """
    obs = obs_seq.copy()
    T = obs.shape[0]
    end = min(t_inject + duration, T)
    for t in range(t_inject, end):
        noise = rng.standard_normal(obs[t].shape).astype(np.float32)
        obs[t] = obs[t] + noise_scale * noise
    return obs, act_seq


def create_perturbed_dataset(input_path, output_path,
                             perturbation_ratio=0.5, seed=42):
    """
    Tạo dataset mới với một phần sequences bị perturbed.

    Args:
        perturbation_ratio: tỷ lệ sequences bị perturb (0.5 = 50%)
    """
    rng = np.random.default_rng(seed)

    with h5py.File(input_path, 'r') as f:
        observations = f['observations'][:]  # (N, T, C, H, W)
        actions = f['actions'][:]            # (N, T-1, action_dim)
        attrs = dict(f.attrs)

    N, T, C, H, W = observations.shape
    print(f"Input: {N} sequences, {T} timesteps, shape=({C},{H},{W})")

    out_obs = observations.copy()
    out_act = actions.copy()
    degeneracy_mask = np.zeros((N, T), dtype=np.int32)
    # Labels: 0=clean, 1=teleport, 2=freeze, 3=noise_burst
    perturbation_types = np.zeros(N, dtype=np.int32)

    n_perturb = int(N * perturbation_ratio)
    perturb_idxs = set(rng.choice(N, size=n_perturb, replace=False).tolist())

    stats = {'clean': 0, 'teleport': 0, 'freeze': 0, 'noise_burst': 0}

    for idx in range(N):
        if idx not in perturb_idxs:
            stats['clean'] += 1
            continue

        ptype = int(rng.choice([1, 2, 3]))
        t_inject = int(rng.integers(3, T - 4))

        if ptype == 1:  # teleport
            out_obs[idx], out_act[idx] = inject_teleport(
                observations[idx], actions[idx], t_inject, rng)
            degeneracy_mask[idx, t_inject] = 1
            stats['teleport'] += 1

        elif ptype == 2:  # freeze
            duration = int(rng.integers(2, 5))
            out_obs[idx], out_act[idx] = inject_freeze(
                observations[idx], actions[idx], t_inject,
                duration=duration, rng=rng)
            end = min(t_inject + duration, T)
            degeneracy_mask[idx, t_inject + 1:end] = 2
            stats['freeze'] += 1

        elif ptype == 3:  # noise burst
            duration = int(rng.integers(2, 5))
            out_obs[idx], out_act[idx] = inject_noise_burst(
                observations[idx], actions[idx], t_inject,
                duration=duration, noise_scale=2.0, rng=rng)
            end = min(t_inject + duration, T)
            degeneracy_mask[idx, t_inject:end] = 3
            stats['noise_burst'] += 1

        perturbation_types[idx] = ptype

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('observations', data=out_obs,
                         compression='gzip', compression_opts=4)
        f.create_dataset('actions', data=out_act,
                         compression='gzip', compression_opts=4)
        f.create_dataset('degeneracy_mask', data=degeneracy_mask)
        f.create_dataset('perturbation_types', data=perturbation_types)
        for k, v in attrs.items():
            f.attrs[k] = v

    total_flagged = (degeneracy_mask > 0).sum()
    total_steps = N * T

    print(f"\nOutput: {output_path}")
    print(f"  Clean sequences:       {stats['clean']}")
    print(f"  Teleport sequences:    {stats['teleport']}")
    print(f"  Freeze sequences:      {stats['freeze']}")
    print(f"  Noise burst sequences: {stats['noise_burst']}")
    print(f"  Total perturbed steps: {total_flagged}/{total_steps} "
          f"({total_flagged / total_steps * 100:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str,
                   default="data/processed/nuscenes_mini_bev.h5")
    p.add_argument("--output", type=str,
                   default="data/processed/nuscenes_mini_perturbed.h5")
    p.add_argument("--perturbation_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    create_perturbed_dataset(args.input, args.output,
                             args.perturbation_ratio, args.seed)
