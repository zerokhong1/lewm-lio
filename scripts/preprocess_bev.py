"""
preprocess_bev.py -- nuScenes point cloud -> BEV HDF5

Chay tren CONT. Output copy sang TEA.

Usage:
    python scripts/preprocess_bev.py \
        --nuscenes_root data/nuscenes \
        --version v1.0-mini \
        --output data/processed/nuscenes_mini_bev.h5 \
        --bev_size 256 \
        --resolution 0.4

Output HDF5 format (tuong thich LeWM):
    observations: (num_seqs, seq_length, 4, H, W)  -- BEV images
    actions:      (num_seqs, seq_length-1, 3)       -- (dx, dy, dyaw)
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from pyquaternion import Quaternion


# -----------------------------------------------------------------------
# Core BEV conversion (khong phu thuoc nuscenes-devkit, co the test rieng)
# -----------------------------------------------------------------------

def load_pointcloud(lidar_path: str) -> np.ndarray:
    """Load nuScenes .bin -> (N, 5): x, y, z, intensity, ring."""
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)


def pointcloud_to_bev(points: np.ndarray,
                      x_range=(-51.2, 51.2),
                      y_range=(-51.2, 51.2),
                      z_range=(-5.0, 3.0),
                      resolution: float = 0.4) -> np.ndarray:
    """
    Point cloud (N, >=3) -> BEV image (4, H, W).
    C=4: [height_max, height_mean, log_density, intensity_max]
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    H = int(round((x_max - x_min) / resolution))
    W = int(round((y_max - y_min) / resolution))

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[mask]

    if len(pts) == 0:
        return np.zeros((4, H, W), dtype=np.float32)

    xi = np.clip(((pts[:, 0] - x_min) / resolution).astype(np.int32), 0, H - 1)
    yi = np.clip(((pts[:, 1] - y_min) / resolution).astype(np.int32), 0, W - 1)

    bev   = np.zeros((4, H, W), dtype=np.float32)
    count = np.zeros((H, W),    dtype=np.float32)

    # height_max via vectorized scatter
    np.maximum.at(bev[0], (xi, yi), pts[:, 2])

    # height_sum (-> mean later)
    np.add.at(bev[1], (xi, yi), pts[:, 2])

    # density count
    np.add.at(count, (xi, yi), 1.0)

    # intensity_max
    intensity = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))
    np.maximum.at(bev[3], (xi, yi), intensity)

    # Finalize
    nonzero = count > 0
    bev[1][nonzero] /= count[nonzero]   # height_mean
    bev[2] = np.log1p(count)             # log_density

    # Normalize each channel to [-1, 1]
    for c in range(4):
        ch = bev[c]
        lo, hi = ch.min(), ch.max()
        if hi - lo > 1e-6:
            bev[c] = 2.0 * (ch - lo) / (hi - lo) - 1.0

    return bev


def compute_ego_action(pose_curr: dict, pose_next: dict) -> np.ndarray:
    """
    Ego-motion action giua 2 poses lien tiep.
    Returns: (dx, dy, dyaw) trong ego frame hien tai.
    """
    t_curr = np.array(pose_curr['translation'])
    t_next = np.array(pose_next['translation'])
    q_curr = Quaternion(pose_curr['rotation'])
    q_next = Quaternion(pose_next['rotation'])

    dt_global = t_next - t_curr
    dt_ego    = q_curr.inverse.rotate(dt_global)

    q_delta = q_curr.inverse * q_next
    dyaw    = q_delta.yaw_pitch_roll[0]

    return np.array([dt_ego[0], dt_ego[1], dyaw], dtype=np.float32)


# -----------------------------------------------------------------------
# Main preprocessing pipeline
# -----------------------------------------------------------------------

def process_nuscenes(nuscenes_root: str, version: str, output_path: str,
                     bev_size: int = 256, resolution: float = 0.4,
                     seq_length: int = 16, stride: int = None):
    """
    nuScenes -> HDF5 voi BEV sequences.

    stride: buoc truot khi slice sequences (default = seq_length // 2)
    """
    from nuscenes.nuscenes import NuScenes

    if stride is None:
        stride = seq_length // 2

    half = bev_size * resolution / 2.0
    x_range = (-half, half)
    y_range = (-half, half)

    print(f"Loading nuScenes {version} from {nuscenes_root}...")
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=True)

    all_obs, all_act = [], []
    skipped = 0

    for scene_idx, scene in enumerate(nusc.scene):
        print(f"\nScene {scene_idx+1}/{len(nusc.scene)}: {scene['name']}", flush=True)

        # Collect samples in scene
        samples = []
        token = scene['first_sample_token']
        while token:
            s     = nusc.get('sample', token)
            samples.append(s)
            token = s['next'] if s['next'] else None

        if len(samples) < seq_length:
            print(f"  Skip: only {len(samples)} samples (need {seq_length})")
            skipped += 1
            continue

        # Load BEV + ego poses for all samples
        bev_frames, ego_poses = [], []
        for sample in samples:
            lidar_tok  = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_tok)
            lidar_path = str(Path(nuscenes_root) / lidar_data['filename'])

            pts = load_pointcloud(lidar_path)
            bev = pointcloud_to_bev(pts, x_range, y_range, resolution=resolution)
            bev_frames.append(bev)

            ego = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            ego_poses.append(ego)

        # Slice into overlapping sequences
        n_seqs_before = len(all_obs)
        for start in range(0, len(bev_frames) - seq_length + 1, stride):
            end = start + seq_length
            obs_seq = np.stack(bev_frames[start:end])          # (T, 4, H, W)
            act_seq = np.stack([
                compute_ego_action(ego_poses[t], ego_poses[t+1])
                for t in range(start, end - 1)
            ])                                                  # (T-1, 3)
            all_obs.append(obs_seq)
            all_act.append(act_seq)

        print(f"  -> {len(all_obs) - n_seqs_before} sequences extracted")

    if len(all_obs) == 0:
        print("ERROR: No sequences extracted. Check data path.")
        sys.exit(1)

    observations = np.stack(all_obs)   # (N, T, 4, H, W)
    actions      = np.stack(all_act)   # (N, T-1, 3)

    print(f"\n{'='*50}")
    print(f"Total sequences:  {observations.shape[0]}")
    print(f"observations:     {observations.shape}")
    print(f"actions:          {actions.shape}")
    print(f"Scenes skipped:   {skipped}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, 'w') as f:
        f.create_dataset('observations', data=observations,
                         compression='gzip', compression_opts=4)
        f.create_dataset('actions', data=actions,
                         compression='gzip', compression_opts=4)
        f.attrs['bev_size']   = bev_size
        f.attrs['resolution'] = resolution
        f.attrs['seq_length'] = seq_length
        f.attrs['x_range']    = x_range
        f.attrs['y_range']    = y_range
        f.attrs['version']    = version

    mb = out.stat().st_size / 1e6
    print(f"Saved: {out}  ({mb:.1f} MB)")
    print(f"{'='*50}\n")


# -----------------------------------------------------------------------
# Unit test: BEV function (chay duoc ma khong can nuScenes data)
# -----------------------------------------------------------------------

def run_unit_tests():
    print("Running unit tests for BEV conversion...")

    # Test 1: empty cloud
    pts_empty = np.zeros((0, 5), dtype=np.float32)
    bev = pointcloud_to_bev(pts_empty, resolution=0.4)
    assert bev.shape == (4, 256, 256), f"Bad shape: {bev.shape}"
    assert bev.sum() == 0.0
    print("  [PASS] empty point cloud")

    # Test 2: single point
    pts_single = np.array([[0.0, 0.0, 1.0, 0.5, 0.0]], dtype=np.float32)
    bev = pointcloud_to_bev(pts_single, resolution=0.4)
    assert bev.shape == (4, 256, 256)
    print("  [PASS] single point")

    # Test 3: dense cloud, output range [-1, 1]
    np.random.seed(0)
    pts_dense = np.random.uniform(-40, 40, (10000, 5)).astype(np.float32)
    bev = pointcloud_to_bev(pts_dense, resolution=0.4)
    assert bev.shape == (4, 256, 256)
    for c in range(4):
        ch = bev[c]
        if ch.max() - ch.min() > 1e-6:
            assert ch.min() >= -1.01 and ch.max() <= 1.01, \
                f"Channel {c} out of range: [{ch.min():.3f}, {ch.max():.3f}]"
    print("  [PASS] dense cloud, value range [-1, 1]")

    # Test 4: ego action (identity)
    pose_a = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
    pose_b = {'translation': [1, 0, 0], 'rotation': [1, 0, 0, 0]}
    act = compute_ego_action(pose_a, pose_b)
    assert act.shape == (3,)
    assert abs(act[0] - 1.0) < 1e-5, f"dx expected 1.0, got {act[0]}"
    assert abs(act[1]) < 1e-5
    assert abs(act[2]) < 1e-5
    print("  [PASS] ego action (straight ahead)")

    print("All unit tests PASSED\n")


# -----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess nuScenes -> BEV HDF5")
    p.add_argument("--nuscenes_root", type=str, default="data/nuscenes")
    p.add_argument("--version",       type=str, default="v1.0-mini",
                   choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"])
    p.add_argument("--output",        type=str,
                   default="data/processed/nuscenes_mini_bev.h5")
    p.add_argument("--bev_size",      type=int,   default=256)
    p.add_argument("--resolution",    type=float, default=0.4)
    p.add_argument("--seq_length",    type=int,   default=16)
    p.add_argument("--stride",        type=int,   default=None,
                   help="Slice stride (default = seq_length // 2)")
    p.add_argument("--test",          action="store_true",
                   help="Chi chay unit tests, khong can data")
    args = p.parse_args()

    if args.test:
        run_unit_tests()
    else:
        process_nuscenes(
            args.nuscenes_root, args.version, args.output,
            args.bev_size, args.resolution, args.seq_length, args.stride
        )
