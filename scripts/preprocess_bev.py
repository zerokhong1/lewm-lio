"""
preprocess_bev.py — Convert point cloud → BEV representation → HDF5

Chạy trên CONT, output copy sang TEA.
Input:  nuScenes/KITTI raw point clouds
Output: HDF5 file với BEV images + actions + metadata
"""
import numpy as np
import h5py
from pathlib import Path
import argparse


def pointcloud_to_bev(points, x_range=(-50, 50), y_range=(-50, 50),
                      z_range=(-3, 3), resolution=0.2):
    """
    Convert point cloud (N, 4) → BEV image (H, W, C).
    C = [height_max, height_mean, density, intensity]
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    H = int((x_max - x_min) / resolution)
    W = int((y_max - y_min) / resolution)

    # Filter points in range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    # Discretize
    xi = ((points[:, 0] - x_min) / resolution).astype(np.int32)
    yi = ((points[:, 1] - y_min) / resolution).astype(np.int32)
    xi = np.clip(xi, 0, H - 1)
    yi = np.clip(yi, 0, W - 1)

    # BEV channels
    bev = np.zeros((H, W, 4), dtype=np.float32)

    for i in range(len(points)):
        x_idx, y_idx = xi[i], yi[i]
        z_val = points[i, 2]
        intensity = points[i, 3] if points.shape[1] > 3 else 0

        bev[x_idx, y_idx, 0] = max(bev[x_idx, y_idx, 0], z_val)     # height_max
        bev[x_idx, y_idx, 1] += z_val                                 # height_sum (→ mean later)
        bev[x_idx, y_idx, 2] += 1                                     # density count
        bev[x_idx, y_idx, 3] = max(bev[x_idx, y_idx, 3], intensity)  # intensity_max

    # Normalize
    density = bev[:, :, 2]
    density_nonzero = density > 0
    bev[:, :, 1][density_nonzero] /= density[density_nonzero]  # height_mean
    bev[:, :, 2] = np.log1p(density)                           # log density

    return bev


def create_dataset_hdf5(data_dir, output_path, dataset_type="nuscenes"):
    """
    Tạo HDF5 dataset tương thích format LeWM.
    Format output HDF5 cần match LeWM:
      - observations: (N, T, C, H, W)  → BEV images
      - actions: (N, T, action_dim)     → ego-motion actions
    """
    # TODO: implement cho từng dataset type
    raise NotImplementedError(f"Dataset type '{dataset_type}' not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["nuscenes", "kitti"], default="nuscenes")
    parser.add_argument("--resolution", type=float, default=0.2)
    args = parser.parse_args()

    create_dataset_hdf5(args.data_dir, args.output, args.dataset)
    print(f"BEV dataset saved to: {args.output}")
