"""
data_lidar.py — HDF5 Dataset cho LiDAR BEV sequences

Format HDF5 (tương thích LeWM):
    observations: (num_episodes, seq_len, C, H, W)  — BEV images
    actions:      (num_episodes, seq_len, action_dim) — ego-motion
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path


class LiDARBEVDataset(Dataset):
    """
    Dataset load BEV sequences từ HDF5.
    Mỗi sample = 1 subsequence (obs_{t:t+T}, act_{t:t+T-1}).
    """
    def __init__(self, h5_path: str, seq_length: int = 16):
        self.h5_path = Path(h5_path)
        self.seq_length = seq_length

        with h5py.File(self.h5_path, 'r') as f:
            self.num_episodes = f['observations'].shape[0]
            self.episode_length = f['observations'].shape[1]
            self.obs_shape = f['observations'].shape[2:]   # (C, H, W)
            self.action_dim = f['actions'].shape[2]

        self.samples_per_episode = max(0, self.episode_length - self.seq_length)
        self.total_samples = self.num_episodes * self.samples_per_episode

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        ep_idx = idx // self.samples_per_episode
        t_start = idx % self.samples_per_episode

        with h5py.File(self.h5_path, 'r') as f:
            obs = f['observations'][
                ep_idx, t_start:t_start + self.seq_length
            ]  # (T, C, H, W)
            act = f['actions'][
                ep_idx, t_start:t_start + self.seq_length - 1
            ]  # (T-1, action_dim)

        return {
            'observations': torch.from_numpy(np.array(obs)).float(),
            'actions':      torch.from_numpy(np.array(act)).float(),
        }


class DummyLiDARBEVDataset(Dataset):
    """
    Dataset giả để test pipeline trên CONT mà không cần data thật.
    Sinh random BEV images + random actions.
    """
    def __init__(self, num_samples: int = 1000, seq_length: int = 16,
                 bev_channels: int = 4, bev_size: int = 64,
                 action_dim: int = 3):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.bev_channels = bev_channels
        self.bev_size = bev_size
        self.action_dim = action_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = torch.randn(
            self.seq_length, self.bev_channels,
            self.bev_size, self.bev_size
        )
        act = torch.randn(self.seq_length - 1, self.action_dim)
        return {'observations': obs, 'actions': act}
