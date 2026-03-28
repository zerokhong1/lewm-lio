"""
encoder_bev.py — Lightweight BEV encoder cho LeWM-LiDAR

Mục tiêu: BEV image (H, W, C) → z_t (latent_dim,)
Constraint: ~5-10M params, chạy real-time
"""
import torch
import torch.nn as nn


class BEVEncoder(nn.Module):
    """
    Simple CNN encoder: BEV → CLS token.
    Lấy cảm hứng từ LeWM encoder nhưng input là BEV thay vì RGB.
    """
    def __init__(self, in_channels=4, latent_dim=192, base_channels=32):
        super().__init__()
        self.backbone = nn.Sequential(
            # (B, 4, H, W) → (B, 32, H/2, W/2)
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # → (B, 64, H/4, W/4)
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # → (B, 128, H/8, W/8)
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # → (B, 256, H/16, W/16)
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection → latent dim (with BN, critical for LeWM stability)
        self.projector = nn.Sequential(
            nn.Linear(base_channels * 8, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, x):
        """
        x: (B, C, H, W) — BEV image
        returns: (B, latent_dim) — latent embedding
        """
        feat = self.backbone(x)             # (B, 256, h, w)
        feat = self.global_pool(feat)       # (B, 256, 1, 1)
        feat = feat.flatten(1)              # (B, 256)
        z = self.projector(feat)            # (B, latent_dim)
        return z
