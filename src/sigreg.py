"""
sigreg.py — Sketched Isotropic Gaussian Regularization

Adapted từ: https://github.com/galilai-group/lejepa
Paper: LeJEPA (arXiv 2511.08544)

Chức năng: Ép latent embeddings → phân phối Gaussian isotropic
           bằng cách chiếu lên random directions rồi test từng chiều 1D.
"""
import torch
import torch.nn as nn


def epps_pulley_test(samples: torch.Tensor, num_points: int = 17) -> torch.Tensor:
    """
    Epps-Pulley test statistic cho normality trên 1D samples.
    So sánh empirical characteristic function vs Gaussian CF.

    Args:
        samples:    (N,) — 1D samples
        num_points: số điểm evaluate CF

    Returns:
        scalar — test statistic (càng nhỏ = càng Gaussian)
    """
    # Standardize
    samples = (samples - samples.mean()) / (samples.std() + 1e-8)

    # Evaluation points cho characteristic function
    t = torch.linspace(0.1, 2.0, num_points, device=samples.device)

    # Empirical CF: E[exp(i*t*X)]
    tx = t.unsqueeze(1) * samples.unsqueeze(0)  # (num_points, N)
    ecf_real = torch.cos(tx).mean(dim=1)         # (num_points,)
    ecf_imag = torch.sin(tx).mean(dim=1)         # (num_points,)

    # Gaussian CF: exp(-t²/2)  (purely real for symmetric dist)
    gcf = torch.exp(-0.5 * t ** 2)

    stat = (((ecf_real - gcf) ** 2) + (ecf_imag ** 2)).mean()
    return stat


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Chiếu embeddings lên M random directions (Cramér-Wold theorem),
    rồi dùng Epps-Pulley test trên mỗi projection 1D.

    Usage:
        sigreg = SIGReg(num_slices=1024, num_points=17)
        loss = sigreg(embeddings)  # embeddings: (B, D)
    """
    def __init__(self, num_slices: int = 1024, num_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.num_points = num_points

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) — batch of latent embeddings

        Returns:
            scalar — SIGReg loss (lower = more Gaussian)
        """
        B, D = z.shape

        # Random projection directions (unit vectors on sphere)
        directions = torch.randn(self.num_slices, D, device=z.device)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)

        # Project: (num_slices, D) × (D, B) → (num_slices, B)
        projections = directions @ z.T

        # Apply Epps-Pulley test on each slice
        total_stat = torch.tensor(0.0, device=z.device)
        for i in range(self.num_slices):
            total_stat = total_stat + epps_pulley_test(
                projections[i], self.num_points
            )

        return total_stat / self.num_slices
