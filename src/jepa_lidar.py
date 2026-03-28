"""
jepa_lidar.py — LeWM adapted cho LiDAR BEV input

Architecture:
    Encoder:   BEV (C, H, W) → z_t (latent_dim,)
    Predictor: (z_t, a_t)    → ẑ_{t+1}

Training:
    L = L_pred + λ * SIGReg(Z)
"""
import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    MLP predictor: (z_t, a_t) → ẑ_{t+1}
    Dropout 0.1 là critical cho stability (từ LeWM).
    """
    def __init__(self, latent_dim: int = 192, action_dim: int = 3,
                 hidden_dim: int = 512, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        input_dim = latent_dim + action_dim
        layers = []
        for i in range(num_layers):
            in_d  = input_dim  if i == 0            else hidden_dim
            out_d = latent_dim if i == num_layers-1  else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.extend([nn.ReLU(inplace=True), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        a: (B, action_dim)
        returns: (B, latent_dim)
        """
        return self.net(torch.cat([z, a], dim=-1))


class LeWMLiDAR(nn.Module):
    """
    LeWM-LiDAR: JEPA world model cho LiDAR BEV.

    Forward pass (training):
        Cho sequence (o_1..o_T, a_1..a_{T-1}):
        1. Encode tất cả frames: z_t = encoder(o_t)
        2. Predict step-by-step: ẑ_{t+1} = predictor(z_t, a_t)
        3. Loss = MSE(ẑ_{t+1}, z_{t+1}.detach()) + λ * SIGReg(Z)
                                         ^^^^^^^^
                          stop gradient vào target (teacher forcing)
    """
    def __init__(self, encoder: nn.Module, predictor: nn.Module):
        super().__init__()
        self.encoder  = encoder
        self.predictor = predictor

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) → (B, latent_dim)"""
        return self.encoder(obs)

    def predict_next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z: (B, D), a: (B, A) → (B, D)"""
        return self.predictor(z, a)

    def forward_sequence(self, observations: torch.Tensor,
                         actions: torch.Tensor):
        """
        Training forward pass.

        observations: (B, T, C, H, W)
        actions:      (B, T-1, action_dim)

        Returns:
            pred_embeddings:   (B, T-1, latent_dim) — ẑ_{t+1} predictions
            target_embeddings: (B, T-1, latent_dim) — z_{t+1} targets (detached)
            all_embeddings:    (B, T,   latent_dim) — cho SIGReg
        """
        B, T, C, H, W = observations.shape

        # Encode all frames in one batched call
        obs_flat = observations.reshape(B * T, C, H, W)
        z_flat   = self.encode(obs_flat)             # (B*T, latent_dim)
        z_all    = z_flat.reshape(B, T, -1)          # (B, T, latent_dim)

        # Step-by-step prediction (teacher forcing)
        z_preds = []
        for t in range(T - 1):
            z_pred = self.predict_next(z_all[:, t], actions[:, t])
            z_preds.append(z_pred)

        pred_embeddings   = torch.stack(z_preds, dim=1)   # (B, T-1, D)
        target_embeddings = z_all[:, 1:].detach()          # (B, T-1, D)

        return pred_embeddings, target_embeddings, z_all
