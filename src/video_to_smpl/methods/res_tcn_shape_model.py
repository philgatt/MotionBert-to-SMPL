"""Figure 3.4 temporal Res-TCN for ten SMPL shape coefficients."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    """BatchNorm → ReLU → dropout → temporal convolution, with a residual skip."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TransitionResidualTemporalBlock(nn.Module):
    """Residual temporal block that changes the feature width."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 dropout: float = 0.5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels), nn.ReLU(inplace=True), nn.Dropout(float(dropout)),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class StagedResTCNShapeRegressor(nn.Module):
    """Released 51 → 128/192/256 → 128 → 10 shape regressor."""

    def __init__(self, num_joints: int = 17, in_dim: int = 3, kernel_size: int = 5,
                 dropout: float = 0.5, head_dropout: float = 0.0,
                 mlp_hidden_dim: int = 128, pooling: str = "avg", num_betas: int = 10):
        super().__init__()
        if pooling != "avg":
            raise ValueError("The released Figure 3.4 checkpoint uses average pooling.")
        self.input_conv = nn.Conv1d(int(num_joints) * int(in_dim), 128,
                                    kernel_size=kernel_size, padding=kernel_size // 2)
        self.stage1 = nn.Sequential(ResidualTemporalBlock(128, kernel_size, dropout),
                                    ResidualTemporalBlock(128, kernel_size, dropout))
        self.transition1 = TransitionResidualTemporalBlock(128, 192, kernel_size, dropout)
        self.stage2 = nn.Sequential(ResidualTemporalBlock(192, kernel_size, dropout),
                                    ResidualTemporalBlock(192, kernel_size, dropout))
        self.transition2 = TransitionResidualTemporalBlock(192, 256, kernel_size, dropout)
        self.stage3 = nn.Sequential(ResidualTemporalBlock(256, kernel_size, dropout),
                                    ResidualTemporalBlock(256, kernel_size, dropout))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(256, int(mlp_hidden_dim)), nn.ReLU(inplace=True),
                                  nn.Dropout(float(head_dropout)), nn.Linear(int(mlp_hidden_dim), int(num_betas)))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, dims = coords.shape
        x = coords.reshape(batch, frames, joints * dims).transpose(1, 2)
        x = self.stage1(self.input_conv(x))
        x = self.stage2(self.transition1(x))
        x = self.stage3(self.transition2(x))
        return self.head(self.pool(x).squeeze(-1))
