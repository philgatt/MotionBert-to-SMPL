"""Pose-only SMPLR adaptation shown in Figure 3.3.

The network takes root-centred H36M/MotionBERT joints, derives sixteen
unit bone directions (48 values), and predicts 24 raw 3x3 pose matrices.
It deliberately has no shape branch: beta is always the neutral zero vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
_EDGES = [(parent, child) for child, parent in enumerate(_PARENTS) if parent >= 0]


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues conversion used by the pose-training loss."""
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / angle
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    skew = skew.reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).view(*((1,) * (axis.dim() - 1)), 3, 3)
    return eye + torch.sin(angle)[..., None] * skew + (1.0 - torch.cos(angle))[..., None] * (skew @ skew)


class PoseOnlySMPLRRegressor(nn.Module):
    """48 -> 1024 -> 1024 -> 512 -> 512 -> 512 dual-head BN regressor."""

    architecture = "pose_only_smplr_bn_v1"

    def __init__(self, activation: str = "relu") -> None:
        super().__init__()
        self.activation = str(activation).lower()
        if self.activation not in {"relu", "elu"}:
            raise ValueError("activation must be 'relu' or 'elu'")
        self.fc1, self.bn1 = nn.Linear(48, 1024), nn.BatchNorm1d(1024)
        self.fc2, self.bn2 = nn.Linear(1024, 1024), nn.BatchNorm1d(1024)
        self.fc3, self.bn3 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc4, self.bn4 = nn.Linear(512, 512), nn.BatchNorm1d(512)
        self.fc41, self.bn41 = nn.Linear(512, 512), nn.BatchNorm1d(512)
        self.fc42, self.bn42 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc5, self.bn5 = nn.Linear(256, 256), nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 24 * 9)
        self.fc43, self.bn43 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.offset = nn.Linear(256, 24 * 9)
        self.combine = nn.Linear(2 * 24 * 9, 24 * 9)

    def _block(self, x: torch.Tensor, fc: nn.Linear, bn: nn.BatchNorm1d) -> torch.Tensor:
        value = bn(fc(x))
        return torch.nn.functional.elu(value) if self.activation == "elu" else torch.relu(value)

    @staticmethod
    def bone_directions(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.reshape(-1, 17, 3)
        x = x - x[:, :1]
        bones = torch.stack([x[:, child] - x[:, parent] for parent, child in _EDGES], dim=1)
        directions = bones / torch.linalg.norm(bones, dim=-1, keepdim=True).clamp_min(1e-8)
        return directions.reshape(x.shape[0], 48)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        features = self.bone_directions(x_flat)
        p1 = self._block(features, self.fc1, self.bn1)
        p2 = self._block(p1, self.fc2, self.bn2)
        p3 = self._block(p2, self.fc3, self.bn3)
        p4 = self._block(p3, self.fc4, self.bn4)
        p41 = self._block(p4, self.fc41, self.bn41)
        main = self._block(p41, self.fc42, self.bn42)
        main = self._block(main, self.fc5, self.bn5)
        main = self.out(main)
        residual = self._block(p41, self.fc43, self.bn43)
        residual = self.offset(residual)
        return torch.tanh(self.combine(torch.cat([main, residual], dim=-1))).view(-1, 24, 3, 3)
