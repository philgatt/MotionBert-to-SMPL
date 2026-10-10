from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


_H36M17_PARENTS: List[int] = [
    -1,  # 0 pelvis
    0,   # 1 right_hip
    1,   # 2 right_knee
    2,   # 3 right_ankle
    0,   # 4 left_hip
    4,   # 5 left_knee
    5,   # 6 left_ankle
    0,   # 7 spine
    7,   # 8 thorax
    8,   # 9 neck
    9,   # 10 head
    8,   # 11 left_shoulder
    11,  # 12 left_elbow
    12,  # 13 left_wrist
    8,   # 14 right_shoulder
    14,  # 15 right_elbow
    15,  # 16 right_wrist
]


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(eps)
    axis = axis_angle / angle
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    k = torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ],
        dim=-1,
    ).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).view(
        *((1,) * (axis.dim() - 1)), 3, 3
    )
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return eye + sin * k + (1.0 - cos) * (k @ k)


def matrix_to_axis_angle(rot_mats: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    tr = rot_mats[..., 0, 0] + rot_mats[..., 1, 1] + rot_mats[..., 2, 2]
    cos_theta = torch.clamp((tr - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    wx = rot_mats[..., 2, 1] - rot_mats[..., 1, 2]
    wy = rot_mats[..., 0, 2] - rot_mats[..., 2, 0]
    wz = rot_mats[..., 1, 0] - rot_mats[..., 0, 1]
    w = torch.stack([wx, wy, wz], dim=-1)

    scale = 1.0 / (2.0 * sin_theta.unsqueeze(-1) + eps)
    axis = w * scale
    small = theta < 1e-4
    if torch.any(small):
        axis_small = 0.5 * w
        axis = torch.where(small.unsqueeze(-1), axis_small, axis)
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + eps)
    return axis * theta.unsqueeze(-1)


@dataclass
class SMPLRStyleOutput:
    pred_beta: torch.Tensor
    pred_theta_mats_raw: torch.Tensor
    pred_theta_mats: torch.Tensor
    pred_theta_aa: Optional[torch.Tensor]
    pose_features_n: torch.Tensor
    shape_features_b: torch.Tensor


class SMPLRStyleRegressor(nn.Module):
    """
    SMPLR-style reverse stage adapted to H36M-17:
    - pose branch R = Omega(N; phi_p) from normalized relative bones N
    - shape branch beta = Psi(B; phi_s) from relative bone offsets B
    """

    def __init__(self, dropout: float = 0.0, shape_feature_mode: str = "bone_offsets"):
        super().__init__()
        shape_feature_mode = str(shape_feature_mode).strip().lower()
        if shape_feature_mode not in {"bone_offsets", "coords", "pose_only_zero_beta"}:
            raise ValueError(
                f"Unsupported shape_feature_mode={shape_feature_mode!r}. "
                "Expected 'bone_offsets', 'coords', or 'pose_only_zero_beta'."
            )
        self.parents = _H36M17_PARENTS
        self.edges = [(p, i) for i, p in enumerate(self.parents) if p >= 0]
        self.n_bones = len(self.edges)  # 16 for H36M-17
        self.shape_feature_mode = shape_feature_mode
        self.predict_shape = self.shape_feature_mode != "pose_only_zero_beta"
        self.shape_feature_dim = self.n_bones if self.shape_feature_mode == "bone_offsets" else 17 * 3
        self.register_buffer("bone_template", torch.zeros(self.n_bones, dtype=torch.float32))
        self.dropout = float(dropout)

        # Shape branch: B (16) -> beta (10), with residual-style dual heads.
        if self.predict_shape:
            self.shape_fc1 = nn.Linear(self.shape_feature_dim, 1024)
            self.shape_bn1 = nn.BatchNorm1d(1024)
            self.shape_fc2 = nn.Linear(1024, 1024)
            self.shape_bn2 = nn.BatchNorm1d(1024)
            self.shape_fc3 = nn.Linear(1024, 512)
            self.shape_bn3 = nn.BatchNorm1d(512)
            self.shape_fc4 = nn.Linear(512, 512)
            self.shape_bn4 = nn.BatchNorm1d(512)
            self.shape_fc41 = nn.Linear(512, 512)
            self.shape_bn41 = nn.BatchNorm1d(512)
            self.shape_fc42 = nn.Linear(512, 256)
            self.shape_bn42 = nn.BatchNorm1d(256)
            self.shape_fc5 = nn.Linear(256, 256)
            self.shape_bn5 = nn.BatchNorm1d(256)
            self.shape_out = nn.Linear(256, 10)
            self.shape_fc43 = nn.Linear(512, 256)
            self.shape_bn43 = nn.BatchNorm1d(256)
            self.shape_offset = nn.Linear(256, 10)
            self.shape_out2 = nn.Linear(20, 10)

        # Pose branch: N (16*3=48) -> rotmats (24*9), with residual-style dual heads.
        self.pose_fc1 = nn.Linear(self.n_bones * 3, 1024)
        self.pose_bn1 = nn.BatchNorm1d(1024)
        self.pose_fc2 = nn.Linear(1024, 1024)
        self.pose_bn2 = nn.BatchNorm1d(1024)
        self.pose_fc3 = nn.Linear(1024, 512)
        self.pose_bn3 = nn.BatchNorm1d(512)
        self.pose_fc4 = nn.Linear(512, 512)
        self.pose_bn4 = nn.BatchNorm1d(512)
        self.pose_fc41 = nn.Linear(512, 512)
        self.pose_bn41 = nn.BatchNorm1d(512)
        self.pose_fc42 = nn.Linear(512, 256)
        self.pose_bn42 = nn.BatchNorm1d(256)
        self.pose_fc5 = nn.Linear(256, 256)
        self.pose_bn5 = nn.BatchNorm1d(256)
        self.pose_out = nn.Linear(256, 24 * 9)
        self.pose_fc43 = nn.Linear(512, 256)
        self.pose_bn43 = nn.BatchNorm1d(256)
        self.pose_offset = nn.Linear(256, 24 * 9)
        self.pose_out2 = nn.Linear(2 * 24 * 9, 24 * 9)

    def set_bone_template(self, template: torch.Tensor) -> None:
        if template.numel() != self.n_bones:
            raise ValueError(f"bone_template has {template.numel()} values, expected {self.n_bones}")
        self.bone_template.copy_(template.detach().to(dtype=torch.float32, device=self.bone_template.device))

    @staticmethod
    def _elu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x)

    def _extract_nb(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_flat.view(-1, 17, 3)
        x = x - x[:, :1, :]

        rel = []
        for p, c in self.edges:
            rel.append(x[:, c, :] - x[:, p, :])
        rel = torch.stack(rel, dim=1)  # (B,16,3)

        lens = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)  # (B,16)
        n = rel / lens.unsqueeze(-1)  # normalized relative bones
        b = lens - self.bone_template.unsqueeze(0)  # relative distance offsets
        if self.shape_feature_mode == "coords":
            b = x.reshape(x.shape[0], -1)
        return n.reshape(x.shape[0], -1), b

    @staticmethod
    def _elu_bn(x: torch.Tensor, fc: nn.Linear, bn: nn.BatchNorm1d) -> torch.Tensor:
        return torch.nn.functional.elu(bn(fc(x)))

    def forward(self, x_flat: torch.Tensor) -> SMPLRStyleOutput:
        n_feat, b_feat = self._extract_nb(x_flat)

        if self.predict_shape:
            # Shape branch
            s1 = self._elu_bn(b_feat, self.shape_fc1, self.shape_bn1)
            s2 = self._elu_bn(s1, self.shape_fc2, self.shape_bn2)
            s3 = self._elu_bn(s2, self.shape_fc3, self.shape_bn3)
            s4 = self._elu_bn(s3, self.shape_fc4, self.shape_bn4)
            s41 = self._elu_bn(s4, self.shape_fc41, self.shape_bn41)
            s42 = self._elu_bn(s41, self.shape_fc42, self.shape_bn42)
            s5 = self._elu_bn(s42, self.shape_fc5, self.shape_bn5)
            shape_main = self.shape_out(s5)
            s43 = self._elu_bn(s41, self.shape_fc43, self.shape_bn43)
            shape_offset = self.shape_offset(s43)
            pred_beta = self.shape_out2(torch.cat([shape_main, shape_offset], dim=-1))
        else:
            pred_beta = torch.zeros(
                (x_flat.shape[0], 10),
                dtype=x_flat.dtype,
                device=x_flat.device,
            )

        # Pose branch
        p1 = self._elu_bn(n_feat, self.pose_fc1, self.pose_bn1)
        p2 = self._elu_bn(p1, self.pose_fc2, self.pose_bn2)
        p3 = self._elu_bn(p2, self.pose_fc3, self.pose_bn3)
        p4 = self._elu_bn(p3, self.pose_fc4, self.pose_bn4)
        p41 = self._elu_bn(p4, self.pose_fc41, self.pose_bn41)
        p42 = self._elu_bn(p41, self.pose_fc42, self.pose_bn42)
        p5 = self._elu_bn(p42, self.pose_fc5, self.pose_bn5)
        pose_main = self.pose_out(p5)
        p43 = self._elu_bn(p41, self.pose_fc43, self.pose_bn43)
        pose_offset = self.pose_offset(p43)
        pose_raw = torch.tanh(self.pose_out2(torch.cat([pose_main, pose_offset], dim=-1)))

        pred_theta_mats_raw = pose_raw.view(-1, 24, 3, 3)
        pred_theta_mats = pred_theta_mats_raw
        pred_theta_aa = None

        return SMPLRStyleOutput(
            pred_beta=pred_beta,
            pred_theta_mats_raw=pred_theta_mats_raw,
            pred_theta_mats=pred_theta_mats,
            pred_theta_aa=pred_theta_aa,
            pose_features_n=n_feat,
            shape_features_b=b_feat,
        )


def smplr_style_loss(
    out: SMPLRStyleOutput,
    gt_theta_aa: torch.Tensor,
    gt_beta: torch.Tensor,
    pred_joints17: torch.Tensor,
    input_joints17: torch.Tensor,
    *,
    w_rotmat: float = 1.0,
    w_theta_aa: float = 0.0,
    w_beta: float = 1.0,
    w_joint: float = 0.0,
    w_beta_reg: float = 0.01,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    gt_rot = axis_angle_to_matrix(gt_theta_aa.view(-1, 24, 3))
    l_rot = torch.mean(torch.abs(torch.nan_to_num(out.pred_theta_mats_raw, nan=0.0) - gt_rot))
    if out.pred_theta_aa is None:
        l_theta_aa = gt_theta_aa.new_tensor(0.0)
    else:
        l_theta_aa = torch.mean(torch.abs(out.pred_theta_aa - gt_theta_aa))
    l_beta = torch.mean(torch.abs(out.pred_beta - gt_beta))
    l_joint = torch.mean(torch.abs(pred_joints17 - input_joints17))
    l_beta_reg = torch.mean(torch.abs(out.pred_beta))

    total = (
        w_rotmat * l_rot
        + w_theta_aa * l_theta_aa
        + w_beta * l_beta
        + w_joint * l_joint
        + w_beta_reg * l_beta_reg
    )
    parts = {
        "l_rotmat": l_rot,
        "l_theta_aa": l_theta_aa,
        "l_beta": l_beta,
        "l_joint": l_joint,
        "l_beta_reg": l_beta_reg,
    }
    return total, parts
