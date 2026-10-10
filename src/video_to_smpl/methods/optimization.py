"""Original root-centered 3D SMPL fitting objective."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

_H36M_BONE_LENGTH_PRIORS_M = [
    (0, 1, 0.11),  # pelvis-rhip
    (0, 4, 0.11),  # pelvis-lhip
    (1, 2, 0.43),  # rhip-rknee
    (2, 3, 0.43),  # rknee-rankle
    (4, 5, 0.43),  # lhip-lknee
    (5, 6, 0.43),  # lknee-lankle
    (0, 7, 0.26),  # pelvis-belly
    (7, 8, 0.26),  # belly-neck
    (8, 11, 0.18),  # neck-lshoulder
    (8, 14, 0.18),  # neck-rshoulder
    (11, 12, 0.27),  # lshoulder-lelbow
    (12, 13, 0.26),  # lelbow-lwrist
    (14, 15, 0.27),  # rshoulder-relbow
    (15, 16, 0.26),  # relbow-rwrist
]

_H36M17_TO_SMPL24_SUBSET = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16], dtype=np.int64)

_SMPL24_SUBSET_FROM_H36M17 = np.asarray([0, 2, 5, 8, 1, 4, 7, 3, 12, 16, 18, 20, 17, 19, 21], dtype=np.int64)

_POSE_TO_SMPL_H36M17_AXIS_SCALE = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

def _compute_sequence_scale_to_smpl_rest(target_h36m: np.ndarray, smpl_rest_h36m17: np.ndarray) -> float:
    target = np.asarray(target_h36m, dtype=np.float64)
    rest = np.asarray(smpl_rest_h36m17, dtype=np.float64)
    if target.ndim != 3 or target.shape[1] < 17 or target.shape[2] < 3:
        return 1.0
    if rest.shape != (17, 3):
        return 1.0
    target_rel = target[:, :17, :3] - target[:, :1, :3]
    rest_rel = rest[None, :, :3] - rest[None, :1, :3]
    try:
        target_len = _compute_h36m_bone_lengths_per_frame(target_rel)
        rest_len = _compute_h36m_bone_lengths_per_frame(rest_rel)[0]
    except Exception:
        return 1.0
    ratios: List[float] = []
    for bone_idx in range(target_len.shape[1]):
        denom = target_len[:, bone_idx]
        numer = float(rest_len[bone_idx])
        valid = np.isfinite(denom) & (denom > 1e-8) & np.isfinite(numer) & (numer > 1e-8)
        if not np.any(valid):
            continue
        ratios.append(float(np.median(numer / denom[valid])))
    if not ratios:
        return 1.0
    scale = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale

def _run_pose_to_smpl_fit_from_x3d(
    data_x3d: np.ndarray,
    *,
    smpl_model,
    device: torch.device,
    iters: int,
    lr: float,
    optimize_shape: bool,
    optimize_scale: bool,
    fit_loss: str,
    pre_scale_to_rest: bool,
    smpl_rest_h36m17: Optional[np.ndarray],
    motionbert_rx180_flip_input: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    x3d = np.asarray(data_x3d, dtype=np.float64)
    if x3d.ndim != 3 or x3d.shape[2] < 3:
        raise ValueError(f"Unexpected 3D keypoints shape for pose_to_smpl_fit: {x3d.shape}")
    if x3d.shape[1] < 17:
        raise ValueError(f"pose_to_smpl_fit expects at least 17 joints, got {x3d.shape[1]}")

    root_abs = np.asarray(x3d[:, 0, :3], dtype=np.float64).copy()
    target_h36m = np.asarray(x3d[:, :17, :3], dtype=np.float64).copy()
    finite_mask = np.isfinite(target_h36m).all(axis=2)
    target_h36m[~finite_mask] = 0.0
    pre_scale_factor = 1.0
    if bool(pre_scale_to_rest) and (smpl_rest_h36m17 is not None):
        pre_scale_factor = float(_compute_sequence_scale_to_smpl_rest(target_h36m, np.asarray(smpl_rest_h36m17, dtype=np.float64)))
        if np.isfinite(pre_scale_factor) and pre_scale_factor > 0.0:
            target_h36m = target_h36m * pre_scale_factor
            root_abs = root_abs * pre_scale_factor
        else:
            pre_scale_factor = 1.0
    target_h36m = target_h36m - target_h36m[:, :1, :]
    target_h36m = target_h36m * _POSE_TO_SMPL_H36M17_AXIS_SCALE[None, None, :]
    n_frames = int(target_h36m.shape[0])
    if n_frames <= 0:
        raise ValueError("pose_to_smpl_fit received empty sequence.")

    target_t = torch.tensor(target_h36m, dtype=torch.double, device=device)
    init_pose = np.zeros((n_frames, 72), dtype=np.float64)
    if not bool(motionbert_rx180_flip_input):
        init_pose[:, 0] = np.pi
    pose_t = torch.nn.Parameter(torch.tensor(init_pose, dtype=torch.double, device=device))
    betas_t = torch.nn.Parameter(torch.zeros((10,), dtype=torch.double, device=device), requires_grad=bool(optimize_shape))
    log_scale_t = torch.nn.Parameter(torch.zeros((1,), dtype=torch.double, device=device), requires_grad=bool(optimize_scale))
    smpl24_reg_t = smpl_model.J_regressor.to(device=device, dtype=torch.double)
    h36m17_reg_t = smpl_model.J_regressor_h36m.to(device=device, dtype=torch.double)
    smpl_subset_idx_t = torch.tensor(_SMPL24_SUBSET_FROM_H36M17.tolist(), dtype=torch.long, device=device)
    h36m_subset_idx_t = torch.tensor(_H36M17_TO_SMPL24_SUBSET.tolist(), dtype=torch.long, device=device)

    if bool(optimize_scale):
        with torch.no_grad():
            out0 = smpl_model(
                betas=betas_t.unsqueeze(0).expand(n_frames, -1),
                body_pose=pose_t[:, 3:],
                global_orient=pose_t[:, :3],
                pose2rot=True,
            )
            if str(fit_loss) == "mse_h36m17":
                pred0 = torch.einsum("jv,nvc->njc", h36m17_reg_t, out0.vertices)
                pred0 = pred0 - pred0[:, :1, :]
                target0 = target_t
            else:
                smpl24_0 = torch.einsum("jv,nvc->njc", smpl24_reg_t, out0.vertices)
                pred0 = smpl24_0.index_select(1, smpl_subset_idx_t)
                target0 = target_t.index_select(1, h36m_subset_idx_t)
            denom0 = torch.max(torch.abs(pred0))
            numer0 = torch.max(torch.abs(target0))
            if float(denom0.detach().cpu().item()) > 1e-12:
                init_scale = torch.clamp(numer0 / denom0, min=1e-8)
                log_scale_t.data[:] = torch.log(init_scale).reshape_as(log_scale_t.data)
    else:
        with torch.no_grad():
            log_scale_t.data.zero_()

    optim_params: List[Dict[str, object]] = [{"params": [pose_t], "lr": float(lr)}]
    if bool(optimize_shape):
        optim_params.append({"params": [betas_t], "lr": float(lr)})
    if bool(optimize_scale):
        optim_params.append({"params": [log_scale_t], "lr": float(lr) * 10.0})
    optimizer = torch.optim.Adam(optim_params)

    last_loss = float("nan")
    for _ in range(max(int(iters), 0)):
        out = smpl_model(
            betas=betas_t.unsqueeze(0).expand(n_frames, -1),
            body_pose=pose_t[:, 3:],
            global_orient=pose_t[:, :3],
            pose2rot=True,
        )
        scale_value = torch.exp(log_scale_t).view(1, 1, 1)
        if str(fit_loss) == "mse_h36m17":
            pred_h36m17 = scale_value * torch.einsum("jv,nvc->njc", h36m17_reg_t, out.vertices)
            pred_h36m17 = pred_h36m17 - pred_h36m17[:, :1, :]
            loss = F.mse_loss(pred_h36m17, target_t)
        else:
            smpl24 = torch.einsum("jv,nvc->njc", smpl24_reg_t, out.vertices)
            pred_subset = scale_value * smpl24.index_select(1, smpl_subset_idx_t)
            target_subset = target_t.index_select(1, h36m_subset_idx_t)
            loss = F.smooth_l1_loss(pred_subset, target_subset)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

    thetas_np = pose_t.detach().cpu().numpy().astype(np.float64)
    beta_np = betas_t.detach().cpu().numpy().astype(np.float64)
    betas_seq = np.repeat(beta_np[None, :], repeats=n_frames, axis=0)
    diag = {
        "backend": "pose_to_smpl_fit",
        "frames": float(n_frames),
        "iters": float(iters),
        "lr": float(lr),
        "final_loss": float(last_loss),
        "final_scale": float(torch.exp(log_scale_t).detach().cpu().item()),
        "optimize_shape": float(bool(optimize_shape)),
        "optimize_scale": float(bool(optimize_scale)),
        "fit_loss": str(fit_loss),
        "mapped_joint_count": float(17 if str(fit_loss) == "mse_h36m17" else len(_H36M17_TO_SMPL24_SUBSET)),
        "pre_scale_to_rest": float(bool(pre_scale_to_rest)),
        "pre_scale_factor": float(pre_scale_factor),
        "motionbert_rx180_flip_input": float(bool(motionbert_rx180_flip_input)),
        "init_global_orient": "identity_after_input_rx180" if bool(motionbert_rx180_flip_input) else "rx180_up_axis_match",
        "target_axis_scale": _POSE_TO_SMPL_H36M17_AXIS_SCALE.astype(float).tolist(),
    }
    return root_abs, thetas_np, betas_seq, diag

def _compute_h36m_bone_lengths_per_frame(joints17: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints17, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 17 or arr.shape[2] < 3:
        raise ValueError(f"Expected [T,17,3], got {arr.shape}")
    lengths = []
    for j0, j1, _ in _H36M_BONE_LENGTH_PRIORS_M:
        d = np.linalg.norm(arr[:, j0, :3] - arr[:, j1, :3], axis=1)
        lengths.append(d)
    return np.stack(lengths, axis=1)
