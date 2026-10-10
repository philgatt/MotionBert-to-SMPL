"""Checkpoint construction and overlapping temporal inference from the thesis."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from .res_tcn_shape_model import (MotionFeatureDilatedTCNShapeRegressor, PaperStyleDilatedResTCNShapeRegressor, PaperStyleResTCNShapeRegressor, ResTCNShapeRegressor, StagedResTCNShapeRegressor, STGCNShapeRegressor)

def _model_from_checkpoint(checkpoint: dict) -> ResTCNShapeRegressor:
    cargs = checkpoint.get("args", {})
    if not isinstance(cargs, dict):
        cargs = vars(cargs)
    if str(cargs.get("architecture", checkpoint.get("architecture", "simple"))) == "staged":
        model = StagedResTCNShapeRegressor(
            num_joints=17,
            in_dim=3,
            kernel_size=int(cargs.get("kernel_size", 3)),
            dropout=float(cargs.get("dropout", 0.5)),
            head_dropout=cargs.get("head_dropout", None),
            mlp_hidden_dim=int(cargs.get("mlp_hidden_dim", 128)),
            pooling=str(cargs.get("pooling", "avg")),
            num_betas=10,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    if str(cargs.get("architecture", checkpoint.get("architecture", "simple"))) == "paper_style":
        model = PaperStyleResTCNShapeRegressor(
            num_joints=17,
            in_dim=3,
            kernel_size=int(cargs.get("kernel_size", 8)),
            dropout=float(cargs.get("dropout", 0.5)),
            head_dropout=cargs.get("head_dropout", None),
            mlp_hidden_dim=int(cargs.get("mlp_hidden_dim", 128)),
            pooling=str(cargs.get("pooling", "avg")),
            num_betas=10,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    if str(cargs.get("architecture", checkpoint.get("architecture", "simple"))) == "paper_style_dilated":
        model = PaperStyleDilatedResTCNShapeRegressor(
            num_joints=17,
            in_dim=3,
            kernel_size=int(cargs.get("kernel_size", 8)),
            dropout=float(cargs.get("dropout", 0.5)),
            mlp_hidden_dim=int(cargs.get("mlp_hidden_dim", 128)),
            pooling=str(cargs.get("pooling", "avg")),
            num_betas=10,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    if str(cargs.get("architecture", checkpoint.get("architecture", "simple"))) == "motion_dilated":
        model = MotionFeatureDilatedTCNShapeRegressor(
            num_joints=17,
            in_dim=3,
            feature_dim=128,
            kernel_size=5,
            dilations=(1, 2, 4, 8, 16),
            dropout=float(cargs.get("dropout", 0.5)),
            num_betas=10,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    if str(cargs.get("architecture", checkpoint.get("architecture", "simple"))) == "stgcn":
        model = STGCNShapeRegressor(
            num_joints=17,
            in_dim=3,
            temporal_kernel_size=9,
            dropout=float(cargs.get("dropout", 0.5)),
            mlp_hidden_dim=int(cargs.get("mlp_hidden_dim", 128)),
            num_betas=10,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    model = ResTCNShapeRegressor(
        num_joints=17,
        in_dim=3,
        hidden_dim=int(cargs.get("hidden_dim", 256)),
        num_blocks=int(cargs.get("num_blocks", 8)),
        kernel_size=int(cargs.get("kernel_size", 3)),
        dropout=float(cargs.get("dropout", 0.5)),
        head_dropout=cargs.get("head_dropout", None),
        mlp_hidden_dim=int(cargs.get("mlp_hidden_dim", 128)),
        pooling=str(cargs.get("pooling", "avg")),
        num_betas=10,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def _contiguous_segments(frame_indices: np.ndarray) -> List[Tuple[int, int]]:
    if len(frame_indices) == 0:
        return []
    starts = [0]
    for idx in range(1, len(frame_indices)):
        if int(frame_indices[idx]) != int(frame_indices[idx - 1]) + 1:
            starts.append(idx)
    ends = starts[1:] + [len(frame_indices)]
    return list(zip(starts, ends))

def _make_windows(coords: np.ndarray, frame_indices: np.ndarray, clip_len: int, stride: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    windows: List[np.ndarray] = []
    covered: List[np.ndarray] = []
    for start, end in _contiguous_segments(frame_indices):
        segment_len = end - start
        if segment_len <= 0:
            continue
        if segment_len < clip_len:
            local = np.arange(start, end, dtype=np.int64)
            pad_count = clip_len - segment_len
            padded = np.concatenate([local, np.repeat(local[-1:], pad_count)])
            windows.append(coords[padded])
            covered.append(local)
            continue
        for local_start in range(start, end - clip_len + 1, stride):
            inds = np.arange(local_start, local_start + clip_len, dtype=np.int64)
            windows.append(coords[inds])
            covered.append(inds)
        if (end - clip_len - start) % stride != 0:
            inds = np.arange(end - clip_len, end, dtype=np.int64)
            windows.append(coords[inds])
            covered.append(inds)
    if not windows:
        return np.empty((0, clip_len, 17, 3), dtype=np.float32), []
    return np.stack(windows, axis=0).astype(np.float32), covered

def _predict_track(
    model: ResTCNShapeRegressor,
    coords: np.ndarray,
    frame_indices: np.ndarray,
    clip_len: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    clip_coordinate_mode: str = "as_is",
) -> np.ndarray:
    windows, covered = _make_windows(coords, frame_indices, clip_len=clip_len, stride=stride)
    if len(windows) == 0:
        raise RuntimeError("No windows created for track")
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device=device, dtype=torch.float32)
            if clip_coordinate_mode == "frame0_pelvis_origin":
                batch = batch - batch[:, :1, :1, :]
            preds.append(model(batch).detach().cpu().numpy().astype(np.float32))
    clip_betas = np.concatenate(preds, axis=0)
    beta_sum = np.zeros((len(coords), 10), dtype=np.float64)
    beta_count = np.zeros((len(coords), 1), dtype=np.float64)
    for beta, inds in zip(clip_betas, covered):
        beta_sum[inds] += beta[None, :]
        beta_count[inds] += 1.0
    uncovered = beta_count[:, 0] <= 0
    if np.any(uncovered):
        fallback = clip_betas.mean(axis=0)
        beta_sum[uncovered] = fallback[None, :]
        beta_count[uncovered] = 1.0
    return (beta_sum / beta_count).astype(np.float32)
