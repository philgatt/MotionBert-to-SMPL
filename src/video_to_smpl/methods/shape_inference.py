"""Load and apply the released staged temporal shape checkpoint."""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from .res_tcn_shape_model import StagedResTCNShapeRegressor


def _model_from_checkpoint(checkpoint: dict) -> StagedResTCNShapeRegressor:
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = vars(args)
    architecture = str(args.get("architecture", checkpoint.get("architecture", "staged")))
    if architecture != "staged":
        raise ValueError(f"Expected the released staged shape checkpoint, received {architecture!r}.")
    model = StagedResTCNShapeRegressor(
        kernel_size=int(args.get("kernel_size", 5)), dropout=float(args.get("dropout", 0.5)),
        head_dropout=float(args.get("head_dropout", 0.0)),
        mlp_hidden_dim=int(args.get("mlp_hidden_dim", 128)), pooling="avg", num_betas=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _contiguous_segments(indices: np.ndarray) -> List[Tuple[int, int]]:
    if not len(indices):
        return []
    starts = [0] + [i for i in range(1, len(indices)) if int(indices[i]) != int(indices[i - 1]) + 1]
    return list(zip(starts, starts[1:] + [len(indices)]))


def _make_windows(coords: np.ndarray, indices: np.ndarray, clip_len: int,
                  stride: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    windows, covered = [], []
    for start, end in _contiguous_segments(indices):
        if end - start < clip_len:
            local = np.arange(start, end, dtype=np.int64)
            padded = np.concatenate([local, np.repeat(local[-1:], clip_len - len(local))])
            windows.append(coords[padded]); covered.append(local)
            continue
        for offset in range(start, end - clip_len + 1, stride):
            local = np.arange(offset, offset + clip_len, dtype=np.int64)
            windows.append(coords[local]); covered.append(local)
        if (end - clip_len - start) % stride:
            local = np.arange(end - clip_len, end, dtype=np.int64)
            windows.append(coords[local]); covered.append(local)
    if not windows:
        return np.empty((0, clip_len, 17, 3), dtype=np.float32), []
    return np.stack(windows).astype(np.float32), covered


def _predict_track(model: StagedResTCNShapeRegressor, coords: np.ndarray, indices: np.ndarray,
                   clip_len: int, stride: int, batch_size: int, device: torch.device,
                   clip_coordinate_mode: str = "as_is") -> np.ndarray:
    if clip_coordinate_mode != "as_is":
        raise ValueError(f"Unsupported released shape coordinate mode: {clip_coordinate_mode}")
    windows, covered = _make_windows(coords, indices, clip_len, stride)
    if not len(windows):
        raise RuntimeError("No windows created for track")
    predictions = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device=device, dtype=torch.float32)
            predictions.append(model(batch).cpu().numpy().astype(np.float32))
    beta_sum = np.zeros((len(coords), 10), dtype=np.float64)
    beta_count = np.zeros((len(coords), 1), dtype=np.float64)
    for beta, local in zip(np.concatenate(predictions), covered):
        beta_sum[local] += beta; beta_count[local] += 1
    return (beta_sum / beta_count).astype(np.float32)
