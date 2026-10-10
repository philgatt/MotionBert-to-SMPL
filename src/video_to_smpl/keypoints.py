"""Select one person while preserving every video frame."""
from __future__ import annotations

import re
import numpy as np


def select_person(detections: list, frame_count: int, track_id: int | None = None):
    if frame_count < 1:
        raise ValueError("The video has no frames.")
    by_frame = {}
    for item in detections:
        digits = re.findall(r"\d+", str(item.get("image_id", "")).rsplit(".", 1)[0])
        if not digits:
            raise ValueError("AlphaPose image_id must contain a zero-based video frame number.")
        index = int(digits[-1])
        if not 0 <= index < frame_count:
            raise ValueError(f"AlphaPose frame {index} is outside the video timeline.")
        k = np.asarray(item.get("keypoints", []), dtype=np.float32)
        if k.size != 78 or not np.isfinite(k).all():
            raise ValueError("Expected finite HALPE-26 keypoints from AlphaPose.")
        idx = item.get("idx", -1)
        idx = idx[0] if isinstance(idx, list) and idx else idx
        if track_id is not None and int(idx) != track_id:
            continue
        by_frame.setdefault(index, []).append(k.reshape(26, 3))
    result = np.zeros((frame_count, 26, 3), dtype=np.float32)
    missing = np.ones(frame_count, dtype=bool)
    center = None
    for index in range(frame_count):
        candidates = [k for k in by_frame.get(index, []) if np.any(k[:, 2] > 0)]
        if not candidates:
            continue
        def rank(k):
            c = k[k[:, 2] > 0, :2].mean(axis=0)
            return -float(k[:, 2].mean()) if center is None else float(np.linalg.norm(c - center))
        chosen = min(candidates, key=rank)
        result[index] = chosen
        center = chosen[chosen[:, 2] > 0, :2].mean(axis=0)
        missing[index] = False
    if missing.all():
        raise ValueError("No detections for the selected person.")
    # Fill gaps in positions; keep zero confidence on missing frames/joints.
    for joint in range(26):
        valid = np.flatnonzero(result[:, joint, 2] > 0)
        if len(valid):
            for axis in range(2):
                result[:, joint, axis] = np.interp(np.arange(frame_count), valid, result[valid, joint, axis])
    return result, missing


def halpe_to_h36m(coords: np.ndarray) -> np.ndarray:
    indices = [19, 12, 14, 16, 11, 13, 15, 19, 18, 0, 17, 5, 7, 9, 6, 8, 10]
    result = coords[:, indices].copy()
    result[:, 7] = (coords[:, 18] + coords[:, 19]) * 0.5
    return result


def validate_coords(coords: np.ndarray) -> np.ndarray:
    x = np.asarray(coords, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (17, 3) or len(x) == 0:
        raise ValueError(f"Expected nonempty MotionBERT coordinates (T,17,3), got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("MotionBERT coordinates contain NaN or infinity.")
    return x
