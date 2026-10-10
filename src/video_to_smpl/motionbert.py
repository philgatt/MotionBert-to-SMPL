"""Inference using the external MotionBERT backbone and original pixel normalization."""
from __future__ import annotations

import sys
from types import SimpleNamespace
import numpy as np
import torch
import yaml

from .keypoints import halpe_to_h36m
from .checkpoints import load_checkpoint

SWAP = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def flip(x):
    x = x.clone()
    x[..., 0] *= -1
    return x[..., SWAP, :]


def estimate(keypoints, info, config, device):
    sys.dont_write_bytecode = True
    sys.path.insert(0, config["motionbert_repo"])
    from lib.utils.learning import load_backbone

    with open(config["motionbert_config"], encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    model = load_backbone(SimpleNamespace(**settings)).to(device).eval()
    checkpoint = load_checkpoint(config["motionbert_checkpoint"])
    state = checkpoint["model_pos"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    coords = halpe_to_h36m(keypoints)
    coords[..., :2] -= np.array([info["width"], info["height"]]) / 2
    coords[..., :2] /= min(info["width"], info["height"]) / 2
    outputs = []
    clip_len = min(config["clip_len"], int(settings["maxlen"]))
    with torch.inference_mode():
        for start in range(0, len(coords), clip_len):
            x = torch.from_numpy(coords[start:start + clip_len]).unsqueeze(0).to(device)
            if settings.get("no_conf", False):
                x = x[..., :2]
            prediction = model(x)
            if settings.get("flip", False):
                prediction = (prediction + flip(model(flip(x)))) / 2
            if settings.get("rootrel", False):
                prediction[:, :, 0] = 0
            else:
                prediction[:, 0, 0, 2] = 0
            if settings.get("gt_2d", False):
                prediction[..., :2] = x[..., :2]
            outputs.append(prediction[0].cpu().numpy())
    # Keep normalized coordinates for downstream thesis regressors, not pixel-scaled values.
    return np.concatenate(outputs).astype(np.float32)
