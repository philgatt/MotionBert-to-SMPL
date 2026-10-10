"""Portable configuration: file paths are relative to the config file."""
from __future__ import annotations

import json
from pathlib import Path

PATH_KEYS = {
    "alphapose_repo", "alphapose_config", "alphapose_checkpoint",
    "motionbert_repo", "motionbert_config", "motionbert_checkpoint",
    "smpl_model", "h36m_regressor", "pose_checkpoint", "shape_checkpoint",
}


def load_config(path: Path) -> dict:
    # Python 3.9 in the MotionBERT environment can raise WinError 5 from
    # Path.resolve() for sandboxed temporary paths. Absolute paths preserve the
    # same configuration-relative semantics without dereferencing the path.
    path = Path(path).absolute()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    for key in PATH_KEYS & config.keys():
        if config[key]:
            value = Path(config[key]).expanduser()
            config[key] = str(value if value.is_absolute() else path.parent / value)
    for key, default in (("clip_len", 243), ("batch_size", 64),
                         ("shape_stride", 25), ("optimization_iterations", 200)):
        config[key] = int(config.get(key, default))
        if config[key] < 1:
            raise ValueError(f"{key} must be positive.")
    if float(config.get("optimization_lr", 0.01)) <= 0:
        raise ValueError("optimization_lr must be positive.")
    for key in ("pose_rx180", "shape_rx180", "shape_root_center"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a JSON boolean.")
    return config


def check_assets(config: dict, pose: str, shape: str, *, from_coords=False,
                 reuse_alphapose=False) -> list[str]:
    required = {"smpl_model", "h36m_regressor"}
    if not from_coords:
        required |= {"motionbert_repo", "motionbert_config", "motionbert_checkpoint"}
        if not reuse_alphapose:
            required |= {"alphapose_repo", "alphapose_config", "alphapose_checkpoint"}
    if pose == "learned":
        required.add("pose_checkpoint")
    if shape == "temporal":
        required.add("shape_checkpoint")
    errors = []
    for key in sorted(required):
        value = config.get(key)
        p = Path(value) if value else None
        valid = p is not None and (p.is_dir() if key.endswith("_repo") else p.is_file())
        if not valid:
            errors.append(f"{key}: missing {'directory' if key.endswith('_repo') else 'file'} ({value or 'not configured'})")
    for key, entry in (("alphapose_repo", "scripts/demo_inference.py"),
                       ("motionbert_repo", "lib/model/DSTformer.py")):
        if key in required and config.get(key) and not (Path(config[key]) / entry).is_file():
            errors.append(f"{key}: expected {entry}")
    return errors
