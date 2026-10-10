from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import cv2


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        info = dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    fps=float(cap.get(cv2.CAP_PROP_FPS)),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if any(v <= 0 for v in info.values()):
            raise ValueError("Video must have valid dimensions, frame count and frame rate.")
        return info
    finally:
        cap.release()


def run_alphapose(video: Path, output: Path, config: dict, device: str) -> list:
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(config["alphapose_repo"])
    command = [config.get("alphapose_python", "python"), "scripts/demo_inference.py",
               "--cfg", config["alphapose_config"], "--checkpoint", config["alphapose_checkpoint"],
               "--video", str(video), "--outdir", str(output), "--sp",
               "--gpus", "-1" if device == "cpu" else "0"]
    env = os.environ.copy()
    # Directly launching a separate Conda Python on Windows needs its DLL paths.
    python_path = Path(config.get("alphapose_python", "python"))
    if os.name == "nt" and python_path.is_absolute():
        prefix = python_path.parent
        env["PATH"] = os.pathsep.join(str(prefix / part) for part in
                                       ("", "Library/bin", "Library/usr/bin", "Library/mingw-w64/bin", "Scripts")) + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    with (output / "alphapose.log").open("w", encoding="utf-8") as log:
        try:
            subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"AlphaPose failed; see {output / 'alphapose.log'}") from exc
    result = output / "alphapose-results.json"
    if not result.is_file():
        raise RuntimeError("AlphaPose did not produce alphapose-results.json; check its log.")
    return json.loads(result.read_text(encoding="utf-8"))
