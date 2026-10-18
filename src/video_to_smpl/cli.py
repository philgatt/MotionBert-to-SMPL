from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .config import check_assets, load_config


def parser():
    p = argparse.ArgumentParser(description="Video -> AlphaPose -> MotionBERT -> SMPL")
    p.add_argument("video", type=Path, nargs="?", help="Input video (one person recommended)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pose", choices=["optimization", "learned"], default="learned")
    p.add_argument("--shape", choices=["neutral", "temporal"], default="neutral")
    p.add_argument("--visualize", nargs="+", choices=["2d", "3d", "mesh"], default=[])
    p.add_argument("--output", type=Path, default=Path("outputs/run"), help="New output directory; existing nonempty folders are refused")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--track-id", type=int, help="Select an existing AlphaPose track ID")
    p.add_argument("--alphapose-json", type=Path, help="Reuse raw HALPE-26 AlphaPose detections for this video")
    p.add_argument("--coords", type=Path, help="Start from normalized MotionBERT (T,17,3) .npy coordinates")
    p.add_argument("--fps", type=float, default=30, help="Preview frame rate when using --coords")
    p.add_argument("--check", action="store_true", help="Check configured files for the selected methods and exit")
    return p


def run(args):
    config = load_config(args.config)
    issues = check_assets(config, args.pose, args.shape, from_coords=args.coords is not None,
                          reuse_alphapose=args.alphapose_json is not None)
    if issues:
        raise ValueError("Missing assets:\n  " + "\n  ".join(issues))
    if args.check:
        print("All configured asset paths for the selected methods exist. Model loading is checked when running.")
        return
    if args.coords is None and args.video is None:
        raise ValueError("Provide a video or --coords.")
    if args.coords is not None and (args.video or args.alphapose_json or "2d" in args.visualize):
        raise ValueError("--coords cannot be combined with a video, --alphapose-json or a 2D overlay.")
    import numpy as np
    import torch
    from .keypoints import select_person, validate_coords
    from .body import recover, save_meshes
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be finite and positive.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available. Use --device cpu.")
    # Do not dereference paths: MotionBERT's Python 3.9 can fail resolving
    # sandboxed temporary files on Windows.
    input_path = Path(args.coords or args.video).absolute()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output = Path(args.output).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "pose_method": args.pose, "shape_method": args.shape,
                "input": str(input_path), "device": device, "config": config,
                "coordinates": "normalized MotionBERT H36M-17; meshes centered on regressed pelvis",
                "translation": "SMPL transl centers the mesh; motionbert_root is uncalibrated and is not world translation"}
    def save_manifest():
        (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    save_manifest()
    try:
        if args.coords:
            coords = validate_coords(np.load(input_path, allow_pickle=False))
            info = {"fps": args.fps, "frames": len(coords)}
            missing = np.zeros(len(coords), dtype=bool)
        else:
            from .video import video_info, run_alphapose
            from .motionbert import estimate
            info = video_info(input_path)
            print("[1/4] AlphaPose: detecting 2D keypoints", flush=True)
            if args.alphapose_json:
                detections = json.loads(args.alphapose_json.read_text(encoding="utf-8-sig"))
            else:
                detections = run_alphapose(input_path, output / "alphapose", config, device)
            if not isinstance(detections, list):
                raise ValueError("AlphaPose JSON must contain a list of detections.")
            keypoints, missing = select_person(detections, info["frames"], args.track_id)
            np.save(output / "keypoints_2d.npy", keypoints)
            print("[2/4] MotionBERT: lifting to 3D", flush=True)
            coords = validate_coords(estimate(keypoints, info, config, device))
            if len(coords) != info["frames"]:
                raise ValueError("MotionBERT output does not match the video frame count.")
        np.save(output / "motionbert.npy", coords)
        np.save(output / "missing_detections.npy", missing)
        print(f"[3/4] SMPL: {args.pose} pose / {args.shape} shape", flush=True)
        model, pose, betas, diagnostics, matrices = recover(coords, args.pose, args.shape, config, device)
        translations = save_meshes(model, pose, betas, config, device, output, matrices)
        from scipy.spatial.transform import Rotation
        if matrices is None:
            matrices = Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(-1,24,3,3).astype(np.float32)
        np.savez_compressed(output / "smpl_params.npz", poses=pose, global_orient=pose[:, :3],
                            body_pose=pose[:, 3:], betas=betas, transl=translations,
                            pose_matrices=matrices,
                            motionbert_root=coords[:, 0], frame_indices=np.arange(len(coords)), fps=info["fps"])
        print("[4/4] Saving outputs" + (" and previews" if args.visualize else ""), flush=True)
        if args.visualize:
            from .visualization import render
            render(output, input_path if not args.coords else None, info, args.visualize)
        manifest.update(status="complete", frames=len(coords), fps=info["fps"],
                        missing_frames=int(missing.sum()), diagnostics=diagnostics,
                        elapsed_seconds=time.perf_counter() - started)
        save_manifest()
        print(f"Saved results to {output}")
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
        save_manifest()
        raise


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
