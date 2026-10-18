"""Evaluate a temporal shape checkpoint with the paper's neutral-pose PVE protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from video_to_smpl.methods.shape_inference import _model_from_checkpoint, _predict_track


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motionbert-root", type=Path, required=True)
    p.add_argument("--vibe-test-db", type=Path, required=True)
    p.add_argument("--motionbert-repo", type=Path, required=True)
    p.add_argument("--smpl-data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--no-rx180", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    cargs = state.get("args", {})
    if not isinstance(cargs, dict):
        cargs = vars(cargs)
    clip_len = int(cargs.get("clip_len", 50))
    model = _model_from_checkpoint(state).to(device).eval()

    sys.path.insert(0, str(args.motionbert_repo))
    from lib.utils.utils_smpl import SMPL
    smpl = SMPL(str(args.smpl_data_root), batch_size=1).to(device).eval()
    for parameter in smpl.parameters():
        parameter.requires_grad_(False)

    db = np.load(args.vibe_test_db, allow_pickle=True)
    lookup = {(str(v), int(f)): i for i, (v, f) in enumerate(zip(db["vid_name"], db["frame_id"]))}
    track_predictions, track_targets, track_sizes = [], [], []
    unmatched = 0
    for sequence_dir in sorted(args.motionbert_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        for person_dir in sorted(sequence_dir.glob("person_*")):
            person = int(person_dir.name.split("_")[1]) - 1
            x_path, f_path = person_dir / "X3D_prediction.npy", person_dir / "frame_indices_prediction.npy"
            if not x_path.exists() or not f_path.exists():
                continue
            frames = np.asarray(np.load(f_path), dtype=np.int64)
            matched = [(i, lookup.get((f"{sequence_dir.name}_{person}", int(f)))) for i, f in enumerate(frames)]
            matched = [(i, row) for i, row in matched if row is not None]
            unmatched += len(frames) - len(matched)
            if not matched:
                continue
            input_idx = np.asarray([i for i, _ in matched], dtype=np.int64)
            db_idx = np.asarray([row for _, row in matched], dtype=np.int64)
            coords = np.asarray(np.load(x_path)[input_idx], dtype=np.float32)
            if not args.no_rx180:
                coords *= np.asarray([1, -1, -1], dtype=np.float32)
            per_frame = _predict_track(model, coords, frames[input_idx], clip_len, 1, args.batch_size,
                                       device, str(cargs.get("clip_coordinate_mode", "as_is")))
            # A body has one shape. Match the paper/VIBE aggregation for both estimates.
            track_predictions.append(np.median(per_frame, axis=0).astype(np.float32))
            track_targets.append(np.median(np.asarray(db["shape"][db_idx], dtype=np.float32), axis=0))
            track_sizes.append(len(db_idx))

    prediction = np.asarray(track_predictions, dtype=np.float32)
    target = np.asarray(track_targets, dtype=np.float32)
    errors = []
    with torch.no_grad():
        zeros = torch.zeros((args.batch_size, 72), dtype=torch.float32, device=device)
        for start in range(0, len(prediction), args.batch_size):
            end = min(start + args.batch_size, len(prediction))
            count = end - start
            pb = torch.from_numpy(prediction[start:end]).to(device)
            tb = torch.from_numpy(target[start:end]).to(device)
            z = zeros[:count]
            pred_vertices = smpl(betas=pb, global_orient=z[:, :3], body_pose=z[:, 3:], pose2rot=True, create_transl=False).vertices
            target_vertices = smpl(betas=tb, global_orient=z[:, :3], body_pose=z[:, 3:], pose2rot=True, create_transl=False).vertices
            errors.append(torch.linalg.norm(pred_vertices - target_vertices, dim=2).mean(1).cpu().numpy())
    pve_by_track = np.concatenate(errors) * 1000.0
    weights = np.asarray(track_sizes, dtype=np.float64)
    result = {
        "protocol": "VIBE/3DPW test: theta=0 for predicted and ground-truth SMPL; PVE only; median beta per track",
        "checkpoint": str(args.checkpoint), "tracks": int(len(prediction)), "frames": int(weights.sum()),
        "unmatched_motionbert_frames": int(unmatched), "rx180": not args.no_rx180,
        "pve_mm_track_mean": float(pve_by_track.mean()),
        "pve_mm_frame_weighted": float(np.average(pve_by_track, weights=weights)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
