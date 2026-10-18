"""VIBE-protocol evaluation of the checkpoint using stored 3DPW MotionBERT coordinates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from video_to_smpl.methods.pose_only_smplr import PoseOnlySMPLRRegressor

H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]


def pa_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc, tc = pred - pred.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ tc)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    scale = np.sum((pc @ rotation) * tc) / max(np.sum(pc ** 2), 1e-12)
    return scale * (pc @ rotation) + target.mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motionbert-root", type=Path, required=True, help=".../predictions/motionbert/test")
    p.add_argument("--vibe-test-db", type=Path, required=True)
    p.add_argument("--vibe-h36m-regressor", type=Path, required=True)
    p.add_argument("--motionbert-repo", type=Path, required=True)
    p.add_argument("--smpl-data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "pose_only_smplr_vibe_relu_bn_20261008.pt")
    p.add_argument("--output", type=Path, default=PROJECT / "docs" / "pose_only_motionbert_vibe_metrics.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--no-rx180", action="store_true", help="Disable the established x-axis 180-degree convention.")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.motionbert_repo))
    from lib.utils.utils_smpl import SMPL
    db = np.load(args.vibe_test_db, allow_pickle=True)
    lookup = {(str(v), int(f)): i for i, (v, f) in enumerate(zip(db["vid_name"], db["frame_id"]))}
    smpl = SMPL(str(args.smpl_data_root), batch_size=1).to(device).eval()
    for parameter in smpl.parameters(): parameter.requires_grad_(False)
    eval_reg = torch.from_numpy(np.load(args.vibe_h36m_regressor)).to(device=device, dtype=torch.float32)
    state = torch.load(args.checkpoint, map_location=device)
    activation = "elu" if "elu" in str(state.get("model_type", "")) else "relu"
    model = PoseOnlySMPLRRegressor(activation=activation).to(device).eval()
    model.load_state_dict(state["model_state_dict"], strict=True)
    predictions, targets, pve = [], [], []
    skipped = 0
    with torch.no_grad():
        for sequence_dir in sorted(args.motionbert_root.iterdir()):
            if not sequence_dir.is_dir():
                continue
            for person_dir in sorted(sequence_dir.glob("person_*")):
                person = int(person_dir.name.split("_")[1]) - 1
                coord_path, frame_path = person_dir / "X3D_prediction.npy", person_dir / "frame_indices_prediction.npy"
                if not coord_path.exists() or not frame_path.exists():
                    continue
                frames = np.load(frame_path).astype(np.int64)
                matched = [(i, lookup.get((f"{sequence_dir.name}_{person}", int(frame)))) for i, frame in enumerate(frames)]
                matched = [(i, j) for i, j in matched if j is not None]
                skipped += len(frames) - len(matched)
                if not matched: continue
                row_idx = np.asarray([x[0] for x in matched]); db_idx = np.asarray([x[1] for x in matched])
                coords = np.asarray(np.load(coord_path)[row_idx], dtype=np.float32)
                if not args.no_rx180: coords *= np.array([1, -1, -1], dtype=np.float32)
                theta = np.asarray(db["pose"][db_idx], dtype=np.float32); beta = np.asarray(db["shape"][db_idx], dtype=np.float32)
                target = np.asarray(db["joints3D"][db_idx], dtype=np.float32)
                for start in range(0, len(coords), args.batch_size):
                    end = min(len(coords), start + args.batch_size)
                    x, gt_theta, gt_beta = (torch.from_numpy(a[start:end]).to(device) for a in (coords, theta, beta))
                    pred_rot = model(x.reshape(len(x), 51)); zeros = torch.zeros_like(gt_beta)
                    pred = smpl(betas=zeros, global_orient=pred_rot[:, :1], body_pose=pred_rot[:, 1:], pose2rot=False, create_transl=False)
                    gt = smpl(betas=gt_beta, global_orient=gt_theta[:, :3], body_pose=gt_theta[:, 3:], pose2rot=True, create_transl=False)
                    predictions.append(torch.einsum("jv,nvc->njc", eval_reg, pred.vertices)[:, H36M_TO_J14].cpu().numpy())
                    targets.append(target[start:end])
                    # VIBE PVE is translation-free mesh PVE, without pelvis alignment.
                    pve.append(torch.linalg.norm(pred.vertices - gt.vertices, dim=2).mean(1).cpu().numpy())
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    pelvis_pred, pelvis_target = (pred[:, 2:3] + pred[:, 3:4]) / 2, (target[:, 2:3] + target[:, 3:4]) / 2
    pred, target = pred - pelvis_pred, target - pelvis_target
    mpjpe = np.linalg.norm(pred - target, axis=2).mean() * 1000
    pa = np.linalg.norm(np.stack([pa_align(a, b) for a, b in zip(pred, target)]) - target, axis=2).mean() * 1000
    result = {"protocol": "VIBE", "input": "stored 3DPW MotionBERT X3D predictions", "activation": activation, "frames": len(pred), "unmatched_motionbert_frames": skipped, "rx180": not args.no_rx180, "mpjpe_mm": float(mpjpe), "pa_mpjpe_mm": float(pa), "pve_mm": float(np.concatenate(pve).mean() * 1000), "checkpoint": str(args.checkpoint)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
