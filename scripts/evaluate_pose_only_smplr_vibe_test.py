"""Evaluate the bundled pose-only SMPLR checkpoint on the VIBE/3DPW test split.

The VIBE database does not include MotionBERT predictions for every test frame.
Consequently this is a controlled *neutral-SMPL 17-joint input* evaluation: input
joints are generated from each ground-truth pose with beta=0, exactly as in the
training protocol. It is not an end-to-end video/MotionBERT benchmark.
"""
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


def similarity_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    p.add_argument("--vibe-test-db", type=Path, required=True)
    p.add_argument("--vibe-h36m-regressor", type=Path, required=True)
    p.add_argument("--motionbert-repo", type=Path, required=True)
    p.add_argument("--smpl-data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "pose_only_smplr_vibe_relu_bn_20261008.pt")
    p.add_argument("--output", type=Path, default=PROJECT / "docs" / "pose_only_vibe_test_metrics.json")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.motionbert_repo))
    from lib.utils.utils_smpl import SMPL
    db = np.load(args.vibe_test_db, allow_pickle=True)
    theta = np.asarray(db["pose"], dtype=np.float32)
    beta = np.asarray(db["shape"], dtype=np.float32)
    target14 = np.asarray(db["joints3D"], dtype=np.float32)
    smpl = SMPL(str(args.smpl_data_root), batch_size=1).to(device).eval()
    for parameter in smpl.parameters(): parameter.requires_grad_(False)
    input_reg = smpl.J_regressor_h36m.to(device=device, dtype=torch.float32)
    eval_reg = torch.from_numpy(np.load(args.vibe_h36m_regressor)).to(device=device, dtype=torch.float32)
    model = PoseOnlySMPLRRegressor().to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    errors, errors_pa, pve = [], [], []
    with torch.no_grad():
        for start in range(0, len(theta), args.batch_size):
            end = min(len(theta), start + args.batch_size)
            gt_pose = torch.from_numpy(theta[start:end]).to(device)
            gt_beta = torch.from_numpy(beta[start:end]).to(device)
            zero_beta = torch.zeros_like(gt_beta)
            neutral = smpl(betas=zero_beta, global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], pose2rot=True, create_transl=False)
            x17 = torch.einsum("jv,nvc->njc", input_reg, neutral.vertices).sub_(torch.einsum("jv,nvc->njc", input_reg, neutral.vertices)[:, :1])
            pred_rot = model(x17.reshape(len(x17), 51))
            pred = smpl(betas=zero_beta, global_orient=pred_rot[:, :1], body_pose=pred_rot[:, 1:], pose2rot=False, create_transl=False)
            gt = smpl(betas=gt_beta, global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], pose2rot=True, create_transl=False)
            pred14 = torch.einsum("jv,nvc->njc", eval_reg, pred.vertices)[:, H36M_TO_J14].cpu().numpy()
            gt14 = target14[start:end].copy()
            pred14 -= (pred14[:, 2:3] + pred14[:, 3:4]) / 2
            gt14 -= (gt14[:, 2:3] + gt14[:, 3:4]) / 2
            errors.extend(np.linalg.norm(pred14 - gt14, axis=2).mean(1))
            errors_pa.extend(np.linalg.norm(np.stack([similarity_align(a, b) for a, b in zip(pred14, gt14)]) - gt14, axis=2).mean(1))
            pred_v = pred.vertices - pred.vertices[:, :1]
            gt_v = gt.vertices - gt.vertices[:, :1]
            pve.extend(torch.linalg.norm(pred_v - gt_v, dim=2).mean(1).cpu().numpy())
    result = {"protocol": "VIBE/3DPW test, controlled neutral-SMPL H36M-17 input; not end-to-end MotionBERT", "frames": len(theta), "mpjpe_mm": float(np.mean(errors) * 1000), "pa_mpjpe_mm": float(np.mean(errors_pa) * 1000), "root_aligned_pve_mm": float(np.mean(pve) * 1000), "checkpoint": str(args.checkpoint)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
