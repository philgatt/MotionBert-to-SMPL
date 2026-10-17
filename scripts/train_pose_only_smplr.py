"""Retrain the exact Figure 3.3 pose-only SMPLR architecture on VIBE splits."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Avoid Path.resolve(): the inherited MotionBERT Python 3.9 environment can
# receive an access-denied response for files on this workspace drive.
PROJECT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from video_to_smpl.methods.pose_only_smplr import PoseOnlySMPLRRegressor, axis_angle_to_matrix

LANDMARKS = [3042, 4428, 3503, 4311, 822, 4361, 877, 4386, 898, 3222, 6620, 2544, 2820, 1258, 4116, 5365, 5989, 331]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vibe-root", type=Path, required=True)
    p.add_argument("--motionbert-repo", type=Path, required=True)
    p.add_argument("--smpl-data-root", type=Path, required=True)
    p.add_argument("--motionbert-predictions-root", type=Path, default=None,
                   help="Stored MotionBERT predictions root containing train/ and val/. Required for real-input training.")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="3DPW root used to convert global VIBE root pose to camera space.")
    p.add_argument("--target-pose-space", choices=["global", "incam"], default="incam")
    p.add_argument("--output-root", type=Path, default=PROJECT / "training-runs")
    p.add_argument("--run-name", default="pose_only_smplr_vibe_relu_bn")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--activation", choices=["relu", "elu"], default="relu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def smpl_inputs(vibe_file, smpl, regressor, device):
    data = np.load(vibe_file, allow_pickle=True)
    pose = np.asarray(data["pose"], dtype=np.float32)
    coords = np.empty((len(pose), 17, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(pose), 256):
            theta = torch.from_numpy(pose[start:start + 256]).to(device)
            zeros = torch.zeros((len(theta), 10), dtype=torch.float32, device=device)
            out = smpl(betas=zeros, global_orient=theta[:, :3], body_pose=theta[:, 3:], pose2rot=True, create_transl=False)
            joints = torch.einsum("jv,nvc->njc", regressor, out.vertices)
            coords[start:start + len(theta)] = (joints - joints[:, :1]).cpu().numpy()
    return coords.reshape(len(coords), 51), pose


def motionbert_inputs(vibe_file, prediction_split_root):
    """Load exact VIBE-split rows from stored MotionBERT X3D predictions."""
    data = np.load(vibe_file, allow_pickle=True)
    pose = np.asarray(data["pose"], dtype=np.float32)
    index = {(str(video), int(frame)): i for i, (video, frame) in enumerate(zip(data["vid_name"], data["frame_id"]))}
    coords = np.empty((len(pose), 17, 3), dtype=np.float32)
    found = np.zeros(len(pose), dtype=bool)
    for sequence_dir in sorted(prediction_split_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        for person_dir in sorted(sequence_dir.glob("person_*")):
            coordinate_file = person_dir / "X3D_prediction.npy"
            frame_file = person_dir / "frame_indices_prediction.npy"
            if not coordinate_file.exists() or not frame_file.exists():
                continue
            person = int(person_dir.name.split("_")[1]) - 1
            for coordinate, frame in zip(np.load(coordinate_file), np.load(frame_file)):
                row = index.get((f"{sequence_dir.name}_{person}", int(frame)))
                if row is not None:
                    coords[row] = coordinate
                    found[row] = True
    if not found.all():
        raise RuntimeError(f"Missing MotionBERT coordinates for {int((~found).sum())} VIBE rows in {prediction_split_root}")
    return coords.reshape(len(coords), 51), pose


def incam_targets(vibe_file, dataset_root, split):
    """Replace only global orientation with 3DPW camera-space orientation."""
    from scipy.spatial.transform import Rotation
    data = np.load(vibe_file, allow_pickle=True)
    theta = np.asarray(data["pose"], dtype=np.float32).copy()
    cache = {}
    folder = "validation" if split == "val" else split
    for row, (video, frame) in enumerate(zip(data["vid_name"], data["frame_id"])):
        sequence, person_s = str(video).rsplit("_", 1)
        if sequence not in cache:
            path = dataset_root / "sequenceFiles" / "sequenceFiles" / folder / f"{sequence}.pkl"
            with path.open("rb") as handle:
                cache[sequence] = pickle.load(handle, encoding="latin1")["cam_poses"]
        cam = np.asarray(cache[sequence][int(frame)], dtype=np.float64)[:3, :3]
        world = Rotation.from_rotvec(theta[row, :3]).as_matrix()
        theta[row, :3] = Rotation.from_matrix(cam @ world).as_rotvec().astype(np.float32)
    return theta


def loss_terms(pred, theta, smpl, smpl_regressor, landmark_idx):
    gt_rot = axis_angle_to_matrix(theta.reshape(-1, 24, 3))
    loss_r = torch.mean(torch.abs(pred - gt_rot))
    zeros = torch.zeros((len(theta), 10), dtype=torch.float32, device=theta.device)
    pred_smpl = smpl(betas=zeros, global_orient=pred[:, :1], body_pose=pred[:, 1:], pose2rot=False, create_transl=False)
    with torch.no_grad():
        gt_smpl = smpl(betas=zeros, global_orient=theta[:, :3], body_pose=theta[:, 3:], pose2rot=True, create_transl=False)
    pred_joints = torch.einsum("jv,nvc->njc", smpl_regressor, pred_smpl.vertices)
    gt_joints = torch.einsum("jv,nvc->njc", smpl_regressor, gt_smpl.vertices)
    loss_joints = torch.mean(torch.abs(pred_joints[:, 1:] - gt_joints[:, 1:]))
    loss_landmarks = torch.mean(torch.abs(pred_smpl.vertices[:, landmark_idx] - gt_smpl.vertices[:, landmark_idx]))
    return loss_r + loss_joints + loss_landmarks, (loss_r, loss_joints, loss_landmarks)


def evaluate(model, loader, smpl, smpl_regressor, landmark_idx, device):
    model.eval(); sums = np.zeros(4, dtype=np.float64); count = 0
    with torch.no_grad():
        for x, theta in loader:
            x, theta = x.to(device), theta.to(device)
            loss, parts = loss_terms(model(x), theta, smpl, smpl_regressor, landmark_idx)
            n = len(x); sums += np.array([loss.item(), *[v.item() for v in parts]]) * n; count += n
    return (sums / count).tolist()


def main():
    args = arguments()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(args.motionbert_repo))
    from lib.utils.utils_smpl import SMPL
    run = args.output_root / args.run_name
    if run.exists() and any(run.iterdir()) and not args.overwrite: raise FileExistsError(run)
    run.mkdir(parents=True, exist_ok=True); (run / "checkpoints").mkdir(exist_ok=True)
    smpl = SMPL(str(args.smpl_data_root), batch_size=1).to(device).eval()
    for parameter in smpl.parameters(): parameter.requires_grad_(False)
    h36m_regressor = smpl.J_regressor_h36m.to(device=device, dtype=torch.float32)
    smpl_regressor = smpl.J_regressor.to(device=device, dtype=torch.float32)
    if args.motionbert_predictions_root is None:
        train_x, train_y = smpl_inputs(args.vibe_root / "3dpw_train_db.npz", smpl, h36m_regressor, device)
        val_x, val_y = smpl_inputs(args.vibe_root / "3dpw_val_db.npz", smpl, h36m_regressor, device)
        input_protocol = "neutral-SMPL H36M-17 joints"
    else:
        train_x, train_y = motionbert_inputs(args.vibe_root / "3dpw_train_db.npz", args.motionbert_predictions_root / "train")
        val_x, val_y = motionbert_inputs(args.vibe_root / "3dpw_val_db.npz", args.motionbert_predictions_root / "val")
        if args.target_pose_space == "incam":
            if args.dataset_root is None:
                raise ValueError("--dataset-root is required with MotionBERT inputs and --target-pose-space incam")
            train_y = incam_targets(args.vibe_root / "3dpw_train_db.npz", args.dataset_root, "train")
            val_y = incam_targets(args.vibe_root / "3dpw_val_db.npz", args.dataset_root, "val")
        input_protocol = "stored MotionBERT H36M-17 X3D predictions"
    if (len(train_x), len(val_x)) != (22663, 10261): raise RuntimeError(f"Unexpected VIBE split sizes: {len(train_x)}, {len(val_x)}")
    np.savez_compressed(run / "dataset_manifest.npz", train_samples=len(train_x), val_samples=len(val_x))
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=args.batch_size, pin_memory=True)
    model = PoseOnlySMPLRRegressor(activation=args.activation).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    landmark_idx = torch.tensor(LANDMARKS, device=device); history = []; best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = np.zeros(4, dtype=np.float64); count = 0
        for x, theta in train_loader:
            x, theta = x.to(device), theta.to(device); optimizer.zero_grad(set_to_none=True)
            loss, parts = loss_terms(model(x), theta, smpl, smpl_regressor, landmark_idx); loss.backward(); optimizer.step()
            n = len(x); total += np.array([loss.item(), *[v.item() for v in parts]]) * n; count += n
        train = (total / count).tolist(); val = evaluate(model, val_loader, smpl, smpl_regressor, landmark_idx, device)
        row = {"epoch": epoch, "train_loss": train[0], "train_rotation": train[1], "train_joints23": train[2], "train_landmarks18": train[3], "val_loss": val[0], "val_rotation": val[1], "val_joints23": val[2], "val_landmarks18": val[3]}; history.append(row)
        print(f"epoch={epoch:02d} train={train[0]:.6f} val={val[0]:.6f} rotation={val[1]:.6f}", flush=True)
        if val[0] < best:
            best = val[0]
            torch.save({"model_state_dict": model.state_dict(), "model_type": f"pose_only_smplr_{args.activation}_bn_v1", "architecture": f"Figure 3.3: {args.activation.upper()}+BatchNorm, 48 input, dual 24x9 heads", "args": vars(args), "neutral_betas": True, "train_samples": len(train_x), "val_samples": len(val_x)}, run / "checkpoints" / "best_model.pt")
        (run / "metrics.json").write_text(json.dumps({"protocol": {"input": input_protocol + "; the model derives 16 root-centred unit bone vectors", "target_pose_space": args.target_pose_space, "predicted_mesh_landmarks_input": False, "prediction_beta": "zero", "ground_truth_beta": "zero", "optimizer": "Adam", "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr, "train_samples": len(train_x), "val_samples": len(val_x)}, "best_val_loss": best, "history": history}, indent=2), encoding="utf-8")
    print(run / "checkpoints" / "best_model.pt")


if __name__ == "__main__": main()
