"""Train the Figure 3.4 shape-only residual temporal convolution network."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from video_to_smpl.methods.res_tcn_shape_model import StagedResTCNShapeRegressor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vibe-root", type=Path, required=True)
    p.add_argument("--motionbert-predictions-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=PROJECT / "training-runs")
    p.add_argument("--run-name", default="shape_only_res_tcn_motionbert")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--clip-len", type=int, default=50)
    p.add_argument("--clip-stride", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-rx180", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Train on CPU (for CUDA troubleshooting).")
    p.add_argument("--disable-cudnn", action="store_true", help="Use PyTorch CUDA kernels instead of cuDNN.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def split_data(vibe_file: Path, prediction_root: Path):
    data = np.load(vibe_file, allow_pickle=True)
    index = {(str(v), int(f)): i for i, (v, f) in enumerate(zip(data["vid_name"], data["frame_id"]))}
    coords = np.empty((len(index), 17, 3), dtype=np.float32)
    frame = np.empty(len(index), dtype=np.int64)
    track = np.empty(len(index), dtype=object)
    filled = np.zeros(len(index), dtype=bool)
    for sequence_dir in sorted(prediction_root.iterdir()):
        if not sequence_dir.is_dir(): continue
        for person_dir in sorted(sequence_dir.glob("person_*")):
            x_file, f_file = person_dir / "X3D_prediction.npy", person_dir / "frame_indices_prediction.npy"
            if not x_file.exists() or not f_file.exists(): continue
            person = int(person_dir.name.split("_")[1]) - 1
            for x, f in zip(np.load(x_file), np.load(f_file)):
                row = index.get((f"{sequence_dir.name}_{person}", int(f)))
                if row is not None:
                    coords[row], frame[row], track[row], filled[row] = x, int(f), f"{sequence_dir.name}_{person}", True
    if not filled.all(): raise RuntimeError(f"Missing MotionBERT coordinates for {int((~filled).sum())} rows")
    return coords, np.asarray(data["shape"], dtype=np.float32), frame, track


class Clips(Dataset):
    def __init__(self, coords, betas, frame, track, clip_len, stride, rx180):
        self.coords, self.betas, self.windows = coords, betas, []
        for key in np.unique(track):
            idx = np.flatnonzero(track == key); idx = idx[np.argsort(frame[idx])]
            for start in range(0, len(idx) - clip_len + 1, stride):
                window = idx[start:start + clip_len]
                if np.all(np.diff(frame[window]) == 1): self.windows.append(window)
        if not self.windows: raise RuntimeError("No contiguous clips")
        self.rx180 = rx180

    def __len__(self): return len(self.windows)
    def __getitem__(self, index):
        window = self.windows[index]
        x = self.coords[window].copy()
        if self.rx180: x *= np.array([1, -1, -1], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(self.betas[window].mean(axis=0).astype(np.float32))


def evaluate(model, loader, device):
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for x, beta in loader:
            value = torch.nn.functional.l1_loss(model(x.to(device)), beta.to(device), reduction="sum")
            total += float(value.item()); count += len(x) * 10
    return total / count


def main():
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} cudnn={torch.backends.cudnn.enabled}", flush=True)
    train = split_data(args.vibe_root / "3dpw_train_db.npz", args.motionbert_predictions_root / "train")
    val = split_data(args.vibe_root / "3dpw_val_db.npz", args.motionbert_predictions_root / "val")
    train_ds = Clips(*train, args.clip_len, args.clip_stride, not args.no_rx180)
    val_ds = Clips(*val, args.clip_len, args.clip_stride, not args.no_rx180)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True)
    # Eight residual units: 2×128, 128→192, 2×192, 192→256, 2×256.
    # This is the staged network from Figure 3.4, with its specified k=5 input block.
    model = StagedResTCNShapeRegressor(kernel_size=5, dropout=args.dropout, head_dropout=0.0, mlp_hidden_dim=128, pooling="avg").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run = args.output_root / args.run_name; checkpoint_dir = run / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for x, beta in train_loader:
            optimizer.zero_grad(set_to_none=True); prediction = model(x.to(device)); loss = torch.nn.functional.l1_loss(prediction, beta.to(device))
            loss.backward(); optimizer.step(); total += float(loss.item()) * len(x); count += len(x)
        train_l1, val_l1 = total / count, evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1})
        print(f"epoch={epoch:03d} train_l1={train_l1:.6f} val_l1={val_l1:.6f}", flush=True)
        if val_l1 < best:
            best = val_l1
            torch.save({"model_state_dict": model.state_dict(), "architecture": "staged", "args": {**vars(args), "architecture": "staged", "kernel_size": 5, "head_dropout": 0.0, "pooling": "avg", "mlp_hidden_dim": 128, "clip_coordinate_mode": "as_is"}, "protocol": "Figure 3.4: raw non-root-centred MotionBERT coordinates, Rx180, L1 beta loss", "train_samples": len(train[0]), "val_samples": len(val[0])}, checkpoint_dir / "best_model.pt")
        (run / "metrics.json").write_text(json.dumps({"best_val_l1": best, "train_samples": len(train[0]), "val_samples": len(val[0]), "train_clips": len(train_ds), "val_clips": len(val_ds), "history": history}, indent=2), encoding="utf-8")
    print(checkpoint_dir / "best_model.pt")


if __name__ == "__main__": main()
