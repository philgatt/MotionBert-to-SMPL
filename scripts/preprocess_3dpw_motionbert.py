import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="3DPW MotionBERT preprocessing stage")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--motionbert-repo", type=Path, required=True)
    parser.add_argument(
        "--motionbert-config",
        type=str,
        default="configs/pose3d/MB_ft_h36m_global_lite.yaml",
    )
    parser.add_argument(
        "--motionbert-checkpoint",
        type=str,
        default="checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
    )
    parser.add_argument("--clip-len", type=int, default=243)
    parser.add_argument("--focus", type=int, default=None)
    parser.add_argument("--pixel", action="store_true")
    parser.add_argument("--render-video", action="store_true", help="Render 3D preview video (X3D_prediction.mp4)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", type=str, default="", choices=["", "train", "test", "val"])
    parser.add_argument("--only-sequence", type=str, default="")
    parser.add_argument("--only-person", type=int, default=0, help="1 or 2; 0 means all")
    parser.add_argument("--dataset-root", type=Path, default=None)  # accepted for compatibility
    parser.add_argument(
        "--rootrel",
        action="store_true",
        help="Force root-relative MotionBERT output (sets model config flag rootrel=True).",
    )
    return parser.parse_args()


def find_person_dirs(alphapose_root, split_filter, only_sequence, only_person):
    person_dirs = []
    if not alphapose_root.exists():
        return person_dirs
    for split_dir in sorted(alphapose_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        if split_filter and split_name != split_filter:
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            if only_sequence and seq_dir.name != only_sequence:
                continue
            for person_dir in sorted(seq_dir.glob("person_*")):
                if only_person:
                    if person_dir.name != f"person_{only_person}":
                        continue
                json_path = person_dir / "keypoints_2d_prediction.json"
                source_video = person_dir / "source_video.txt"
                source_frames_dir = person_dir / "source_frames_dir.txt"
                if json_path.exists() and (source_video.exists() or source_frames_dir.exists()):
                    person_dirs.append((split_name, seq_dir.name, person_dir.name, person_dir))
    return person_dirs


def resolve_source(person_dir: Path):
    source_video = person_dir / "source_video.txt"
    source_frames_dir = person_dir / "source_frames_dir.txt"
    if source_video.exists():
        path = Path(source_video.read_text(encoding="utf-8").strip())
        if path.exists():
            return "video", path
    if source_frames_dir.exists():
        path = Path(source_frames_dir.read_text(encoding="utf-8").strip())
        if path.exists():
            return "frames", path
    raise FileNotFoundError(
        f"Missing usable source in {person_dir} (expected source_video.txt or source_frames_dir.txt)"
    )


def _frame_size_from_dir(frames_dir: Path):
    first = frames_dir / "image_00000.jpg"
    if not first.exists():
        candidates = sorted(frames_dir.glob("image_*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"No image_*.jpg found in {frames_dir}")
        first = candidates[0]
    img = imageio.v2.imread(str(first))
    h, w = int(img.shape[0]), int(img.shape[1])
    return int(w), int(h)


def infer_one_person(
    json_path,
    source_kind,
    source_path,
    out_dir,
    opts,
    args,
    model_pos,
    WildDetDataset,
    flip_data,
    render_and_save=None,
):
    testloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": int(opts.num_workers),
        "pin_memory": True,
        "drop_last": False,
    }

    if source_kind == "video":
        vid = imageio.get_reader(str(source_path), "ffmpeg")
        meta = vid.get_meta_data()
        fps_in = float(meta.get("fps", 30.0))
        vid_size = tuple(meta.get("size", (0, 0)))
        if not vid_size or len(vid_size) != 2 or min(vid_size) <= 0:
            raise RuntimeError(f"Invalid video metadata for {source_path}: size={vid_size}")
    else:
        vid_size = _frame_size_from_dir(Path(source_path))
        fps_in = 30.0
    os.makedirs(out_dir, exist_ok=True)

    if opts.pixel:
        wild_dataset = WildDetDataset(
            str(json_path),
            clip_len=opts.clip_len,
            vid_size=vid_size,
            scale_range=None,
            focus=opts.focus,
        )
    else:
        wild_dataset = WildDetDataset(
            str(json_path),
            clip_len=opts.clip_len,
            scale_range=[1, 1],
            focus=opts.focus,
        )

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader, leave=False, desc="MotionBERT clips", unit="clip"):
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0
            else:
                predicted_3d_pos[:, 0, 0, 2] = 0
            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)

    if opts.render_video and render_and_save is not None:
        out_mp4 = Path(out_dir) / "X3D_prediction.mp4"
        render_and_save(results_all, str(out_mp4), keep_imgs=False, fps=fps_in)

    if opts.pixel:
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0

    out_npy = Path(out_dir) / "X3D_prediction.npy"
    np.save(str(out_npy), results_all)
    return out_npy


def load_frame_indices_from_alphapose(json_path: Path, person_dir: Path) -> np.ndarray:
    idx_npy = person_dir / "frame_indices_prediction.npy"
    if idx_npy.exists():
        return np.load(idx_npy).astype(np.int32)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frame_idxs = []
    for i, item in enumerate(data):
        frame_idx = item.get("frame_idx", i)
        frame_idxs.append(int(frame_idx))
    return np.asarray(frame_idxs, dtype=np.int32)


def main():
    opts = parse_args()
    motionbert_repo = opts.motionbert_repo
    if str(motionbert_repo) not in sys.path:
        sys.path.insert(0, str(motionbert_repo))

    from lib.data.dataset_wild import WildDetDataset
    from lib.utils.learning import load_backbone
    from lib.utils.tools import get_config
    from lib.utils.utils_data import flip_data
    from lib.utils.vismo import render_and_save

    args = get_config(str(motionbert_repo / opts.motionbert_config))
    if opts.rootrel:
        args.rootrel = True
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    checkpoint_path = motionbert_repo / opts.motionbert_checkpoint
    print("Loading checkpoint", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)
    model_pos = model_backbone
    model_pos.eval()

    alphapose_root = opts.output_root / "predictions" / "alphapose"
    motionbert_root = opts.output_root / "predictions" / "motionbert"
    person_dirs = find_person_dirs(
        alphapose_root=alphapose_root,
        split_filter=opts.split,
        only_sequence=opts.only_sequence,
        only_person=opts.only_person,
    )
    print("Found person folders:", len(person_dirs))
    print("MotionBERT rootrel:", bool(getattr(args, "rootrel", False)))

    processed = 0
    skipped = 0
    failed = 0
    for split_name, seq_name, person_name, person_dir in tqdm(person_dirs, desc="MotionBERT persons", unit="person"):
        tqdm.write(f"[MotionBERT] Processing {split_name}/{seq_name}/{person_name}")
        source_kind, source_path = resolve_source(person_dir)
        keypoints_json = person_dir / "keypoints_2d_prediction.json"
        out_dir = motionbert_root / split_name / seq_name / person_name
        out_npy = out_dir / "X3D_prediction.npy"

        if out_npy.exists() and not opts.overwrite:
            skipped += 1
            tqdm.write(f"[MotionBERT] Skipping existing: {split_name}/{seq_name}/{person_name}")
            continue

        try:
            frame_indices = load_frame_indices_from_alphapose(keypoints_json, person_dir)
            saved = infer_one_person(
                json_path=keypoints_json,
                source_kind=source_kind,
                source_path=source_path,
                out_dir=str(out_dir),
                opts=opts,
                args=args,
                model_pos=model_pos,
                WildDetDataset=WildDetDataset,
                flip_data=flip_data,
                render_and_save=render_and_save,
            )
            # A regular load keeps this validation portable on Windows network
            # and sandboxed filesystems, where memmap may reject path resolution.
            x3d = np.load(saved)
            if len(x3d) != len(frame_indices):
                raise RuntimeError(
                    f"Frame index length mismatch: x3d={len(x3d)} vs frame_indices={len(frame_indices)}"
                )
            np.save(out_dir / "frame_indices_prediction.npy", frame_indices)
            (out_dir / "source_json.txt").write_text(str(keypoints_json), encoding="utf-8")
            if source_kind == "video":
                (out_dir / "source_video.txt").write_text(str(source_path), encoding="utf-8")
            else:
                (out_dir / "source_frames_dir.txt").write_text(str(source_path), encoding="utf-8")
            tqdm.write(f"[MotionBERT] Saved: {saved}")
            processed += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"[MotionBERT] Failed {split_name}/{seq_name}/{person_name}: {repr(exc)}")

    print("MotionBERT summary:", {"processed": processed, "skipped": skipped, "failed": failed})


if __name__ == "__main__":
    main()

