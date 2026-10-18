import argparse
import gc
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3DPW AlphaPose preprocessing stage")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alphapose-repo", type=Path, required=True)
    parser.add_argument("--alphapose-cfg", type=Path, required=True)
    parser.add_argument("--alphapose-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--alphapose-conda-env",
        type=str,
        default="",
        help="If set, run AlphaPose demo_inference.py in this conda env for subprocess fallback.",
    )
    parser.add_argument(
        "--alphapose-python",
        type=str,
        default="python",
        help="Python executable used for AlphaPose subprocess fallback when --alphapose-conda-env is not set.",
    )
    parser.add_argument(
        "--preprocessed-train",
        type=Path,
        default=None,
        help="VIBE-format 3DPW training metadata. Defaults to <dataset-root>/preprocessed_train/3dpw_train.npz.",
    )
    parser.add_argument(
        "--preprocessed-test",
        type=Path,
        default=None,
        help="VIBE-format 3DPW test metadata. Defaults to <dataset-root>/preprocessed_test/3dpw_test.npz.",
    )
    parser.add_argument(
        "--preprocessed-val",
        type=Path,
        default=None,
        help="VIBE-format 3DPW validation metadata. Defaults to <dataset-root>/preprocessed_val/3dpw_validation.npz.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-sequence",
        type=str,
        default="",
        help="If set, run only this sequence name (e.g. flat_packBags_00)",
    )
    parser.add_argument(
        "--split-filter",
        type=str,
        default="",
        choices=["", "train", "val", "test"],
        help="If set, process only sequences belonging to this split.",
    )
    parser.add_argument(
        "--keep-shared-raw",
        action="store_true",
        help="Keep shared AlphaPose raw JSON under <sequence>/_shared after per-person split.",
    )
    parser.add_argument(
        "--alphapose-extra-args",
        type=str,
        default="",
        help="Extra args appended to AlphaPose demo_inference.py command",
    )
    parser.add_argument(
        "--no-reuse-model",
        action="store_true",
        help="Disable persistent AlphaPose mode and use subprocess-per-video mode.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="auto",
        choices=["auto", "video", "frames"],
        help="AlphaPose input mode: auto prefers frames when available; video forces <sequence>.mp4; frames forces image_*.jpg.",
    )
    parser.add_argument(
        "--bbox-source",
        type=str,
        default="gt",
        choices=["gt", "detector", "precomputed"],
        help=(
            "Bounding box source for AlphaPose. "
            "'gt' skips detector and uses GT boxes from preprocessed npz. "
            "'precomputed' skips detector and uses cached per-person boxes from --precomputed-bbox-root."
        ),
    )
    parser.add_argument(
        "--precomputed-bbox-root",
        type=Path,
        default=None,
        help=(
            "Root containing cached per-person bbox tracks laid out as "
            "<root>/<split>/<sequence>/person_<id>/bbox_xyxy.npy. "
            "Used when --bbox-source precomputed."
        ),
    )
    return parser.parse_args()


def calculate_min_bounding_box(keypoints: np.ndarray, conf_thresh: float = 0.5):
    valid_kp = keypoints[keypoints[:, 2] >= conf_thresh]
    if valid_kp.shape[0] == 0:
        return None
    min_x = np.min(valid_kp[:, 0])
    min_y = np.min(valid_kp[:, 1])
    max_x = np.max(valid_kp[:, 0])
    max_y = np.max(valid_kp[:, 1])
    return float(min_x), float(min_y), float(max_x), float(max_y)


def _kp_to_bbox_param(kp: np.ndarray, vis_thresh: float) -> Optional[np.ndarray]:
    if kp is None:
        return None
    vis = kp[:, 2] > float(vis_thresh)
    if not np.any(vis):
        return None
    min_pt = np.min(kp[vis, :2], axis=0)
    max_pt = np.max(kp[vis, :2], axis=0)
    person_height = float(np.linalg.norm(max_pt - min_pt))
    if person_height < 0.5:
        return None
    center = (min_pt + max_pt) / 2.0
    scale = 150.0 / person_height
    return np.asarray([float(center[0]), float(center[1]), float(scale)], dtype=np.float64)


def _get_all_bbox_params(kps: np.ndarray, vis_thresh: float) -> Tuple[np.ndarray, int, int]:
    num_to_interpolate = 0
    start_index = -1
    bbox_params = np.empty(shape=(0, 3), dtype=np.float64)
    last_i = -1

    for i, kp in enumerate(kps):
        last_i = i
        bbox_param = _kp_to_bbox_param(kp, vis_thresh=vis_thresh)
        if bbox_param is None:
            num_to_interpolate += 1
            continue

        if start_index == -1:
            start_index = i
            num_to_interpolate = 0

        if num_to_interpolate > 0:
            previous = bbox_params[-1]
            interpolated = np.array(
                [
                    np.linspace(prev, curr, num_to_interpolate + 2)
                    for prev, curr in zip(previous, bbox_param)
                ]
            )
            bbox_params = np.vstack((bbox_params, interpolated.T[1:-1]))
            num_to_interpolate = 0

        bbox_params = np.vstack((bbox_params, bbox_param))

    if last_i < 0:
        return np.empty((0, 3), dtype=np.float64), -1, 0
    return bbox_params, start_index, int(last_i - num_to_interpolate + 1)


def _smooth_bbox_params(bbox_params: np.ndarray, kernel_size: int = 11, sigma: float = 8.0) -> np.ndarray:
    smoothed = np.array([signal.medfilt(param, kernel_size) for param in bbox_params.T]).T
    return np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T


def _build_vibe_style_bboxes_xyxy(person_frames: np.ndarray) -> List[Tuple[float, float, float, float]]:
    # Match VIBE 3DPW preprocessing: get_smooth_bbox_params(..., vis_thresh=0.3, sigma=8),
    # then convert [cx, cy, scale] -> square bbox with side=(150/scale)*1.1.
    n_frames = int(person_frames.shape[0])
    kps = np.asarray(person_frames, dtype=np.float64).transpose(0, 2, 1)  # (T, K, 3)
    bbox_params, start_idx, end_idx = _get_all_bbox_params(kps, vis_thresh=0.3)

    out: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)] * n_frames
    if start_idx < 0 or bbox_params.shape[0] == 0:
        return out

    smoothed = _smooth_bbox_params(bbox_params, kernel_size=11, sigma=8.0)
    valid_len = int(min(smoothed.shape[0], max(0, end_idx - start_idx)))
    if valid_len <= 0:
        return out

    for t in range(valid_len):
        frame_idx = int(start_idx + t)
        if frame_idx < 0 or frame_idx >= n_frames:
            continue
        cx, cy, scale = smoothed[t]
        if abs(float(scale)) <= 1e-12:
            continue
        side = (150.0 / float(scale)) * 1.1
        half = side / 2.0
        out[frame_idx] = (float(cx - half), float(cy - half), float(cx + half), float(cy + half))
    return out


def _bbox_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def _distance(a, b) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def closest_keypoints_to_bbox(keypoint_sets: List[List[float]], true_bbox):
    def calc_bbox_from_flat_keypoints(keypoints_flat: List[float]):
        coords = [(keypoints_flat[i], keypoints_flat[i + 1]) for i in range(0, len(keypoints_flat), 3)]
        x_coords, y_coords = zip(*coords)
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    if true_bbox is None:
        return keypoint_sets[0]

    true_center = _bbox_center(true_bbox)
    best_keypoints = None
    min_dist = float("inf")
    for keypoints in keypoint_sets:
        bbox = calc_bbox_from_flat_keypoints(keypoints)
        center = _bbox_center(bbox)
        dist = _distance(center, true_center)
        if dist < min_dist:
            min_dist = dist
            best_keypoints = keypoints
    return best_keypoints


def filter_duplicates(
    data: List[dict],
    true_bboxes: List[Tuple[float, float, float, float]],
) -> List[dict]:
    grouped_data: "OrderedDict[str, List[dict]]" = OrderedDict()
    for entry in data:
        image_id = entry["image_id"]
        grouped_data.setdefault(image_id, []).append(entry)

    filtered_data: List[dict] = []
    for idx, entries in enumerate(grouped_data.values()):
        true_bbox = true_bboxes[idx] if idx < len(true_bboxes) else None
        if len(entries) == 1:
            filtered_data.append(entries[0])
            continue

        keypoint_sets = [entry["keypoints"] for entry in entries]
        closest_keypoints = closest_keypoints_to_bbox(keypoint_sets, true_bbox)
        for entry in entries:
            if entry["keypoints"] == closest_keypoints:
                filtered_data.append(entry)
                break

    return filtered_data


def _parse_frame_idx(image_id: str) -> Optional[int]:
    name = Path(str(image_id)).name
    stem = Path(name).stem
    digits = re.findall(r"\d+", stem)
    if not digits:
        return None
    return int(digits[-1])


def _keypoint_dim(entries: List[dict]) -> int:
    for e in entries:
        kp = e.get("keypoints", None)
        if isinstance(kp, list) and len(kp) > 0:
            return len(kp)
    return 26 * 3


def densify_by_frame_index(filtered_data: List[dict], expected_frames: int) -> Tuple[List[dict], np.ndarray]:
    keypoint_dim = _keypoint_dim(filtered_data)
    by_frame: Dict[int, dict] = {}

    # First pass: parse explicit frame indices from image_id.
    for e in filtered_data:
        idx = _parse_frame_idx(str(e.get("image_id", "")))
        if idx is None:
            continue
        if idx not in by_frame:
            by_frame[idx] = e

    # Fallback if parsing failed for all entries: use original order as frame index.
    if len(by_frame) == 0:
        for i, e in enumerate(filtered_data):
            if i >= expected_frames:
                break
            by_frame[i] = e

    dense: List[dict] = []
    missing_mask = np.zeros((expected_frames,), dtype=bool)
    for i in range(expected_frames):
        if i in by_frame:
            e = dict(by_frame[i])
            e["frame_idx"] = int(i)
            e["is_missing"] = False
            dense.append(e)
        else:
            missing_mask[i] = True
            dense.append(
                {
                    "image_id": f"image_{i:05d}.jpg",
                    "category_id": 1,
                    "idx": 0,
                    "score": 0.0,
                    "keypoints": [0.0] * keypoint_dim,
                    "frame_idx": int(i),
                    "is_missing": True,
                }
            )
    return dense, missing_mask


def interpolate_missing_keypoints_linear(dense: List[dict], missing_mask: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate missing keypoints for internal gaps only.
    Gaps before the first valid frame or after the last valid frame stay unchanged.
    """
    n_frames = len(dense)
    interpolated_mask = np.zeros((n_frames,), dtype=bool)
    if n_frames == 0:
        return interpolated_mask

    missing_mask = np.asarray(missing_mask, dtype=bool).reshape(-1)
    if len(missing_mask) != n_frames:
        return interpolated_mask

    valid_idx = np.where(~missing_mask)[0]
    if len(valid_idx) < 2:
        return interpolated_mask

    keypoint_dim = _keypoint_dim(dense)
    if keypoint_dim <= 0:
        return interpolated_mask

    keypoints = np.zeros((n_frames, keypoint_dim), dtype=np.float64)
    for i, entry in enumerate(dense):
        kp = entry.get("keypoints", [])
        if not isinstance(kp, list):
            kp = []
        if len(kp) < keypoint_dim:
            kp = kp + [0.0] * (keypoint_dim - len(kp))
        elif len(kp) > keypoint_dim:
            kp = kp[:keypoint_dim]
        keypoints[i] = np.asarray(kp, dtype=np.float64)

    frame_idx = np.arange(n_frames, dtype=np.int32)
    internal_missing = missing_mask & (frame_idx > int(valid_idx[0])) & (frame_idx < int(valid_idx[-1]))
    target_idx = np.where(internal_missing)[0]
    if len(target_idx) == 0:
        return interpolated_mask

    for d in range(keypoint_dim):
        keypoints[target_idx, d] = np.interp(target_idx, valid_idx, keypoints[valid_idx, d])

    interpolated_mask[target_idx] = True
    for i in target_idx:
        dense[i]["keypoints"] = keypoints[i].astype(float).tolist()
        dense[i]["is_interpolated"] = True

    return interpolated_mask


def extract_sequence_name(image_path: str) -> str:
    return Path(image_path.replace("\\", "/")).parent.name


def build_bbox_map(npz_path: Path) -> Dict[str, List[List[Tuple[float, float, float, float]]]]:
    data = np.load(npz_path, allow_pickle=True)
    bbox_map: Dict[str, List[List[Tuple[float, float, float, float]]]] = {}

    for seq_img_names, seq_person_poses in zip(data["all_img_names"], data["all_poses_2d"]):
        seq_name = extract_sequence_name(str(seq_img_names[0]))
        person_boxes_all: List[List[Tuple[float, float, float, float]]] = []

        for person_frames in seq_person_poses:
            frame_boxes = _build_vibe_style_bboxes_xyxy(np.asarray(person_frames, dtype=np.float64))
            person_boxes_all.append(frame_boxes)

        bbox_map[seq_name] = person_boxes_all

    return bbox_map


def _resolve_preprocessed_path(
    supplied: Optional[Path], dataset_root: Path, relative_default: str, split_name: str
) -> Path:
    """Resolve the metadata layout distributed with the 3DPW preprocessing bundle."""
    path = Path(supplied) if supplied is not None else dataset_root / relative_default
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split_name} VIBE-format 3DPW metadata: {path}. "
            f"Pass --preprocessed-{split_name} explicitly when it is stored elsewhere."
        )
    return path


def build_split_maps(args: argparse.Namespace):
    train_path = _resolve_preprocessed_path(
        args.preprocessed_train, args.dataset_root, "preprocessed_train/3dpw_train.npz", "train"
    )
    test_path = _resolve_preprocessed_path(
        args.preprocessed_test, args.dataset_root, "preprocessed_test/3dpw_test.npz", "test"
    )
    val_path = _resolve_preprocessed_path(
        args.preprocessed_val, args.dataset_root, "preprocessed_val/3dpw_validation.npz", "val"
    )
    split_bbox_maps = {
        "train": build_bbox_map(train_path),
        "test": build_bbox_map(test_path),
        "val": build_bbox_map(val_path),
    }
    seq_to_split = {}
    for split, seq_map in split_bbox_maps.items():
        for seq_name in seq_map.keys():
            seq_to_split[seq_name] = split
    return split_bbox_maps, seq_to_split


def load_precomputed_bbox_map(root: Path, split: str) -> Dict[str, List[List[Tuple[float, float, float, float]]]]:
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Missing precomputed bbox split root: {split_root}")

    bbox_map: Dict[str, List[List[Tuple[float, float, float, float]]]] = {}
    for seq_dir in sorted(split_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        person_boxes_all: List[List[Tuple[float, float, float, float]]] = []
        for person_dir in sorted(seq_dir.glob("person_*")):
            bbox_npy = person_dir / "bbox_xyxy.npy"
            if not bbox_npy.exists():
                continue
            bbox_np = np.asarray(np.load(bbox_npy), dtype=np.float64)
            if bbox_np.ndim != 2 or bbox_np.shape[1] != 4:
                raise ValueError(f"Expected bbox_xyxy.npy shape [T,4], got {bbox_np.shape} at {bbox_npy}")
            person_boxes = [tuple(float(v) for v in row.tolist()) for row in bbox_np]
            person_boxes_all.append(person_boxes)
        if person_boxes_all:
            bbox_map[seq_dir.name] = person_boxes_all
    if not bbox_map:
        raise ValueError(f"No precomputed bbox tracks found under: {split_root}")
    return bbox_map


def sequence_folders(dataset_root: Path) -> List[Path]:
    seq_root = dataset_root / "imageFiles" / "imageFiles"
    if not seq_root.exists():
        return []
    return sorted([p for p in seq_root.iterdir() if p.is_dir()])


def main_video_for_sequence_folder(seq_dir: Path) -> Optional[Path]:
    seq_name = seq_dir.name
    candidate = seq_dir / f"{seq_name}.mp4"
    if candidate.exists():
        return candidate
    return None


def has_sequence_frames(seq_dir: Path) -> bool:
    return (seq_dir / "image_00000.jpg").exists() or any(seq_dir.glob("image_*.jpg"))


def _bbox_is_valid_xyxy(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    min_area: float = 4.0,
) -> bool:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if not np.isfinite([x1, y1, x2, y2]).all():
        return False
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width - 1)))
    y2 = max(0.0, min(y2, float(height - 1)))
    w = x2 - x1
    h = y2 - y1
    return (w > 0.0) and (h > 0.0) and (w * h >= float(min_area))


def _clip_bbox_xyxy(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width - 1)))
    y2 = max(0.0, min(y2, float(height - 1)))
    if x2 <= x1:
        x2 = min(float(width - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height - 1), y1 + 1.0)
    return (x1, y1, x2, y2)


def run_alphapose_inference(
    input_path: Path,
    input_mode: str,
    out_dir: Path,
    args: argparse.Namespace,
    persistent_engine=None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json = out_dir / "alphapose-results_prediction_raw.json"

    if raw_json.exists() and not args.overwrite:
        print(f"Skipping inference (exists): {raw_json}")
        return raw_json

    if persistent_engine is not None:
        persistent_engine.run_source(input_path=input_path, input_mode=input_mode, out_dir=out_dir)
        default_json = out_dir / "alphapose-results.json"
        if not default_json.exists():
            raise FileNotFoundError(f"Expected AlphaPose output not found: {default_json}")
        default_json.replace(raw_json)
        return raw_json

    if args.alphapose_conda_env:
        cmd = ["conda", "run", "-n", str(args.alphapose_conda_env), "python", "scripts/demo_inference.py"]
    else:
        cmd = [str(args.alphapose_python), "scripts/demo_inference.py"]
    cmd.extend([
        "--cfg",
        str(args.alphapose_cfg),
        "--checkpoint",
        str(args.alphapose_checkpoint),
        "--outdir",
        str(out_dir),
    ])
    if input_mode == "frames":
        cmd.extend(["--indir", str(input_path)])
    else:
        cmd.extend(["--video", str(input_path)])
    if args.alphapose_extra_args.strip():
        cmd.extend(shlex.split(args.alphapose_extra_args))

    print("Running AlphaPose:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(args.alphapose_repo))
    start_ts = time.time()
    last_heartbeat = start_ts
    heartbeat_sec = 60.0
    while True:
        ret = proc.poll()
        now = time.time()
        if ret is not None:
            break
        if now - last_heartbeat >= heartbeat_sec:
            elapsed = int(now - start_ts)
            print(f"[AlphaPose] still running {input_path.name} ... {elapsed}s elapsed")
            last_heartbeat = now
        time.sleep(1.0)

    if ret != 0:
        raise RuntimeError(f"AlphaPose failed for {input_path} with exit code {ret}")

    default_json = out_dir / "alphapose-results.json"
    if not default_json.exists():
        raise FileNotFoundError(f"Expected AlphaPose output not found: {default_json}")

    default_json.replace(raw_json)
    return raw_json


def _parse_alphapose_extra_args(extra: str) -> SimpleNamespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--detbatch", type=int, default=4)
    parser.add_argument("--posebatch", type=int, default=80)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--format", type=str, default=None)
    parser.add_argument("--detector", type=str, default="yolo")
    parser.add_argument("--qsize", type=int, default=64)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--debug", action="store_true")
    known, _ = parser.parse_known_args(shlex.split(extra) if extra else [])
    return known


class _PersistentAlphaPoseEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._initialized = False
        self._repo_cwd = str(args.alphapose_repo)
        self._orig_cwd = os.getcwd()

    def initialize(self) -> None:
        if self._initialized:
            return
        os.chdir(self._repo_cwd)
        if self._repo_cwd not in sys.path:
            sys.path.insert(0, self._repo_cwd)

        import torch
        from alphapose.models import builder
        from alphapose.utils.config import update_config
        from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL

        extra = _parse_alphapose_extra_args(self.args.alphapose_extra_args)
        opt = SimpleNamespace()
        opt.cfg = str(self.args.alphapose_cfg)
        opt.checkpoint = str(self.args.alphapose_checkpoint)
        opt.sp = True if os.name == "nt" else False
        opt.detector = extra.detector
        opt.outputpath = ""
        opt.save_img = False
        opt.vis = False
        opt.showbox = False
        opt.profile = False
        opt.format = extra.format
        opt.min_box_area = 0
        opt.detbatch = int(extra.detbatch)
        opt.posebatch = int(extra.posebatch)
        opt.eval = False
        opt.gpus = [int(i) for i in str(extra.gpus).split(",")] if torch.cuda.device_count() >= 1 else [-1]
        opt.qsize = int(extra.qsize)
        opt.flip = bool(extra.flip)
        opt.debug = bool(extra.debug)
        opt.video = ""
        opt.webcam = -1
        opt.save_video = False
        opt.vis_fast = False
        opt.pose_flow = False
        opt.pose_track = False
        opt.tracking = False
        opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")
        opt.detbatch = opt.detbatch * max(1, len(opt.gpus))
        opt.posebatch = opt.posebatch * max(1, len(opt.gpus))

        cfg = update_config(opt.cfg)
        detector = None
        if self.args.bbox_source == "detector":
            from detector.apis import get_detector

            detector = get_detector(opt)
        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print("Loading AlphaPose model once from", opt.checkpoint)
        pose_model.load_state_dict(torch.load(opt.checkpoint, map_location=opt.device))
        pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        if len(opt.gpus) > 1:
            pose_model = torch.nn.DataParallel(pose_model, device_ids=opt.gpus).to(opt.device)
        else:
            pose_model.to(opt.device)
        pose_model.eval()

        self.torch = torch
        self.getTime = __import__("alphapose.utils.vis", fromlist=["getTime"]).getTime
        self.flip = __import__("alphapose.utils.transforms", fromlist=["flip"]).flip
        self.flip_heatmap = __import__("alphapose.utils.transforms", fromlist=["flip_heatmap"]).flip_heatmap
        self.DetectionLoader = __import__(
            "alphapose.utils.detector", fromlist=["DetectionLoader"]
        ).DetectionLoader
        self.DataWriter = __import__("alphapose.utils.writer", fromlist=["DataWriter"]).DataWriter

        self.cfg = cfg
        self.opt = opt
        self.detector = detector
        self.pose_model = pose_model
        self.pose_dataset = pose_dataset

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA
        if cfg.DATA_PRESET.TYPE == "simple":
            self.transformation = SimpleTransform(
                pose_dataset,
                scale_factor=0,
                input_size=self._input_size,
                output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
                rot=0,
                sigma=self._sigma,
                train=False,
                add_dpg=False,
                gpu_device=self.opt.device,
            )
        elif cfg.DATA_PRESET.TYPE == "simple_smpl":
            from easydict import EasyDict as edict

            dummy_set = edict(
                {
                    "joint_pairs_17": None,
                    "joint_pairs_24": None,
                    "joint_pairs_29": None,
                    "bbox_3d_shape": (2.2, 2.2, 2.2),
                }
            )
            self.transformation = SimpleTransform3DSMPL(
                dummy_set,
                scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR,
                sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False,
                add_dpg=False,
                loss_type=cfg.LOSS["TYPE"],
            )
        else:
            raise ValueError(f"Unsupported DATA_PRESET.TYPE: {cfg.DATA_PRESET.TYPE}")
        self._initialized = True

    def run_source(self, input_path: Path, input_mode: str, out_dir: Path) -> None:
        self.initialize()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.opt.outputpath = str(out_dir)
        output_json = out_dir / "alphapose-results.json"
        if output_json.exists():
            output_json.unlink()

        # Keep queues bounded to avoid multi-GB RAM growth on long videos.
        queue_size = int(max(8, min(self.opt.qsize, 64)))
        input_source = str(input_path)
        if input_mode == "frames":
            self.opt.inputpath = str(input_path)
            self.opt.inputlist = ""
            self.opt.inputimg = ""
            im_names = [p.name for p in sorted(Path(input_path).glob("image_*.jpg"))]
            if len(im_names) == 0:
                raise FileNotFoundError(f"No image_*.jpg frames found in sequence folder: {input_path}")
            input_source = im_names
        det_loader = self.DetectionLoader(
            input_source,
            self.detector,
            self.cfg,
            self.opt,
            batchSize=self.opt.detbatch,
            mode="image" if input_mode == "frames" else "video",
            queueSize=queue_size,
        )
        det_loader.start()
        writer = self.DataWriter(self.cfg, self.opt, save_video=False, queueSize=queue_size).start()

        batch_size = self.opt.posebatch
        if self.opt.flip:
            batch_size = max(1, int(batch_size / 2))

        data_len = det_loader.length
        for _ in tqdm(range(data_len), leave=False, desc="AlphaPose frames", unit="frame"):
            with self.torch.no_grad():
                inps, orig_img, im_name, boxes, scores, ids, cropped_boxes = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue

                inps = inps.to(self.opt.device)
                datalen = inps.size(0)
                num_batches = datalen // batch_size + (1 if (datalen % batch_size) else 0)
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batch_size : min((j + 1) * batch_size, datalen)]
                    if self.opt.flip:
                        inps_j = self.torch.cat((inps_j, self.flip(inps_j)))
                    hm_j = self.pose_model(inps_j)
                    if self.opt.flip:
                        hm_j_flip = self.flip_heatmap(
                            hm_j[int(len(hm_j) / 2) :], self.pose_dataset.joint_pairs, shift=True
                        )
                        hm_j = (hm_j[0 : int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = self.torch.cat(hm).cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)

        while writer.running():
            time.sleep(1)
            print(
                f"[AlphaPose] Rendering remaining queue for {input_path.name}: {writer.count()}",
                end="\r",
            )
        writer.stop()
        det_loader.terminate()
        det_loader.stop()
        print("")
        del writer
        del det_loader
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        gc.collect()

    def run_person_from_gt_boxes(
        self,
        input_path: Path,
        input_mode: str,
        person_boxes: List[Tuple[float, float, float, float]],
        out_dir: Path,
    ) -> Path:
        self.initialize()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.opt.outputpath = str(out_dir)
        raw_json = out_dir / "alphapose-results_prediction_raw.json"
        if raw_json.exists():
            raw_json.unlink()

        queue_size = int(max(8, min(int(self.opt.qsize), 64)))
        writer = self.DataWriter(
            self.cfg,
            self.opt,
            save_video=False,
            queueSize=queue_size,
            outputfile=raw_json.name,
        ).start()

        expected_frames = len(person_boxes)
        stream = None
        if input_mode == "video":
            stream = cv2.VideoCapture(str(input_path))
            if not stream.isOpened():
                raise FileNotFoundError(f"Cannot open video source for GT-bbox AlphaPose: {input_path}")

        try:
            for frame_idx in tqdm(
                range(expected_frames),
                leave=False,
                desc="3DPW AlphaPose GT-bbox frames",
                unit="frame",
            ):
                if input_mode == "frames":
                    frame_path = Path(input_path) / f"image_{frame_idx:05d}.jpg"
                    if not frame_path.exists():
                        continue
                    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                else:
                    assert stream is not None
                    grabbed, bgr = stream.read()
                    if not grabbed:
                        break
                if bgr is None:
                    continue

                orig_img = bgr[:, :, ::-1]  # RGB
                h, w = int(orig_img.shape[0]), int(orig_img.shape[1])
                bbox_raw = person_boxes[frame_idx]
                if not _bbox_is_valid_xyxy(bbox_raw, width=w, height=h):
                    continue
                bbox = _clip_bbox_xyxy(bbox_raw, width=w, height=h)

                boxes = self.torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=self.torch.float32)
                inps = self.torch.zeros((1, 3, int(self._input_size[0]), int(self._input_size[1])), dtype=self.torch.float32)
                cropped_boxes = self.torch.zeros((1, 4), dtype=self.torch.float32)

                inps[0], cropped_box = self.transformation.test_transform(orig_img, boxes[0])
                cropped_boxes[0] = self.torch.tensor(cropped_box, dtype=self.torch.float32)

                with self.torch.no_grad():
                    inps = inps.to(self.opt.device)
                    hm = self.pose_model(inps)
                    if self.opt.flip:
                        hm_flip = self.flip_heatmap(
                            self.pose_model(self.flip(inps)),
                            self.pose_dataset.joint_pairs,
                            shift=True,
                        )
                        hm = (hm + hm_flip) / 2
                    hm = hm.cpu()

                scores = self.torch.ones((1, 1), dtype=self.torch.float32)
                ids = self.torch.zeros((1, 1), dtype=self.torch.float32)
                im_name = f"image_{frame_idx:05d}.jpg"
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
        finally:
            if stream is not None:
                stream.release()
            writer.stop()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            gc.collect()

        if not raw_json.exists():
            raise FileNotFoundError(f"Expected AlphaPose GT-bbox output not found: {raw_json}")
        return raw_json

    def close(self) -> None:
        if os.getcwd() != self._orig_cwd:
            os.chdir(self._orig_cwd)


def main() -> None:
    args = parse_args()
    if args.bbox_source == "precomputed":
        if args.precomputed_bbox_root is None:
            raise ValueError("--precomputed-bbox-root is required when --bbox-source precomputed.")
        split_bbox_maps = {
            str(args.split_filter or "train"): {},
            str(args.split_filter or "test"): {},
            str(args.split_filter or "val"): {},
        }
        seq_to_split = {}
        requested_splits = [args.split_filter] if args.split_filter else ["train", "test", "val"]
        split_bbox_maps = {}
        for split_name in requested_splits:
            seq_map = load_precomputed_bbox_map(args.precomputed_bbox_root, split_name)
            split_bbox_maps[split_name] = seq_map
            for seq_name in seq_map.keys():
                seq_to_split[seq_name] = split_name
    else:
        split_bbox_maps, seq_to_split = build_split_maps(args)

    seq_dirs = sequence_folders(args.dataset_root)
    if not seq_dirs:
        raise FileNotFoundError("No sequence folders found under dataset root.")

    work_items = []
    for seq_dir in seq_dirs:
        sequence = seq_dir.name
        if args.only_sequence and sequence != args.only_sequence:
            continue
        split = seq_to_split.get(sequence)
        if split is None:
            print(f"Skipping sequence not found in split maps: {sequence}")
            continue
        if args.split_filter and split != args.split_filter:
            continue

        input_mode = "frames"
        input_path = seq_dir
        if args.input_mode == "video":
            video_path = main_video_for_sequence_folder(seq_dir)
            if video_path is None:
                print(f"Skipping sequence without main video <sequence>.mp4: {sequence}")
                continue
            input_mode = "video"
            input_path = video_path
        elif args.input_mode == "frames":
            if not has_sequence_frames(seq_dir):
                print(f"Skipping sequence without image_*.jpg frames: {sequence}")
                continue
            input_mode = "frames"
            input_path = seq_dir
        else:
            # auto mode: prefer original frames, fallback to video
            if not has_sequence_frames(seq_dir):
                video_path = main_video_for_sequence_folder(seq_dir)
                if video_path is None:
                    print(f"Skipping sequence without image_*.jpg frames or main video <sequence>.mp4: {sequence}")
                    continue
                input_mode = "video"
                input_path = video_path
        seq_person_bboxes = split_bbox_maps[split][sequence]
        if len(seq_person_bboxes) <= 0:
            print(f"Skipping {sequence}: no persons in GT 2D boxes")
            continue
        work_items.append((input_path, input_mode, split, sequence, seq_person_bboxes))

    print("AlphaPose work items:", len(work_items))

    persistent_engine = None
    if args.bbox_source in {"gt", "precomputed"}:
        try:
            persistent_engine = _PersistentAlphaPoseEngine(args)
            persistent_engine.initialize()
            if args.bbox_source == "gt":
                print("AlphaPose mode: persistent GT-bbox inference (detector skipped)")
            else:
                print("AlphaPose mode: persistent precomputed-bbox inference (detector skipped)")
        except Exception as exc:
            raise RuntimeError(
                f"{args.bbox_source} AlphaPose requires persistent engine init, but init failed: {repr(exc)}"
            )
    elif not args.no_reuse_model:
        try:
            persistent_engine = _PersistentAlphaPoseEngine(args)
            persistent_engine.initialize()
            print("AlphaPose mode: persistent (single model load)")
        except Exception as exc:
            print(f"Persistent AlphaPose init failed, falling back to subprocess mode: {repr(exc)}")
            persistent_engine = None
    else:
        print("AlphaPose mode: subprocess-per-video (--no-reuse-model)")

    for input_path, input_mode, split, sequence, seq_person_bboxes in tqdm(
        work_items, desc="AlphaPose sequences", unit="sequence"
    ):
        tqdm.write(f"[AlphaPose] Processing sequence {split}/{sequence}")
        sequence_root = args.output_root / "predictions" / "alphapose" / split / sequence
        sequence_root.mkdir(parents=True, exist_ok=True)
        if args.bbox_source in {"gt", "precomputed"}:
            if persistent_engine is None:
                raise RuntimeError(f"{args.bbox_source} AlphaPose requested but persistent engine is not available.")
            # Build one person output folder per target-person bbox track, running pose directly on provided boxes.
            for person_idx, person_boxes in enumerate(seq_person_bboxes):
                person_id = person_idx + 1
                person_dir = sequence_root / f"person_{person_id}"
                person_dir.mkdir(parents=True, exist_ok=True)

                raw_json = persistent_engine.run_person_from_gt_boxes(
                    input_path=input_path,
                    input_mode=input_mode,
                    person_boxes=person_boxes,
                    out_dir=person_dir,
                )
                data = json.loads(raw_json.read_text(encoding="utf-8"))
                expected_frames = len(person_boxes)
                dense_filtered, missing_mask = densify_by_frame_index(data, expected_frames=expected_frames)
                interpolated_mask = interpolate_missing_keypoints_linear(dense_filtered, missing_mask)

                final_json = person_dir / "keypoints_2d_prediction.json"
                final_json.write_text(json.dumps(dense_filtered, indent=2), encoding="utf-8")
                np.save(person_dir / "frame_indices_prediction.npy", np.arange(expected_frames, dtype=np.int32))
                np.save(person_dir / "frame_missing_mask_prediction.npy", missing_mask)
                np.save(person_dir / "frame_interpolated_mask_prediction.npy", interpolated_mask)
                if input_mode == "video":
                    (person_dir / "source_video.txt").write_text(str(input_path), encoding="utf-8")
                else:
                    (person_dir / "source_frames_dir.txt").write_text(str(input_path), encoding="utf-8")
                    maybe_video = main_video_for_sequence_folder(Path(input_path))
                    if maybe_video is not None:
                        (person_dir / "source_video.txt").write_text(str(maybe_video), encoding="utf-8")

                if not args.keep_shared_raw and raw_json.exists():
                    try:
                        raw_json.unlink()
                    except Exception:
                        pass

                mode_label = "GT-bbox" if args.bbox_source == "gt" else "Precomputed-bbox"
                tqdm.write(
                    f"[AlphaPose][{mode_label}] Saved {split}/{sequence}/person_{person_id}: "
                    f"frames={expected_frames}, missing={int(missing_mask.sum())}, "
                    f"interpolated={int(interpolated_mask.sum())}"
                )
        else:
            shared_dir = sequence_root / "_shared"
            shared_dir.mkdir(parents=True, exist_ok=True)

            raw_json = run_alphapose_inference(
                input_path=input_path,
                input_mode=input_mode,
                out_dir=shared_dir,
                args=args,
                persistent_engine=persistent_engine,
            )
            data = json.loads(raw_json.read_text(encoding="utf-8"))
            # Build one person output folder per GT person track.
            for person_idx, person_boxes in enumerate(seq_person_bboxes):
                person_id = person_idx + 1
                person_dir = sequence_root / f"person_{person_id}"
                person_dir.mkdir(parents=True, exist_ok=True)

                filtered = filter_duplicates(data, person_boxes)
                expected_frames = len(person_boxes)
                dense_filtered, missing_mask = densify_by_frame_index(filtered, expected_frames=expected_frames)
                interpolated_mask = interpolate_missing_keypoints_linear(dense_filtered, missing_mask)

                final_json = person_dir / "keypoints_2d_prediction.json"
                final_json.write_text(json.dumps(dense_filtered, indent=2), encoding="utf-8")
                np.save(person_dir / "frame_indices_prediction.npy", np.arange(expected_frames, dtype=np.int32))
                np.save(person_dir / "frame_missing_mask_prediction.npy", missing_mask)
                np.save(person_dir / "frame_interpolated_mask_prediction.npy", interpolated_mask)
                if input_mode == "video":
                    (person_dir / "source_video.txt").write_text(str(input_path), encoding="utf-8")
                else:
                    (person_dir / "source_frames_dir.txt").write_text(str(input_path), encoding="utf-8")
                    maybe_video = main_video_for_sequence_folder(Path(input_path))
                    if maybe_video is not None:
                        (person_dir / "source_video.txt").write_text(str(maybe_video), encoding="utf-8")
                tqdm.write(
                    f"[AlphaPose] Saved {split}/{sequence}/person_{person_id}: "
                    f"frames={expected_frames}, missing={int(missing_mask.sum())}, "
                    f"interpolated={int(interpolated_mask.sum())}"
                )
            if not args.keep_shared_raw:
                try:
                    if raw_json.exists():
                        raw_json.unlink()
                    if not any(shared_dir.iterdir()):
                        shared_dir.rmdir()
                except Exception:
                    pass

    if persistent_engine is not None:
        persistent_engine.close()


if __name__ == "__main__":
    main()

