"""Prepare 3DPW intermediates, train one learned model, and evaluate it locally.

Run this from any Python environment.  The AlphaPose and MotionBERT stages are
started with their own Python executables, allowing their dependencies to stay
isolated.  Training and evaluation use the MotionBERT executable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


PROJECT = Path(__file__).absolute().parent.parent
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("pose", "shape"), help="Learned model to train.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="3DPW dataset root.")
    parser.add_argument(
        "--vibe-root", type=Path, required=True,
        help="Folder containing 3dpw_train_db.npz, 3dpw_val_db.npz, and 3dpw_test_db.npz.",
    )
    parser.add_argument("--pipeline-root", type=Path, required=True,
                        help="Where AlphaPose and MotionBERT predictions are stored or created.")
    parser.add_argument("--runs-root", type=Path, default=PROJECT / "training-runs")
    parser.add_argument("--run-name", default="", help="Defaults to <model>_3dpw_motionbert.")
    parser.add_argument("--alphapose-python", type=Path, required=True)
    parser.add_argument("--motionbert-python", type=Path, required=True)
    parser.add_argument("--alphapose-repo", type=Path, required=True)
    parser.add_argument("--alphapose-cfg", type=Path, required=True)
    parser.add_argument("--alphapose-checkpoint", type=Path, required=True)
    parser.add_argument("--motionbert-repo", type=Path, required=True)
    parser.add_argument("--smpl-data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--activation", choices=("relu", "elu"), default="elu",
                        help="Pose-only activation; ignored when --model shape.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for shape training.")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Recreate AlphaPose and MotionBERT predictions even if they are present.")
    parser.add_argument("--overwrite", action="store_true", help="Allow reuse of an existing training-run directory.")
    return parser.parse_args()


def _python_environment(executable: Path) -> dict:
    """Make a direct conda Python invocation work on Windows as well."""
    executable = executable.absolute()
    root = executable.parent
    env = os.environ.copy()
    extra = [str(root), str(root / "Library" / "bin"), str(root / "Scripts"), str(root / "DLLs")]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    env["CONDA_PREFIX"] = str(root)
    return env


def _run(executable: Path, arguments: Iterable[object]) -> None:
    command = [str(executable)] + [str(item) for item in arguments]
    print("\n[run]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=str(PROJECT), env=_python_environment(executable), check=True)


def _has_alphapose_outputs(root: Path) -> bool:
    return all(any((root / "predictions" / "alphapose" / split).rglob("keypoints_2d_prediction.json"))
               for split in SPLITS)


def _has_motionbert_outputs(root: Path) -> bool:
    return all(any((root / "predictions" / "motionbert" / split).rglob("X3D_prediction.npy"))
               for split in SPLITS)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError("Missing {}: {}".format(label, path))


def validate_inputs(args: argparse.Namespace) -> None:
    for split in SPLITS:
        _require_file(args.vibe_root / "3dpw_{}_db.npz".format(split), "VIBE {} database".format(split))
    for path, label in (
        (args.alphapose_python, "AlphaPose Python executable"),
        (args.motionbert_python, "MotionBERT Python executable"),
        (args.alphapose_cfg, "AlphaPose configuration"),
        (args.alphapose_checkpoint, "AlphaPose checkpoint"),
        (args.smpl_data_root / "J_regressor_h36m_correct.npy", "H36M joint regressor"),
    ):
        _require_file(path, label)
    if not args.alphapose_repo.is_dir():
        raise NotADirectoryError("Missing AlphaPose checkout: {}".format(args.alphapose_repo))
    if not args.motionbert_repo.is_dir():
        raise NotADirectoryError("Missing MotionBERT checkout: {}".format(args.motionbert_repo))


def prepare_predictions(args: argparse.Namespace) -> None:
    needs_motionbert = args.force_preprocess or not _has_motionbert_outputs(args.pipeline_root)
    if not needs_motionbert:
        print("[prepare] Reusing existing AlphaPose and MotionBERT predictions.", flush=True)
        return

    needs_alphapose = args.force_preprocess or not _has_alphapose_outputs(args.pipeline_root)
    if needs_alphapose:
        alpha_arguments: List[object] = [
            PROJECT / "scripts" / "preprocess_3dpw_alphapose.py",
            "--dataset-root", args.dataset_root,
            "--output-root", args.pipeline_root,
            "--alphapose-repo", args.alphapose_repo,
            "--alphapose-cfg", args.alphapose_cfg,
            "--alphapose-checkpoint", args.alphapose_checkpoint,
            "--bbox-source", "gt",
            "--input-mode", "frames",
        ]
        if args.force_preprocess:
            alpha_arguments += ["--overwrite"]
        _run(args.alphapose_python, alpha_arguments)
    else:
        print("[prepare] Reusing existing AlphaPose predictions.", flush=True)

    motionbert_arguments: List[object] = [
        PROJECT / "scripts" / "preprocess_3dpw_motionbert.py",
        "--output-root", args.pipeline_root,
        "--motionbert-repo", args.motionbert_repo,
    ]
    if args.force_preprocess:
        motionbert_arguments += ["--overwrite"]
    _run(args.motionbert_python, motionbert_arguments)
    if not _has_motionbert_outputs(args.pipeline_root):
        raise RuntimeError("MotionBERT finished without outputs for every 3DPW split.")


def train_and_evaluate(args: argparse.Namespace) -> Path:
    run_name = args.run_name or "{}_3dpw_motionbert".format(args.model)
    run_dir = args.runs_root / run_name
    train_common: List[object] = ["--vibe-root", args.vibe_root, "--output-root", args.runs_root,
                                  "--run-name", run_name, "--lr", args.lr]
    if args.epochs is not None:
        train_common += ["--epochs", args.epochs]
    if args.batch_size is not None:
        train_common += ["--batch-size", args.batch_size]
    if args.overwrite:
        train_common += ["--overwrite"]

    motionbert_predictions = args.pipeline_root / "predictions" / "motionbert"
    if args.model == "pose":
        _run(args.motionbert_python, [
            PROJECT / "scripts" / "train_pose_only_smplr.py",
            "--motionbert-repo", args.motionbert_repo,
            "--smpl-data-root", args.smpl_data_root,
            "--motionbert-predictions-root", motionbert_predictions,
            "--dataset-root", args.dataset_root,
            "--target-pose-space", "incam",
            "--activation", args.activation,
        ] + train_common)
        evaluator = PROJECT / "scripts" / "evaluate_pose_only_smplr_motionbert_vibe.py"
        evaluation_args: List[object] = [
            "--motionbert-root", motionbert_predictions / "test",
            "--vibe-test-db", args.vibe_root / "3dpw_test_db.npz",
            "--vibe-h36m-regressor", args.smpl_data_root / "J_regressor_h36m_correct.npy",
            "--motionbert-repo", args.motionbert_repo,
            "--smpl-data-root", args.smpl_data_root,
            "--checkpoint", run_dir / "checkpoints" / "best_model.pt",
            "--output", run_dir / "evaluation.json",
        ]
    else:
        shape_args: List[object] = [
            PROJECT / "scripts" / "train_shape_only_res_tcn.py",
            "--motionbert-predictions-root", motionbert_predictions,
        ] + train_common
        if args.cpu:
            shape_args += ["--cpu"]
        _run(args.motionbert_python, shape_args)
        evaluator = PROJECT / "scripts" / "evaluate_shape_only_res_tcn_vibe.py"
        evaluation_args = [
            "--motionbert-root", motionbert_predictions / "test",
            "--vibe-test-db", args.vibe_root / "3dpw_test_db.npz",
            "--motionbert-repo", args.motionbert_repo,
            "--smpl-data-root", args.smpl_data_root,
            "--checkpoint", run_dir / "checkpoints" / "best_model.pt",
            "--output", run_dir / "evaluation.json",
        ]

    _run(args.motionbert_python, [evaluator] + evaluation_args)
    result = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    (run_dir / "workflow.json").write_text(json.dumps({
        "model": args.model,
        "pipeline_root": str(args.pipeline_root),
        "vibe_root": str(args.vibe_root),
        "evaluation": result,
    }, indent=2), encoding="utf-8")
    return run_dir / "evaluation.json"


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.pipeline_root.mkdir(parents=True, exist_ok=True)
    prepare_predictions(args)
    result_path = train_and_evaluate(args)
    print("\nCompleted. Evaluation: {}".format(result_path), flush=True)


if __name__ == "__main__":
    main()
