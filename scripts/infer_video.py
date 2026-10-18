"""Run the released learned models on one video.

Use --parameters pose, shape, or both.  Configure third-party AlphaPose,
MotionBERT, and SMPL assets once in configs/local.json; the project-specific
pose and shape checkpoints are already selected by configs/example.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video containing one person.")
    parser.add_argument("--parameters", choices=("pose", "shape", "both"), default="both",
                        help="Learned SMPL parameters to infer (default: both).")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "local.json",
                        help="Configured asset paths; defaults to configs/local.json.")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs" / "inference",
                        help="New output directory.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--visualize", nargs="+", choices=("2d", "3d", "mesh"), default=["mesh"],
                        help="Previews to create; default: mesh.")
    return parser.parse_args()


def pipeline_args(args: argparse.Namespace) -> argparse.Namespace:
    """Translate the concise user-facing choices to the full inference CLI."""
    if args.parameters == "pose":
        pose, shape = "learned", "neutral"
    elif args.parameters == "shape":
        # Shape does not depend on the learned pose network.  A kinematic pose
        # is generated only for the optional mesh export.
        pose, shape = "kinematic", "temporal"
    else:
        pose, shape = "learned", "temporal"

    from video_to_smpl.cli import parser as full_parser
    command = [str(args.video), "--config", str(args.config), "--pose", pose, "--shape", shape,
               "--output", str(args.output), "--device", args.device, "--visualize", *args.visualize]
    if args.track_id is not None:
        command += ["--track-id", str(args.track_id)]
    return full_parser().parse_args(command)


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise SystemExit(
            "Missing {}. Copy configs/example.json to configs/local.json and set your AlphaPose, "
            "MotionBERT, and SMPL asset paths.".format(args.config)
        )
    from video_to_smpl.cli import run
    run(pipeline_args(args))


if __name__ == "__main__":
    main()
