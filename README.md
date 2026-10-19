# Project for Bachelors Thesis in Artificial Intelligence

**Video → AlphaPose (HALPE-26) → MotionBERT (H36M-17) → SMPL pose + shape**

> [!IMPORTANT]
> ## [Read the accompanying Bachelor's thesis](docs/Monocular_Human_Mesh_Reconstruction_from_Sparse_3D_Keypoints.pdf)
> **Monocular Human Mesh Reconstruction from Sparse 3D Keypoints**<br>

This repository is the accompanying code for the thesis. It turns a single-person video into AlphaPose 2D keypoints, lifts them to MotionBERT 3D keypoints, and estimates SMPL pose and shape parameters with learned regressors or a per-frame optimization baseline.

**External projects and data:** [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) · [MotionBERT](https://github.com/Walter0807/MotionBERT) · [3DPW dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

https://github.com/user-attachments/assets/fc9e1649-e26f-4ee7-aa43-ad0a5fd82ad3

*Input video, root-centered MotionBERT skeleton, and SMPL reconstruction.*

## What is included

| Component | Choices |
| --- | --- |
| Pose | Adam optimization or learned pose-only SMPLR regressor |
| Shape | Neutral body shape or learned temporal shape regressor |
| Visualization | 2D keypoints, MotionBERT skeleton, and SMPL mesh video |
| Training and evaluation | 3DPW preprocessing, training, and VIBE-style evaluation |

The recreated learned pose and shape checkpoints are included. AlphaPose, MotionBERT, SMPL assets, and dataset files must be obtained separately under their respective licenses.

## Reported 3DPW metrics

All errors are in millimetres; lower is better. These are the results reported in the accompanying thesis using its VIBE-style 3DPW evaluation protocol.

| Evaluation setting | Method | PA-MPJPE ↓ | MPJPE ↓ | PVE ↓ |
| --- | --- | ---: | ---: | ---: |
| Pose only, neutral shape | Optimization | 80.12 | 132.44 | 164.26 |
| Pose only, neutral shape | Adapted SMPLR | **62.11** | **95.54** | **111.27** |
| Shape only, neutral pose | Temporal TCN | — | — | **26.84** |
| Full pipeline | Adapted SMPLR + TCN | 63.81 | 101.84 | 120.62 |

See Tables 4.1-4.3 in the [thesis](docs/Monocular_Human_Mesh_Reconstruction_from_Sparse_3D_Keypoints.pdf) for the full comparison, including VIBE baselines.

## Quick start

Create the verified `motionbert` environment, install this package, then copy the configuration template and set the paths to AlphaPose, MotionBERT, and SMPL assets:

```sh
conda env create -f environment.yml
conda activate motionbert
python -m pip install -e . --no-build-isolation
cp configs/example.json configs/local.json
```

Run a video with the included learned pose and shape checkpoints:

```sh
python scripts/infer_video.py input.mp4 --parameters both --config configs/local.json --output outputs/demo
```

Use `--parameters pose`, `shape`, or `both`. A mesh preview is created by default; pass `--visualize 2d 3d mesh` for every preview. The `video-to-smpl` command also supports `--pose optimization|learned` and `--shape neutral|temporal`.

## Train and evaluate on 3DPW

Place the dataset in [`data/3DPW`](data/README.md) and the VIBE split databases in `data/vibe_db`. The tracked [`data/README.md`](data/README.md) documents the required layout; downloaded data remains ignored by Git.

`run_3dpw_training.py` creates AlphaPose and MotionBERT intermediates when absent, then trains and evaluates the selected model. Provide the paths to your AlphaPose and MotionBERT installations and checkpoints:

```sh
python scripts/run_3dpw_training.py \
  --model pose \
  --dataset-root data/3DPW \
  --vibe-root data/vibe_db \
  --pipeline-root outputs/3dpw \
  --runs-root training-runs \
  --run-name pose_3dpw \
  --alphapose-python /path/to/alphapose/python \
  --motionbert-python /path/to/motionbert/python \
  --alphapose-repo /path/to/AlphaPose \
  --alphapose-cfg /path/to/AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
  --alphapose-checkpoint /path/to/halpe26_fast_res50_256x192.pth \
  --motionbert-repo /path/to/MotionBERT \
  --smpl-data-root /path/to/MotionBERT/data/mesh
```

Use `--model pose` to train the pose-only SMPLR regressor, or `--model shape` to train the temporal shape regressor. Both modes preprocess 3DPW when the required AlphaPose and MotionBERT outputs are absent, then write the selected checkpoint and `evaluation.json` under `training-runs/<run-name>`.

## Network architectures

### Pose-only SMPLR regressor

The pose network maps 48 root-centered MotionBERT bone-direction values to 24 SMPL joint rotations, adapted from [SMPLR: Deep SMPL Reverse for 3D Human Pose and Shape Recovery](https://arxiv.org/abs/1812.10766) by Madadi, Bertiche, and Escalera.

![Pose-only SMPLR regressor architecture](assets/pose_network_architecture.png)

### Temporal shape regressor

The shape network processes non-root-centered MotionBERT keypoint sequences and predicts ten SMPL shape coefficients. Its temporal-convolution design is based on [Temporal Convolutional Networks for Action Segmentation and Detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf) by Lea, Vidal, Reiter, and Hager.

![Temporal shape regressor architecture](assets/shape_network_architecture.png)

## Notes

- This pipeline is designed for one person per video. Long occlusions and identity crossings remain unreliable.
- Outputs include `smpl_params.npz`, SMPL vertices and faces, MotionBERT coordinates, and the requested preview videos.
- See [`THIRD_PARTY.md`](THIRD_PARTY.md) for external dependencies and their terms.
