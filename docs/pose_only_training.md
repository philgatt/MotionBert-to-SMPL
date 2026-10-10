# Pose-only SMPLR retraining

`scripts/train_pose_only_smplr.py` implements the architecture in Figure 3.3 exactly:

- Input: 17 H36M/MotionBERT joints, root centred inside the network.
- Features: 16 normalized parent-to-child bone directions, flattened to 48 values.
- Trunk: `48 → 1024 → 1024 → 512 → 512 → 512`, with `FC → BatchNorm → ReLU` at every hidden block.
- Heads: one `512 → 256 → 256 → 24×9` head and one `512 → 256 → 24×9` head; their concatenation is mapped by `FC(2×24×9 → 24×9)` and `tanh`.
- Output: 24 raw 3×3 SMPL pose matrices. There is no shape branch.

The loss is `L_R + L_SMPL`: L1 on the raw pose matrices, plus L1 on 23 SMPL joints and 18 SMPL landmarks. Both predicted and ground-truth SMPL forwards use `beta = 0`; mesh landmarks only appear in the loss, never in the network input.

## Completed run

The complete run is in `training-runs/pose_only_smplr_vibe_relu_bn_20261008/` and is Git-ignored. Its selected 10.8 MB checkpoint is copied to `models/pose_only_smplr_vibe_relu_bn_20261008.pt` so the standalone project can be used after cloning. Its `metrics.json` records the complete run:

| Setting | Value |
| --- | --- |
| Split | VIBE/3DPW: 22,663 train, 10,261 validation samples |
| Optimizer | Adam |
| Epochs | 40 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Best validation loss | 0.107679 (epoch 39) |

The validation loss drops quickly in the first eleven epochs (0.161460 to 0.113147) and later changes only modestly, while training loss keeps decreasing (0.049040 by epoch 40). This is the expected overfitting pattern described in the procedure.

## Controlled test-set result

On all 34,561 VIBE/3DPW test frames, the bundled checkpoint achieved **55.29 mm MPJPE**, **34.12 mm PA-MPJPE**, and **79.37 mm root-aligned PVE**. The machine-readable result is in `docs/pose_only_vibe_test_metrics.json` and can be reproduced by `scripts/evaluate_pose_only_smplr_vibe_test.py`.

This is a controlled pose-regression measurement. The test database has ground-truth SMPL labels but no stored MotionBERT predictions per frame, so it generates the 17-joint inputs using neutral SMPL and the H36M regressor. It measures the learned conversion model but must not be presented as an end-to-end video → AlphaPose → MotionBERT result.

The VIBE databases provide SMPL pose labels but do not contain saved MotionBERT predictions for every training sample. To make the exact stated split reproducible, this run obtains the 17-joint coordinates by forwarding each VIBE pose through neutral SMPL and the MotionBERT H36M-17 regressor. The model itself receives the same root-centred 17-joint representation used at inference from MotionBERT. Training directly on stored MotionBERT predictions would require a matched cache for every VIBE frame; no such cache was present.

## Reproduce

Run it with the compatible MotionBERT environment:

```powershell
E:\miniconda-envs\motionbert\python.exe scripts\train_pose_only_smplr.py `
  --vibe-root F:\Bachelor\Vibe\VIBE\data\vibe_db `
  --motionbert-repo E:\Users\Philipp\Dokumente\Pose_Estimation_3D\MotionBert\MotionBERT `
  --smpl-data-root E:\Users\Philipp\Dokumente\Pose_Estimation_3D\MotionBert\MotionBERT\data\mesh
```

The ELU MotionBERT-coordinate retraining is the bundled default: `models/pose_only_smplr_motionbert_global_elu_bn_20261009.pt`. It achieved **103.44 mm MPJPE**, **64.75 mm PA-MPJPE**, and **121.03 mm PVE** on the 34,561-frame 3DPW/VIBE test set. See `docs/pose_only_motionbert_elu_vibe_metrics.json`. `configs/example.json` selects this checkpoint and disables the legacy `rx180` conversion, which does not apply to it. The pipeline selects its ELU activation from checkpoint metadata automatically.
