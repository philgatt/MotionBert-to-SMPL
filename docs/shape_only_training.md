# Shape-only Res-TCN recreation

The included checkpoint `models/shape_only_res_tcn_motionbert_rx180_20261010.pt` recreates the shape-only network in Figure 3.4.

- Input: raw, non-root-centred MotionBERT H36M-17 coordinates.
- Temporal input: 50 consecutive frames; 180-degree rotation around the x axis.
- Network: 51-to-128 convolution with kernel size 5; two 128-channel residual units; 128-to-192 transition; two 192-channel units; 192-to-256 transition; two 256-channel units; global average pooling; `256 → 128 → 10` regression head with ReLU and no head dropout.
- Training: VIBE 3DPW train/validation data (22,663 / 10,261 matched frames), 50-frame clips with stride 25, AdamW (learning rate 0.001, weight decay 0.0001), L1 beta loss, batch size 64, 80 epochs.
- Best validation beta L1: 0.250745 (epoch 50).

The VIBE/3DPW test evaluation fixes the pose to zero on both the predicted and target SMPL meshes and reports PVE only. Predictions are averaged across overlapping one-frame-stride clips, then their median is taken per person track. The recreated checkpoint obtains 26.62 mm frame-weighted PVE across 34,561 matched test frames; the thesis result was 26.84 mm.

Run the recreation with `scripts/train_shape_only_res_tcn.py`; reproduce the metric with `scripts/evaluate_shape_only_res_tcn_vibe.py`. `--disable-cudnn` is included because this Windows CUDA setup requires PyTorch's CUDA kernels for stable temporal-convolution training.
