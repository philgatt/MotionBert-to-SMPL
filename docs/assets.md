# Assets and configuration

The package intentionally separates source code from models and weights.

| Config key | Expected asset |
| --- | --- |
| `alphapose_repo` | Installed AlphaPose checkout, including detector weights |
| `alphapose_python` | AlphaPose environment's Python executable |
| `alphapose_config` | HALPE-26 configuration compatible with its checkpoint |
| `alphapose_checkpoint` | HALPE-26 pose weights |
| `motionbert_repo` | MotionBERT checkout containing `lib.utils.learning.load_backbone` |
| `motionbert_config` | Backbone YAML; original pipeline used global-lite, 243 frames |
| `motionbert_checkpoint` | Checkpoint containing `model_pos`; DataParallel prefixes are handled |
| `smpl_model` | Licensed neutral SMPL `.pkl` model, ten betas |
| `h36m_regressor` | Thesis-compatible `(17,6890)` H36M regressor in the documented joint order |
| `pose_checkpoint` | SMPLR-style checkpoint with `model_state_dict`, optional `args`, `x_mean`, `x_std` |
| `shape_checkpoint` | Res-TCN checkpoint with `model_state_dict`, architecture and training `args` |

Do not substitute a 17-joint regressor with a different joint order merely because its shape matches. The order is pelvis, right hip/knee/ankle, left hip/knee/ankle, spine, thorax, nose, head, left shoulder/elbow/wrist, right shoulder/elbow/wrist.

The pose network derives normalized bone directions internally, matching the current thesis inference script; no external feature standardization is applied. `pose_rx180` controls the input rotation used by the later thesis pose checkpoint. Earlier checkpoints may need `false`. The saved bone-template buffer is loaded from the checkpoint, but the independent pose branch does not use it; no training dataset is needed. Raw output matrices are retained for the original SMPL forward behavior, with an additional projected axis-angle export.

The temporal shape loader supports the architectures implemented in the existing checkpoint loader: simple, staged, paper-style, dilated paper-style, motion-feature dilated, and ST-GCN. The checkpoint's `clip_len` and `clip_coordinate_mode` are honored. `shape_root_center` and `shape_rx180` must match training; the thesis shape description uses raw, non-root-centered coordinates. Short sequences repeat their last frame for inference; overlapping estimates are averaged per frame. Outputs remain per-frame, as in the source implementation.

The supplied example points to assets under the new project's own `assets` and `third_party` directories. It is a template, not an automatic downloader. No account access or model-license acceptance is performed by this project.
