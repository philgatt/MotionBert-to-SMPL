# Why the reproduced metrics differ from the paper

The directly comparable run uses the stored 3DPW MotionBERT coordinates from `outputs_main_gtbbox_frame_mode` and the VIBE evaluation protocol. It obtains 106.31 mm MPJPE, 65.82 mm PA-MPJPE, and 123.42 mm PVE, compared with the paper's 95.54, 62.11, and 111.27 mm.

The following differences were verified.

1. The historical training dataset was built with camera-space (`incam`) root orientation, whereas the VIBE database pose array has global root orientation. Only the first axis-angle triplet differs, but its mean absolute difference is 1.63 radians. Training the camera-space target directly and evaluating it against VIBE global joints gives 793.36 mm MPJPE, despite a similar 69.17 mm PA-MPJPE. The paper result therefore needs an explicit camera-to-evaluation-frame conversion if it used `incam` targets.
2. The historical training input dataset does not match the current `outputs_main_gtbbox_frame_mode` MotionBERT cache exactly. After root-centering and converting to the network's unit bone directions, corresponding train frames differ by 3.98 degrees on average; the 90th and 99th percentiles are 7.54 and 30.07 degrees. This is a material distribution shift because these directions are the complete network input.
3. The historical available SMPLR checkpoints implement ELU activations and include a shape branch. The requested Figure 3.3 reproduction implements ReLU activations and no shape branch, with neutral beta throughout. Thus those archived checkpoints cannot be treated as the same model.
4. The old available training metadata validates on 957 samples, while the reproduced protocol uses the stated 10,261-frame VIBE validation split. This changes checkpoint selection.

Per-track reprojection scaling also differs in historical datasets, but it has little direct effect here because the pose network normalizes every bone direction. The primary remaining requirement for exact paper reproduction is the original MotionBERT cache plus its camera/frame conversion and the original checkpoint-selection dataset.
