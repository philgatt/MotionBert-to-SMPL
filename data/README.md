# Local datasets

Place datasets here after obtaining them from their respective licensors. This directory is part of the repository only as a layout guide: all dataset files are ignored by Git.

## 3DPW

Download the 3DPW dataset and its VIBE-compatible preprocessing metadata, then use this layout:

```text
data/
  3DPW/
    imageFiles/imageFiles/<sequence>/image_00000.jpg
    sequenceFiles/sequenceFiles/<split>/<sequence>.pkl
    preprocessed_train/3dpw_train.npz
    preprocessed_val/3dpw_validation.npz
    preprocessed_test/3dpw_test.npz
```

The `preprocessed_*` files provide the VIBE-style tracks and ground-truth person boxes used by the AlphaPose stage. They are required for reproducing the complete 3DPW preprocessing pipeline; the native image archive alone is insufficient.

## VIBE split databases

For training and evaluation, obtain the VIBE 3DPW split databases and place them here:

```text
data/
  vibe_db/
    3dpw_train_db.npz
    3dpw_val_db.npz
    3dpw_test_db.npz
```

Then use `--dataset-root data/3DPW` and `--vibe-root data/vibe_db` with `scripts/run_3dpw_training.py`.