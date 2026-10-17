# External projects and source attribution

- AlphaPose supplies 2D keypoint detection. Its source, detector weights and pose checkpoints are external: https://github.com/MVIG-SJTU/AlphaPose
- MotionBERT supplies the 3D backbone: https://github.com/Walter0807/MotionBERT. The adapter follows its HALPE conversion, normalization, flip augmentation and root treatment.
- SMPL models must be obtained from their official distribution under the applicable model terms: https://smpl.is.tue.mpg.de/
- SMPL-X's Python library supplies the SMPL forward implementation: https://github.com/vchoutas/smplx
- The pose and temporal shape network definitions and numerical conversion routines originate from this author's thesis project.

External source repositories and their licenses are not bundled or relicensed. Model data, training data, videos and checkpoints are excluded. This file records attribution; it does not grant rights to redistribute external assets or choose a license for the author's code.
