# Validation of the extracted project

Checked locally on 2026-10-08. This is a functional smoke test, not an accuracy benchmark.

- A four-frame real video completed AlphaPose → MotionBERT → kinematic pose → neutral shape → SMPL export. AlphaPose ran in its existing separate environment; the new package ran on CPU.
- Kinematic, optimization and learned pose each completed using four real saved MotionBERT frames and the existing neutral SMPL model. Optimization used five iterations for the smoke test, rather than the normal 200. Learned pose used an available earlier compatible trained checkpoint, not the later checkpoint whose configured path was missing.
- Temporal shape checkpoint loading, short-clip padding, inference and final mesh integration completed using a deterministic **untrained** staged network checkpoint. No trained temporal shape checkpoint was found in the inspected source project. Shape accuracy and the final intended checkpoint's preprocessing remain unverified.
- 2D overlay, 3D skeleton and shaded SMPL preview MP4s were rendered and decoded successfully, with four frames each. The mesh preview was visually inspected.
- Exported matrices, betas and translation reconstructed the saved meshes within `2e-6` absolute tolerance. This check included the learned model's raw, non-orthogonal matrix representation.
- Automated tests cover missing detections, track selection, coordinate validation, joint mapping, degenerate rotation cases, temporal coverage, matrix projection, portable checkpoint loading, config paths and overwrite protection.
- All 11 automated tests passed. A Python wheel built successfully with the configured setuptools backend; no model assets are included in it.
- Original extracted source-file hashes were checked again after testing. Only the separate new project folder was written.

The existing Windows AlphaPose environment needed its own DLL directories in the subprocess path; the launcher now handles this. The local sandbox also required permission to read/load those DLLs for the full test. No original environment files were edited.

Large/private inputs, checkpoints, local configuration and smoke-test outputs are excluded from Git and the source archive. The portable configuration still needs the user's own asset paths. Full-length video performance, CUDA execution, other operating systems, tracking through crossings and quantitative pose/shape accuracy have not been validated here.
