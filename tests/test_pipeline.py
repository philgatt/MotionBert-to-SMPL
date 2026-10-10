import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from video_to_smpl.cli import parser, run
from video_to_smpl.config import load_config, check_assets
from video_to_smpl.keypoints import select_person, halpe_to_h36m, validate_coords
from video_to_smpl.methods.kinematics import _m2b_rodrigues_from_a_to_b
from video_to_smpl.methods.shape_inference import _make_windows, _predict_track


def detection(frame, x, idx=1):
    points = np.ones((26,3), dtype=float)
    points[:,0] = x
    return {"image_id": f"{frame}.jpg", "idx": idx, "keypoints": points.reshape(-1).tolist()}


class PipelineTests(unittest.TestCase):
    def test_checkpoint_paths_and_tensors_round_trip(self):
        from video_to_smpl.checkpoints import load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"args":{"path":Path("training/data")},"weight":torch.ones(2)},path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["args"]["path"], Path("training/data"))
            torch.testing.assert_close(loaded["weight"],torch.ones(2))

    def test_missing_frames_keep_timeline_and_zero_confidence(self):
        points, missing = select_person([detection(0,10), detection(2,30)], 4)
        np.testing.assert_array_equal(missing, [False,True,False,True])
        np.testing.assert_allclose(points[:,0,0], [10,20,30,30])
        self.assertEqual(points[1,0,2], 0)

    def test_explicit_track_does_not_switch_people(self):
        points, missing = select_person([detection(0,10,2), detection(1,50,3)], 2, 2)
        self.assertTrue(missing[1])
        self.assertEqual(points[1,0,0], 10)

    def test_reject_wrong_format_empty_and_bad_frame(self):
        with self.assertRaises(ValueError): select_person([], 2)
        with self.assertRaises(ValueError): select_person([detection(3,1)], 2)
        with self.assertRaises(ValueError): validate_coords(np.zeros((0,17,3)))
        with self.assertRaises(ValueError): validate_coords(np.full((2,17,3), np.nan))

    def test_joint_mapping(self):
        source = np.broadcast_to(np.arange(26)[None,:,None], (2,26,3)).astype(float)
        mapped = halpe_to_h36m(source)
        self.assertEqual(mapped.shape, (2,17,3))
        self.assertEqual(mapped[0,0,0], 19)
        self.assertEqual(mapped[0,7,0], 18.5)
        self.assertEqual(mapped[0,9,0], 0)

    def test_rodrigues_parallel_and_antiparallel_are_rotations(self):
        a = np.array([1.,0,0])
        for b in [a, -a, np.array([0.,1,0])]:
            matrix = _m2b_rodrigues_from_a_to_b(a,b)
            np.testing.assert_allclose(matrix @ a, b, atol=1e-7)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-7)
            self.assertAlmostEqual(np.linalg.det(matrix), 1)

    def test_temporal_windows_cover_tail_and_do_not_cross_gaps(self):
        coords = np.zeros((12,17,3), dtype=np.float32)
        indices = np.array([0,1,2,3,4,5,6,20,21,22,23,24])
        windows, covered = _make_windows(coords, indices, 4, 3)
        self.assertEqual(windows.shape[1:], (4,17,3))
        self.assertEqual(set(np.concatenate(covered)), set(range(12)))
        for clip in covered:
            self.assertTrue(np.all(np.diff(indices[clip]) == 1))

    def test_short_shape_clip_padding_does_not_duplicate_output(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.ones((len(x),10), device=x.device) * 2
        coords = np.zeros((3,17,3), dtype=np.float32)
        result = _predict_track(Constant(), coords, np.arange(3), 50, 25, 2, torch.device("cpu"))
        self.assertEqual(result.shape, (3,10))
        np.testing.assert_allclose(result, 2)

    def test_learned_raw_matrix_export_projects_reflections(self):
        from video_to_smpl.body import matrices_to_axis_angle
        from scipy.spatial.transform import Rotation
        matrices = np.tile(np.diag([1.,1.,-1.]), (2,24,1,1))
        pose = matrices_to_axis_angle(matrices)
        self.assertEqual(pose.shape, (2,72))
        proper = Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix()
        np.testing.assert_allclose(np.linalg.det(proper), 1, atol=1e-7)
        # Raw network output must remain untouched for original mesh reconstruction.
        self.assertEqual(np.linalg.det(matrices[0,0]), -1)

    def test_config_paths_relative_to_config_not_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.json"
            path.write_text(json.dumps({"smpl_model":"models/body.pkl"}))
            config = load_config(path)
            self.assertEqual(Path(config["smpl_model"]), Path(directory) / "models/body.pkl")
            self.assertTrue(check_assets(config,"kinematic","neutral",from_coords=True))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}")
            coords = root / "input.npy"
            np.save(coords, np.zeros((1,17,3)))
            output = root / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("untouched")
            args = parser().parse_args(["--config",str(config),"--coords",str(coords),"--output",str(output)])
            with patch("video_to_smpl.cli.check_assets", return_value=[]):
                with self.assertRaisesRegex(ValueError,"new or empty"):
                    run(args)
            self.assertEqual(sentinel.read_text(),"untouched")


if __name__ == "__main__":
    unittest.main()
