import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_3dpw_training.py"
SPEC = importlib.util.spec_from_file_location("run_3dpw_training", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TrainingRunnerTests(unittest.TestCase):
    def test_prediction_reuse_requires_every_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in RUNNER.SPLITS:
                alpha = root / "predictions" / "alphapose" / split / "sequence" / "person_1"
                motionbert = root / "predictions" / "motionbert" / split / "sequence" / "person_1"
                alpha.mkdir(parents=True)
                motionbert.mkdir(parents=True)
                (alpha / "keypoints_2d_prediction.json").write_text("[]")
                (motionbert / "X3D_prediction.npy").write_bytes(b"placeholder")

            self.assertTrue(RUNNER._has_alphapose_outputs(root))
            self.assertTrue(RUNNER._has_motionbert_outputs(root))

            (root / "predictions" / "motionbert" / "test" / "sequence" / "person_1" / "X3D_prediction.npy").unlink()
            self.assertFalse(RUNNER._has_motionbert_outputs(root))


if __name__ == "__main__":
    unittest.main()
