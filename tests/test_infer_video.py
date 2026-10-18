import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "infer_video.py"
SPEC = importlib.util.spec_from_file_location("infer_video", SCRIPT)
INFER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFER)


class InferenceCommandTests(unittest.TestCase):
    def _args(self, parameters):
        return Namespace(
            video=Path("input.mp4"), parameters=parameters, config=Path("local.json"),
            output=Path("output"), device="cpu", track_id=None, visualize=["mesh"],
        )

    def test_parameter_modes_select_the_expected_learned_models(self):
        pose = INFER.pipeline_args(self._args("pose"))
        shape = INFER.pipeline_args(self._args("shape"))
        both = INFER.pipeline_args(self._args("both"))
        self.assertEqual((pose.pose, pose.shape), ("learned", "neutral"))
        self.assertEqual((shape.pose, shape.shape), ("kinematic", "temporal"))
        self.assertEqual((both.pose, both.shape), ("learned", "temporal"))


if __name__ == "__main__":
    unittest.main()
