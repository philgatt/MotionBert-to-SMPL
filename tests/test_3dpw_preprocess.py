"""Regression tests for the 3DPW AlphaPose timeline contract.

The preprocessing stage deliberately retains every original video frame.  Its
missing-frame behaviour is part of the learned-model input contract, so test it
separately from the heavyweight AlphaPose and MotionBERT dependencies.
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_3dpw_alphapose.py"
SPEC = importlib.util.spec_from_file_location("preprocess_3dpw_alphapose", SCRIPT)
ALPHAPOSE_STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ALPHAPOSE_STAGE)


def entry(frame, value):
    return {
        "image_id": "image_{:05d}.jpg".format(frame),
        "idx": 7,
        "score": 0.9,
        "keypoints": [float(value)] * (26 * 3),
    }


class AlphaPoseTimelineTests(unittest.TestCase):
    def test_densify_preserves_video_timeline_and_internal_gaps_only(self):
        dense, missing = ALPHAPOSE_STAGE.densify_by_frame_index(
            [entry(1, 10), entry(3, 30)], expected_frames=5
        )

        np.testing.assert_array_equal(missing, [True, False, True, False, True])
        self.assertEqual([item["frame_idx"] for item in dense], list(range(5)))
        self.assertTrue(dense[0]["is_missing"])
        self.assertEqual(dense[0]["keypoints"], [0.0] * (26 * 3))

        interpolated = ALPHAPOSE_STAGE.interpolate_missing_keypoints_linear(dense, missing)
        np.testing.assert_array_equal(interpolated, [False, False, True, False, False])
        self.assertEqual(dense[2]["keypoints"], [20.0] * (26 * 3))
        self.assertTrue(dense[2]["is_interpolated"])
        # Leading and trailing gaps are intentionally retained as zero tracks.
        self.assertEqual(dense[0]["keypoints"], [0.0] * (26 * 3))
        self.assertEqual(dense[4]["keypoints"], [0.0] * (26 * 3))

    def test_frame_number_is_taken_from_image_id_not_detection_order(self):
        dense, missing = ALPHAPOSE_STAGE.densify_by_frame_index(
            [entry(4, 40), entry(1, 10)], expected_frames=6
        )

        np.testing.assert_array_equal(missing, [True, False, True, True, False, True])
        self.assertEqual(dense[1]["keypoints"][0], 10.0)
        self.assertEqual(dense[4]["keypoints"][0], 40.0)


if __name__ == "__main__":
    unittest.main()
