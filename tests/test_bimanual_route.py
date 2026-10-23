from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_bimanual_route import check_side, joint_bounds


class BimanualRouteTests(unittest.TestCase):
    def test_intersection_insets_both_limits_without_expanding_either(self):
        preset = [[-1, 1]] * 7
        scene = [[-0.5, 2]] * 7
        np.testing.assert_allclose(
            joint_bounds(preset, scene, 0.02), [[-0.48, 0.98]] * 7
        )
        with self.assertRaises(ValueError):
            joint_bounds(preset, scene, 1)
        with self.assertRaises(ValueError):
            joint_bounds(preset, scene, float("nan"))

    def test_failed_initial_pose_is_not_a_continuous_route(self):
        rows = [dict(seconds=t, hand_pose=np.eye(4)) for t in (0, 1)]
        with patch("check_bimanual_route.centered_inverse", return_value=None):
            result = check_side(
                rows, np.eye(4), [0] * 7, np.array([[-1, 1]] * 7), lambda q: np.eye(4)
            )
        self.assertFalse(result["continuous"])
        self.assertEqual(result["segments"], [])


if __name__ == "__main__":
    unittest.main()
