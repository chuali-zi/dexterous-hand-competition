"""Analytic box opening cases, independent of hand/contact model fitting."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pregrasp_geometry import bottom_gap_at_xy


class BottomOpeningTests(unittest.TestCase):
    def test_flat_cube_has_no_opening_and_outside_is_unknown(self):
        self.assertEqual(
            bottom_gap_at_xy([0, 0, 0.025], [1, 0, 0, 0], 0, [[0, 0], [0.026, 0]]),
            [0.0, None],
        )

    def test_edge_pivot_opening_shrinks_with_insertion_depth(self):
        angle = np.deg2rad(20)
        rotation = Rotation.from_euler("y", -angle)
        pivot = np.array([-0.025, 0, 0])
        center = pivot + rotation.apply([0.025, 0, 0.025])
        # Original bottom points at free edge, 10 mm inward and the pivot.
        points = pivot + rotation.apply([[0.05, 0, 0], [0.04, 0, 0], [0, 0, 0]])
        gaps = bottom_gap_at_xy(
            center, rotation.as_quat()[[3, 0, 1, 2]], 0, points[:, :2]
        )
        np.testing.assert_allclose(
            gaps, np.sin(angle) * np.array([0.05, 0.04, 0]), atol=1e-12
        )
        self.assertLess(gaps[1], 0.015)
        self.assertGreater(gaps[0], 0.015)

    def test_world_yaw_and_translation_preserve_gap(self):
        rotation = Rotation.from_euler("z", 37, degrees=True) * Rotation.from_euler(
            "y", -20, degrees=True
        )
        offset = np.array([0.23, 0.08, -0.56])
        center = offset + rotation.apply([0.025, 0, 0.025])
        point = offset + rotation.apply([0.04, 0, 0])
        result = bottom_gap_at_xy(
            center, rotation.as_quat()[[3, 0, 1, 2]], offset[2], [point[:2]]
        )
        self.assertAlmostEqual(result[0], 0.04 * np.sin(np.deg2rad(20)))

    def test_vertical_face_and_invalid_quaternion_are_not_clearance(self):
        q = Rotation.from_euler("y", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
        self.assertEqual(bottom_gap_at_xy([0, 0, 0.025], q, 0, [[0, 0]]), [None])
        with self.assertRaises(ValueError):
            bottom_gap_at_xy([0, 0, 0.025], [2, 0, 0, 0], 0, [[0, 0]])


if __name__ == "__main__":
    unittest.main()
