"""Signed right-hand hinges and retained TF consistency, without mesh resources."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_bimanual_proxy import measured_hinge_angle


class RightHingeTests(unittest.TestCase):
    def test_negative_yaw_axis_and_origin_rotation_preserve_the_measured_angle(self):
        for axis, origin in [
            ([0.0, 0.0, -1.0], [0.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, -1.1529, -2.0944]),
        ]:
            zero = Rotation.from_euler("xyz", origin)
            for angle in [-0.08, 0.35, 1.1]:
                quaternion = (
                    zero * Rotation.from_rotvec(np.array(axis) * angle)
                ).as_quat()
                for sign in [-1, 1]:
                    actual, residual = measured_hinge_angle(
                        zero, quaternion * sign, axis
                    )
                    self.assertAlmostEqual(actual, angle, places=12)
                    self.assertLess(residual, 1e-12)

    def test_non_hinge_rotation_is_rejected_instead_of_becoming_an_angle(self):
        zero = Rotation.from_euler("xyz", [0.0, -1.1529, -2.0944])
        actual = zero * Rotation.from_rotvec([0.05, 0.2, 0.0])
        with self.assertRaisesRegex(ValueError, "declared hinge"):
            measured_hinge_angle(zero, actual.as_quat(), [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
