"""Test projection and visible-pixel contradiction scores without robot or MuJoCo."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_hand_dynamics import cube_pixels, projection_constraints


class ProjectionTests(unittest.TestCase):
    def test_cube_projection_uses_depth_and_rejects_behind_camera(self):
        row = {'local_target_position': [0., 0., 2.], 'local_target_quaternion_wxyz': [1., 0., 0., 0.]}
        sdk = np.eye(3), np.zeros(3)
        camera = np.eye(3), np.zeros(3), np.diag([100., 100., 1.])
        uv = cube_pixels(row, sdk, camera)
        np.testing.assert_allclose(uv.min(axis=0), [-2.5 / 1.975] * 2)
        np.testing.assert_allclose(uv.max(axis=0), [2.5 / 1.975] * 2)
        row['local_target_position'][2] = 0.
        with self.assertRaises(ValueError):
            cube_pixels(row, sdk, camera)

    def test_constraints_distinguish_inside_green_and_inside_wood(self):
        uv = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
        green = np.array([[3., 3.], [4., 4.], [5., 5.]])
        wood = np.array([[12., 3.], [13., 4.], [14., 5.]])
        good = projection_constraints(uv, (green, wood))
        self.assertEqual(good['squared_cost'], 0.)
        bad = projection_constraints(uv + [10., 0.], (green, wood))
        self.assertGreater(bad['maximum_green_miss_px'], 0.)
        self.assertGreater(bad['maximum_wood_intrusion_px'], 0.)

    def test_hidden_green_is_unavailable_instead_of_zero_error(self):
        result = projection_constraints(np.zeros((8, 2)), (np.empty((0, 2)), np.ones((3, 2))))
        self.assertFalse(result['available'])
        self.assertNotIn('squared_cost', result)


if __name__ == '__main__':
    unittest.main()
