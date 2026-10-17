"""Independent analytic checks for robot/camera frame composition."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'calibration', Path(__file__).resolve().parents[1] / 'scripts/rabokit_calibration_analysis.py')
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class CalibrationGeometryTests(unittest.TestCase):
    def test_rotated_parent_changes_child_translation(self):
        frames = {'parent': {'parent': 'world', 'xyz': [2, 3, 4],
                              'xyzw': [0, 0, math.sqrt(.5), math.sqrt(.5)]},
                  'child': {'parent': 'parent', 'xyz': [1, 0, 0], 'xyzw': [0, 0, 0, 1]}}
        _, point = calibration.world_transform(frames, 'child')
        for actual, expected in zip(point, (2, 4, 4)):
            self.assertAlmostEqual(actual, expected)

    def test_downward_optical_camera_projection_and_sign(self):
        camera = (((1, 0, 0), (0, -1, 0), (0, 0, -1)), (0.5, 0, 2))
        k = [400, 0, 160, 0, 400, 120, 0, 0, 1]
        self.assertEqual(calibration.project(camera, k, (0.7, .1, 0)), (200, 100))
        with self.assertRaises(ValueError):
            calibration.project(camera, k, (0, 0, 3))

    def test_rejects_zero_quaternion_and_cycle(self):
        with self.assertRaises(ValueError):
            calibration.quaternion_rotation((0, 0, 0, 0))
        with self.assertRaises(ValueError):
            calibration.world_transform({'a': {'parent': 'a', 'xyz': [0, 0, 0],
                                              'xyzw': [0, 0, 0, 1]}}, 'a')


if __name__ == '__main__':
    unittest.main()
