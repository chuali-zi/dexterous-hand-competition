"""Stereo matching checks independent of ROS and scene object truth."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_stereo import match_candidates, reconstruct
from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT
from hand_competition.closed_loop.geometry import Camera


class StereoTests(unittest.TestCase):
    def test_sensor_numeric_intrinsics_are_normalized_before_geometry(self):
        class SensorFloat(float):
            pass

        pairs, metadata = {}, {}
        frames = {LEFT_ROOT: {'parent': 'world', 'xyz': [0, .1, 1], 'xyzw': [0, 0, 0, 1]},
                  RIGHT_ROOT: {'parent': 'world', 'xyz': [0, -.1, 1], 'xyzw': [0, 0, 0, 1]}}
        for name, left, origin in (('first', 96, [.5, 0, 2]), ('second', 196, [0, 0, 2])):
            rgb = bytearray(320 * 240 * 3)
            for y in range(96, 106):
                for x in range(left, left + 10):
                    rgb[(y * 320 + x) * 3:(y * 320 + x) * 3 + 3] = b'\xff\0\0'
            image = SimpleNamespace(width=320, height=240, encoding='rgb8', step=960, data=rgb)
            pairs[name] = (image, None)
            metadata[name] = {'frame_id': name, 'width': 320, 'height': 240,
                              'k': [SensorFloat(v) for v in (400, 0, 160, 0, 400, 120, 0, 0, 1)]}
            frames[name] = {'parent': 'world', 'xyz': origin, 'xyzw': [1, 0, 0, 0]}
        report = reconstruct(pairs, metadata, frames)
        self.assertFalse(report['issues'])
        self.assertEqual(len(report['matches']), 1)

    def cameras(self):
        rotation = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
        return [Camera(name, (x, 0, 2), rotation, 400, 400, 160, 120, 320, 240, 'test')
                for name, x in (('first', .5), ('second', 0))]

    def test_known_scalar_projection_recovers_sdk_surface_point(self):
        # World point (.2,.1,0); pinhole equations with z depth 2 m.
        first = {'color': 'green', 'centroid': (160 + 400 * (.2 - .5) / 2, 120 - 400 * .1 / 2)}
        second = {'color': 'green', 'centroid': (160 + 400 * .2 / 2, 120 - 400 * .1 / 2)}
        matches, issues = match_candidates(self.cameras(), [[first], [second]], (.1, 0, 1),
                                            ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self.assertFalse(issues)
        for actual, expected in zip(matches[0]['colored_surface_point_sdk_m'], (.1, .1, -1)):
            self.assertAlmostEqual(actual, expected)
        self.assertGreater(matches[0]['half_pixel_perturbation_radius_m'], 0)

    def test_duplicate_same_color_correspondence_is_not_selected(self):
        first = {'color': 'red', 'centroid': (100, 100)}
        second = {'color': 'red', 'centroid': (200, 100)}
        matches, issues = match_candidates(self.cameras(), [[first], [second, second]], (0, 0, 0),
                                            ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self.assertFalse(matches)
        self.assertTrue(issues)


if __name__ == '__main__':
    unittest.main()
