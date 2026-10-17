"""Stereo matching checks independent of ROS and scene object truth."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_stereo import match_candidates
from hand_competition.closed_loop.geometry import Camera


class StereoTests(unittest.TestCase):
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
