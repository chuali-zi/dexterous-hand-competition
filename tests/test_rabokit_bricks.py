"""Candidate checks for the overhead development camera."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'rabokit_bricks', Path(__file__).resolve().parents[1] / 'scripts/rabokit_bricks.py')
bricks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bricks)


class BrickCandidateTests(unittest.TestCase):
    def rotated_frame(self, angle, *, center=(50, 50), sides=(32, 32), hole=False):
        width = height = 100
        rgb = bytearray((180, 110, 60) * (width * height))
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for y in range(height):
            for x in range(width):
                u = (x - center[0]) * c + (y - center[1]) * s
                v = -(x - center[0]) * s + (y - center[1]) * c
                if abs(u) < sides[0] / 2 and abs(v) < sides[1] / 2:
                    if hole and u > 0 and v > 0:
                        continue
                    offset = (y * width + x) * 3
                    rgb[offset:offset + 3] = bytes((0, 255, 0))
        return width, height, rgb

    def test_wood_and_dark_lettering_are_not_primary_brick_colors(self):
        for rgb in ((190, 110, 55), (170, 90, 40), (20, 15, 65)):
            self.assertEqual(bricks.saturated_color(*rgb), (0, 0, 0))
        self.assertEqual(bricks.saturated_color(255, 0, 0), (255, 0, 0))

    def test_separated_squares_are_candidates_but_merged_shape_is_rejected(self):
        width, height = 90, 40
        rgb = bytearray((180, 110, 60) * (width * height))
        for left, right, color in ((5, 14, (255, 0, 0)), (20, 29, (0, 255, 0)),
                                   (40, 69, (0, 0, 255))):
            for y in range(5, 15):
                for x in range(left, right + 1):
                    offset = (y * width + x) * 3
                    rgb[offset:offset + 3] = bytes(color)
        result = bricks.detect(width, height, rgb)
        self.assertEqual([c['color'] for c in result['candidates']], ['red', 'green'])
        self.assertEqual(len(result['rejected_components']), 1)
        self.assertFalse(result['metric_calibration_available'])

    def test_rotated_squares_remain_candidates(self):
        for angle in (15, 30, 45, 75):
            with self.subTest(angle=angle):
                result = bricks.detect(*self.rotated_frame(angle))
                self.assertEqual(len(result['candidates']), 1)
                self.assertEqual(result['rejected_components'], [])
                candidate = result['candidates'][0]
                self.assertEqual(candidate['centroid'], (50, 50))
                if candidate['fill_ratio'] < .7:
                    self.assertGreaterEqual(candidate['rotated_square']['fill_ratio'], .9)

    def test_rotation_does_not_accept_clipping_missing_corner_or_elongated_merge(self):
        for options in ({'center': (5, 50)}, {'hole': True}, {'sides': (60, 20)}):
            with self.subTest(options=options):
                result = bricks.detect(*self.rotated_frame(30, **options))
                self.assertEqual(result['candidates'], [])
                self.assertEqual(len(result['rejected_components']), 1)


if __name__ == '__main__':
    unittest.main()
