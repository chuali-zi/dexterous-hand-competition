"""Candidate checks for the overhead development camera."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'rabokit_bricks', Path(__file__).resolve().parents[1] / 'scripts/rabokit_bricks.py')
bricks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bricks)


class BrickCandidateTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
