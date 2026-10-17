"""High empty-hand inspection must preserve height and reject failed IK."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_inspection import inspection_plan


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.start = {'joints': [0., .222, -.6082, 0., 0., 0., 0.],
                      'pose': [[0., .222, -.6082], [0.] * 3]}
        self.surface = {'colored_surface_point_sdk_m': [.22875, .08125, -.497625],
                        'half_pixel_perturbation_radius_m': .009}
        self.reference = {'completed': [{'name': 'contact', 'measured': {
            'joints': [.14875, .08125, -.457625, 0., 0., 0., 0.]}}]}

    @staticmethod
    def forward(joints):
        transform = np.eye(4)
        transform[:3, 3] = joints[:3]
        return transform

    @staticmethod
    def inverse(target, seed):
        return [*target[:3, 3], 0., 0., 0., 0.]

    def test_peeks_preserve_height_rotation_and_exclude_low_approach(self):
        route, high_index, _ = inspection_plan(self.start, self.surface, self.reference,
                                               self.inverse, self.forward, peek_y=True)
        self.assertFalse(any(e['name'] in ('contact', 'approached', 'tilted') for e in route))
        high = self.forward(route[high_index]['joints'])
        self.assertAlmostEqual(high[2, 3], self.surface['colored_surface_point_sdk_m'][2] + .25)
        for event in route[high_index:]:
            actual = self.forward(event['joints'])
            self.assertAlmostEqual(actual[2, 3], high[2, 3])
            np.testing.assert_allclose(actual[:3, :3], high[:3, :3])
        for name, offset in (('peek_plus04', .04), ('peek_minus04', -.04), ('peek_minus08', -.08)):
            event = next(e for e in route if e['name'] == name)
            self.assertAlmostEqual(event['joints'][1] - high[1, 3], offset)

    def test_failed_peek_is_rejected_before_a_route_is_returned(self):
        def inverse(target, seed):
            if abs(target[0, 3] - .14875) < 1e-6 and .12 < target[1, 3] < .13:
                return None
            return self.inverse(target, seed)
        with self.assertRaisesRegex(ValueError, 'peek_plus04: No seeded IK solution'):
            inspection_plan(self.start, self.surface, self.reference, inverse, self.forward, peek_y=True)


if __name__ == '__main__':
    unittest.main()
