"""Check contact ordering and RGB relocation verification without ROS."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_grasp import make_plan, relocation


class GraspPlanTests(unittest.TestCase):
    def setUp(self):
        self.start = {'pose': [[0, .222, -.6082], [0, 0, 0]]}
        self.source = (.2275, .0825, -.5065)

    def test_noncontact_stages_never_close_or_enter_contact(self):
        for stage in ('orient', 'approach'):
            events, _ = make_plan(self.start, self.source, .018, stage)
            self.assertFalse(any(event.get('hand') == 'close' for event in events))
            self.assertNotIn('contact', [event['name'] for event in events])

    def test_release_precedes_withdrawal_and_target_tracks_observed_source(self):
        events, destination = make_plan(self.start, self.source, .018, 'grasp')
        names = [event['name'] for event in events]
        self.assertLess(names.index('contact'), names.index('closed'))
        self.assertLess(names.index('closed'), names.index('lifted'))
        self.assertLess(names.index('lowered'), names.index('released'))
        self.assertLess(names.index('released'), names.index('withdrawn'))
        self.assertAlmostEqual(destination[1], self.source[1] - .12)

    def test_unchanged_or_invisible_green_does_not_verify_relocation(self):
        _, destination = make_plan(self.start, self.source, .018, 'grasp')
        unchanged = {'issues': [], 'matches': [{'color': 'green', 'colored_surface_point_sdk_m': self.source}]}
        self.assertFalse(relocation(unchanged, destination, self.source)['relocation_verified'])
        self.assertFalse(relocation({'issues': [], 'matches': []}, destination, self.source)['relocation_verified'])
        moved = {'issues': [], 'matches': [{'color': 'green', 'colored_surface_point_sdk_m': destination}]}
        self.assertTrue(relocation(moved, destination, self.source)['relocation_verified'])


if __name__ == '__main__':
    unittest.main()
