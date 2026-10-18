"""Check contact ordering and RGB relocation verification without ROS."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_grasp import make_plan, pickup_offset, relocation
from rabokit_pickup_vision import neighboring_layout, pickup_evidence


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

    def test_open_approach_enters_from_outside_and_retraces_saved_targets(self):
        events, _ = make_plan(self.start, self.source, .009, 'touch', .04)
        self.assertFalse(any(event.get('hand') == 'close' for event in events))
        contact = next(event for event in events if event['name'] == 'contact')
        self.assertAlmostEqual(contact['position'][0], self.source[0] - .08)
        self.assertAlmostEqual(contact['position'][1], self.source[1])
        seen = set()
        for event in events:
            if 'reverse_of' in event:
                self.assertIn(event['reverse_of'], seen)
            seen.add(event['name'])
        returning = events[next(i for i, event in enumerate(events) if event['name'] == 'contact') + 1:]
        self.assertTrue(all('reverse_of' in event for event in returning))

    def test_transfer_requires_visual_pickup_confirmation(self):
        events, _ = make_plan(self.start, self.source, .009, 'grasp')
        transfer = next(event for event in events if event['name'] == 'transferred')
        self.assertTrue(transfer['requires_verified_pickup'])


class PickupVisionTests(unittest.TestCase):
    def test_positive_tabletop_evidence_allows_high_open_even_after_target_was_pushed(self):
        source = [0., 0., 0.]
        tools = [[-.08, -.08, .20], [-.08, -.08, .25]]
        table = {'valid': True, 'point': [-.062, .037, .0023]}
        proof = pickup_evidence(source, table, table, *tools, [.08, 0.], .035)
        self.assertFalse(proof['physical_grasp_verified'])
        self.assertTrue(proof['target_stationary_on_tabletop'])

    def test_missing_moving_or_elevated_target_cannot_trigger_high_open(self):
        tools = [[-.08, -.08, .20], [-.08, -.08, .25]]
        table = {'valid': True, 'point': [0., 0., 0.]}
        cases = [(table, {'valid': False}),
                 (table, {'valid': True, 'point': [.01, 0., 0.]}),
                 ({'valid': True, 'point': [0., 0., .16]},
                  {'valid': True, 'point': [0., 0., .16]}),
                 (table, {'valid': True, 'point': [0., 0., .025]}),
                 (table, {'valid': True, 'point': [float('nan'), 0., 0.]})]
        for first, second in cases:
            with self.subTest(first=first, second=second):
                proof = pickup_evidence([0., 0., 0.], first, second, *tools, [.08, 0.], .035)
                self.assertFalse(proof['target_stationary_on_tabletop'])

    def test_tool_shift_is_removed_from_expected_surface_offset(self):
        import math
        source = [0., 0., 0.]
        tools = [[-.085, -.07, .20], [-.085, -.07, .25]]
        held = [{'valid': True, 'point': [0., -.08, height]} for height in (.165, .215)]
        proof = pickup_evidence(source, *held, *tools,
                                pickup_offset(-math.pi / 2, .01, tool_offset_x=-.005), .035)
        self.assertTrue(proof['physical_grasp_verified'])
        self.assertAlmostEqual(proof['attachment_error_m'], 0.)
        table = {'valid': True, 'point': source}
        self.assertFalse(pickup_evidence(source, table, table, *tools,
                         pickup_offset(-math.pi / 2, .01, tool_offset_x=-.005), .035)['physical_grasp_verified'])

    def test_peek_rejects_a_visible_brick_still_at_the_source(self):
        source = [0., 0., 0.]
        tools = [[-.08, -.08, .20], [-.08, -.08, .25]]
        table = {'valid': True, 'point': source}
        proof = pickup_evidence(source, table, table, *tools, [.08, 0.], .04)
        self.assertFalse(proof['physical_grasp_verified'])
        self.assertEqual(proof['minimum_elevation_m'], 0.)
        self.assertGreater(proof['following_error_m'], .04)
        held = [{'valid': True, 'point': [0., -.08, z]} for z in (.16, .21)]
        self.assertTrue(pickup_evidence(source, *held, *tools, [.08, 0.], .04)['physical_grasp_verified'])

    def test_only_elevated_following_target_verifies_pickup(self):
        source = [0, 0, 0]
        tools = [[-.08, 0, .2], [-.08, 0, .25]]
        held = [{'valid': True, 'point': [0, 0, z]} for z in (.17, .22)]
        verify = lambda a, b: pickup_evidence(source, a, b, *tools, [.08, 0], .03)['physical_grasp_verified']
        self.assertTrue(verify(*held))
        table = {'valid': True, 'point': source}
        self.assertFalse(verify(table, table))
        self.assertFalse(verify(held[0], held[0]))
        self.assertFalse(verify(held[0], {'valid': False}))

    def test_neighbor_matching_accepts_order_change_but_rejects_movement_or_missing(self):
        def block(x):
            return {'color': 'red', 'pixels': [[x, 10], [x + 100, 10]]}
        before = {'issues': [], 'matches': [block(10), block(30)]}
        self.assertTrue(neighboring_layout(before, {'issues': [], 'matches': [block(30), block(10)]})['consistent'])
        self.assertFalse(neighboring_layout(before, {'issues': [], 'matches': [block(30), block(13)]})['consistent'])
        self.assertFalse(neighboring_layout(before, {'issues': [], 'matches': [block(10)]})['consistent'])


if __name__ == '__main__':
    unittest.main()
