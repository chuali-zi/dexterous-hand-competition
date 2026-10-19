"""Check contact ordering and RGB relocation verification without ROS."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_grasp import high_pass_through_plan, make_plan, pickup_offset, preshape_fingers_plan, relocation
from rabokit_pickup_vision import neighboring_layout, pickup_evidence


class GraspPlanTests(unittest.TestCase):
    def test_pad_mode_rejects_geometry_overrides_before_observation(self):
        import contextlib
        import io
        from rabokit_grasp import main
        argv = ['rabokit_grasp.py', '--output', 'unused-pad-output', '--stage', 'pickup',
                '--seed-record', 'unused-seed.json', '--keep-pinky-open', '--preshape-fingers',
                '--limited-pinch', '--inspect-pickup', '--prepare-thumb-high', '--pad-grasp',
                '--tool-offset-x', '-.03']
        with patch.dict('os.environ', {'ROS2_NAMESPACE': '/test'}), patch('sys.argv', argv), \
                patch('rabokit_grasp.observe') as observe, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()
        observe.assert_not_called()

    def test_pad_plan_keeps_arm_route_and_uses_profile_goals_for_both_comparisons(self):
        from rabokit_hand import targets
        events, _ = make_plan(self.start, self.source, .009, 'pickup')
        staged = preshape_fingers_plan(events, limited_pinch=True, prepare_thumb_high=True,
                                      compare_empty_closure=True, pad_grasp=True)
        self.assertEqual([e for e in staged if 'hand' not in e], [e for e in events if 'hand' not in e])
        hands = {e['name']: e for e in staged if 'hand' in e}
        self.assertTrue(all(e['pad_grasp'] and 'thumb_yaw_rad' not in e for e in hands.values()))
        for high, low in (('empty_finger_ready', 'thumb_ready'), ('empty_thumb_opposed', 'thumb_opposed'),
                          ('empty_closed', 'closed')):
            self.assertEqual(targets(hands[high]['hand'], pad_grasp=True),
                             targets(hands[low]['hand'], pad_grasp=True))
        with self.assertRaises(ValueError):
            preshape_fingers_plan(events, limited_pinch=True, pad_grasp=True)

    def test_empty_closure_comparison_restores_high_shape_before_any_descent(self):
        events, _ = make_plan(self.start, self.source, .009, 'pickup')
        old = preshape_fingers_plan(events, limited_pinch=True, prepare_thumb_high=True)
        new = preshape_fingers_plan(events, limited_pinch=True, prepare_thumb_high=True,
                                   compare_empty_closure=True)
        names = [e['name'] for e in new]
        i = names.index('preshaped')
        self.assertEqual(names[i:i+6], ['preshaped', 'empty_finger_ready', 'empty_thumb_opposed',
                                      'empty_closed', 'empty_reprepared', 'approached'])
        self.assertEqual([e for e in new if e['name'] not in (
            'empty_finger_ready', 'empty_thumb_opposed', 'empty_closed', 'empty_reprepared')], old)
        finger_ready, opposed, baseline, restored = new[i+1:i+5]
        self.assertEqual(finger_ready['hand'], 'pinch_finger_ready_without_pinky')
        self.assertTrue(finger_ready['wait_arrival'])
        self.assertNotIn('thumb_yaw_rad', finger_ready)
        self.assertEqual(baseline['hand'], 'pinch_without_pinky')
        closed = next(e for e in new if e['name'] == 'closed')
        self.assertEqual(baseline['thumb_yaw_rad'], 1.15)
        self.assertEqual(baseline['thumb_yaw_rad'], closed['thumb_yaw_rad'])
        self.assertEqual(opposed['hand'], 'pinch_high_thumb_opposed_without_pinky')
        low_opposed = next(e for e in new if e['name'] == 'thumb_opposed')
        self.assertEqual(opposed['thumb_yaw_rad'], low_opposed['thumb_yaw_rad'])
        self.assertTrue(opposed['wait_arrival'])
        self.assertFalse(low_opposed.get('wait_arrival', False))
        self.assertNotIn('thumb_yaw_rad', restored)
        self.assertEqual(baseline['preshape_feedback_key'], 'empty_closed_feedback')
        self.assertEqual(restored['hand'], new[i]['hand'])
        self.assertTrue(restored['wait_arrival'])
        self.assertTrue(all(e['hand_only'] and e['hand_ramp_seconds'] == 2. for e in (finger_ready, opposed, baseline, restored)))
        with self.assertRaises(ValueError):
            preshape_fingers_plan(events, limited_pinch=True, compare_empty_closure=True)

    def test_high_thumb_preparation_changes_only_the_high_hand_event(self):
        events, _ = make_plan(self.start, self.source, .009, 'pickup')
        old = preshape_fingers_plan(events, limited_pinch=True)
        new = preshape_fingers_plan(events, limited_pinch=True, prepare_thumb_high=True)
        changed = [(a, b) for a, b in zip(old, new) if a != b]
        self.assertEqual(len(changed), 1)
        before, after = changed[0]
        self.assertEqual(after['name'], 'preshaped')
        self.assertEqual(after['hand'], 'pinch_high_thumb_ready_without_pinky')
        self.assertEqual({**before, 'hand': after['hand']}, after)
        self.assertTrue(after['wait_arrival'])
        self.assertEqual(new[new.index(after)-1]['name'], 'aligned')
        self.assertGreater(new[new.index(after)-1]['position'][2]-self.source[2], .25)
        with self.assertRaises(ValueError):
            preshape_fingers_plan(events, prepare_thumb_high=True)

    def test_pass_through_only_marks_high_uncaptured_outbound_joint_points(self):
        def point(name, z, capture=False):
            return {'name': name, 'joints': [0.] * 7, 'position': [0., 0., z], 'capture': capture}
        events = [{'name': 'opened', 'hand': 'open', 'capture': True}, point('high1', .3),
                  point('snapshot', .3, True), point('low', .1), point('rising', .3), point('high2', .3),
                  point('before_hand', .3),
                  {'name': 'preshaped', 'hand': 'pinch_preshape_without_pinky', 'capture': True},
                  point('after_hand', .3)]
        original = [dict(e) for e in events]
        result = high_pass_through_plan(events, {'pose': [[0., 0., .3], [0.] * 3]}, 0.)
        self.assertEqual([e['name'] for e in result if e.get('pass_through')], ['high1', 'high2'])
        self.assertEqual(events, original)
        self.assertEqual([{k: v for k, v in e.items() if k != 'pass_through'} for e in result], original)

    def test_limited_pinch_uses_high_yaw_shape_and_symmetric_release_without_changing_arm_route(self):
        events, _ = make_plan(self.start, self.source, .009, 'pickup')
        release_index = next(i for i, e in enumerate(events) if e['name'] == 'source_release_position')
        events.insert(release_index, {'name': 'empty_pickup_open', 'hand': 'open', 'capture': False,
                                     'requires_tabletop_target': True, 'wait_open': True})
        original = [dict(e) for e in events]
        staged = preshape_fingers_plan(events, limited_pinch=True)
        hands = {e['name']: e for e in staged if 'hand' in e}
        self.assertEqual(hands['opened']['hand'], 'open')
        self.assertEqual(hands['preshaped']['hand'], 'pinch_preshape_without_pinky')
        self.assertTrue(hands['preshaped']['wait_arrival'])
        self.assertEqual(hands['thumb_ready']['hand'], 'pinch_contact_ready_without_pinky')
        self.assertFalse(hands['thumb_ready'].get('wait_arrival', False))
        self.assertEqual(hands['closed']['hand'], 'pinch_without_pinky')
        self.assertEqual(hands['released']['hand'], 'pinch_high_thumb_ready_without_pinky')
        self.assertTrue(all(event['hand_ramp_seconds'] == 2. for event in hands.values()))
        self.assertEqual(hands['empty_pickup_preshape']['hand'], hands['preshaped']['hand'])
        self.assertTrue(hands['empty_pickup_preshape']['requires_tabletop_target'])
        self.assertEqual([e for e in staged if 'hand' not in e], [e for e in events if 'hand' not in e])
        names = [e['name'] for e in staged]
        self.assertLess(names.index('aligned'), names.index('preshaped'))
        self.assertLess(names.index('preshaped'), names.index('approached'))
        self.assertEqual(names[names.index('contact') + 1:names.index('closed') + 1],
                         ['thumb_ready', 'thumb_opposed', 'closed'])
        self.assertEqual(events, original)

    def test_preshape_prepares_fingers_high_and_thumb_at_contact_before_closing(self):
        events, _ = make_plan(self.start, self.source, .009, 'pickup')
        original = [dict(event) for event in events]
        staged = preshape_fingers_plan(events)
        support_index = next(i for i, e in enumerate(staged) if e['name'] == 'preshaped')
        self.assertEqual(staged[support_index]['hand'], 'preshape_without_pinky')
        self.assertTrue(staged[support_index]['capture'])
        self.assertEqual(staged[support_index - 1]['name'], 'aligned')
        self.assertGreater(staged[support_index - 1]['position'][2] - self.source[2], .25)
        self.assertEqual(staged[support_index + 1]['name'], 'approached')
        self.assertEqual([e for e in staged if 'hand' not in e], [e for e in original if 'hand' not in e])
        released = next(e for e in staged if e['name'] == 'released')
        self.assertEqual(released['hand'], 'preshape_without_pinky')
        self.assertEqual(released['preshape_feedback_key'], 'release_preshape_feedback')
        contact_index = next(i for i, e in enumerate(staged) if e['name'] == 'contact')
        ready = staged[contact_index + 1]
        self.assertEqual(ready['name'], 'thumb_ready')
        self.assertEqual(ready['hand'], 'thumb_ready_without_pinky')
        self.assertTrue(ready['capture'])
        self.assertEqual(ready['preshape_feedback_key'], 'thumb_ready_feedback')
        self.assertEqual(staged[contact_index + 2]['name'], 'closed')
        self.assertFalse(any(e.get('hand') == 'open' for e in staged[contact_index:]))
        self.assertEqual(events, original)
        with self.assertRaises(ValueError):
            preshape_fingers_plan([e for e in events if e['name'] != 'closed'])
        reordered = [e for e in events if e['name'] != 'aligned'] + [next(e for e in events if e['name'] == 'aligned')]
        with self.assertRaises(ValueError):
            preshape_fingers_plan(reordered)

    def test_known_empty_return_uses_the_same_half_shape_before_lowering(self):
        events, _ = make_plan(self.start, self.source, .009, 'pickup')
        release_index = next(i for i, e in enumerate(events) if e['name'] == 'source_release_position')
        event = {'name': 'empty_pickup_open', 'hand': 'open', 'capture': False,
                 'requires_tabletop_target': True, 'wait_open': True}
        events.insert(release_index, event)
        staged = preshape_fingers_plan(events)
        empty = next(e for e in staged if e['name'] == 'empty_pickup_preshape')
        self.assertTrue(empty['requires_tabletop_target'])
        self.assertTrue(empty['capture'])
        self.assertFalse(empty['wait_open'])
        self.assertEqual(empty['hand'], 'preshape_without_pinky')
        self.assertEqual(empty['preshape_feedback_key'], 'empty_return_preshape_feedback')
        self.assertEqual(event['hand'], 'open')

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
