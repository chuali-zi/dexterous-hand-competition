"""Reject the observed IK branch jump before any path can be used for motion."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_joint_path import continuous_inverse, interpolate_rotation, pickup_inspection_plan, pose_path, seeded_approach_plan, translation_path


class JointPathTests(unittest.TestCase):
    def test_fine_samples_merge_only_when_the_original_interpolation_checks_pass(self):
        def forward(joints):
            pose = np.eye(4)
            pose[:3, 3] = joints[:3]
            return pose
        start = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}
        inverse = lambda target, seed: [*target[:3, 3], 0., 0., 0., 0.]
        result = translation_path(start, [.04, 0., 0.], inverse, forward, spacing=.005)
        self.assertTrue(result['continuous'])
        self.assertLess(len(result['points']), result['control_point_reduction']['original_count'])

        def curved_forward(joints):
            pose = forward(joints)
            pose[1, 3] += 200 * joints[0] ** 2
            return pose
        curved_inverse = lambda t, s: [t[0, 3], t[1, 3] - 200 * t[0, 3] ** 2, t[2, 3], 0., 0., 0., 0.]
        curved = translation_path(start, [.02, 0., 0.], curved_inverse, curved_forward, spacing=.005)
        self.assertTrue(curved['continuous'])
        self.assertEqual(len(curved['points']), curved['control_point_reduction']['original_count'])

    def test_fk_continuation_preserves_accepted_sdk_solution_without_extra_work(self):
        solution = [.04, 0., 0., 0., 0., 0., 0.]
        def unexpected_forward(joints):
            self.fail('An accepted SDK result should not trigger local solving')
        result = continuous_inverse(np.eye(4), [0.] * 7, lambda t, s: solution,
                                    unexpected_forward, [(-1., 1.)] * 7)
        self.assertEqual(result, solution)

    def test_fk_continuation_recovers_a_close_pose_but_rejects_bounds(self):
        def forward(joints):
            pose = np.eye(4)
            angle = joints[3]
            pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                            [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
            pose[:3, 3] = joints[:3]
            return pose
        target = forward([.04, .02, .01, .07, 0., 0., 0.])
        jumped = lambda t, s: [2.] * 7
        recorded = []
        solved = continuous_inverse(target, [0.] * 7, jumped, forward, [(-1., 1.)] * 7, recorded)
        np.testing.assert_allclose(forward(solved), target, atol=1e-5)
        self.assertLess(max(abs(value) for value in solved), .15)
        self.assertEqual(len(recorded), 1)
        self.assertIsNone(continuous_inverse(target, [0.] * 7, jumped, forward,
                                            [(-.03, .03)] + [(-1., 1.)] * 6))
        target[0, 3] = .2
        self.assertIsNone(continuous_inverse(target, [0.] * 7, jumped, forward, [(-1., 1.)] * 7))

    def test_tool_xy_adjustment_keeps_source_and_source_return_consistent(self):
        def forward(joints):
            transform = np.eye(4)
            transform[:3, 3] = joints[:3]
            return transform

        inverse = lambda target, seed: [*target[:3, 3], 0., 0., 0., 0.]
        start = {'joints': [0., .222, -.6082, 0., 0., 0., 0.], 'pose': [[0., .222, -.6082], [0.] * 3]}
        point = [.22875, .08125, -.497625]
        reference = {'completed': [{'name': 'contact', 'measured': {
            'joints': [.14875, .08125, point[2] + .035, 0., 0., 0., 0.]}}]}
        events, _ = seeded_approach_plan(start, point, .009, .035, reference, 'pickup', inverse, forward,
                                         tool_offset_y=.01, tool_offset_x=-.005)
        events, _ = pickup_inspection_plan(events, point, inverse, forward,
                                           tool_offset_y=.01, tool_offset_x=-.005)
        by_name = {event['name']: event for event in events}
        for name in ('contact', 'followed', 'inspection_low', 'inspection_followed'):
            self.assertAlmostEqual(by_name[name]['joints'][0], point[0] - .08 - .005)
        self.assertAlmostEqual(by_name['contact']['joints'][1], point[1] + .01)
        self.assertAlmostEqual(by_name['followed']['joints'][1], point[1] + .01)
        self.assertAlmostEqual(by_name['inspection_low']['joints'][1], point[1] + .01 - .08)
        self.assertEqual(by_name['source_release_position']['reverse_of'], 'contact')
        self.assertEqual(point, [.22875, .08125, -.497625])

    def test_seeded_route_refines_translation_without_raising_joint_step_limit(self):
        def forward(joints):
            transform = np.eye(4)
            transform[:3, 3] = [joints[0], joints[1], joints[2] / 20]
            return transform

        def inverse(target, seed):
            return [target[0, 3], target[1, 3], target[2, 3] * 20, 0., 0., 0., 0.]

        start = {'joints': [0., .222, -.6082 * 20, 0., 0., 0., 0.],
                 'pose': [[0., .222, -.6082], [0.] * 3]}
        point = [.22875, .08125, -.497625]
        reference = {'completed': [{'name': 'contact', 'measured': {
            'joints': [.14875, .08125, (point[2] + .04) * 20, 0., 0., 0., 0.]}}]}
        events, checks = seeded_approach_plan(start, point, .009, .04, reference, 'pickup', inverse, forward)
        self.assertEqual(checks[0]['spacing_m'], .005)
        self.assertIn('changed branch', checks[0]['coarse_plan_rejection'])
        previous = start['joints']
        for event in events:
            if 'joints' in event:
                self.assertLessEqual(max(abs(a - b) for a, b in zip(previous, event['joints'])), .15)
                previous = event['joints']

        def unavoidable_jump(target, seed):
            joints = inverse(target, seed)
            if target[2, 3] > point[2] + .08:
                joints[-1] = .5
            return joints

        with self.assertRaisesRegex(ValueError, '5 mm route rejected:.*changed branch'):
            seeded_approach_plan(start, point, .009, .04, reference, 'pickup', unavoidable_jump, forward)

    def test_pickup_peek_samples_two_heights_and_retraces_before_source_release(self):
        def forward(joints):
            transform = np.eye(4)
            transform[:3, 3] = joints[:3]
            return transform
        def inverse(target, seed):
            if target[2, 3] > seed[2] + 1e-6:
                return None  # The observed upward IK failure must not be retried.
            return [*target[:3, 3], 0., 0., 0., 0.]
        point = [.22875, .08125, -.497625]
        events = [{'name': 'followed', 'joints': [.14875, .08125, -.247625, 0., 0., 0., 0.], 'capture': True},
                  {'name': 'source_release_position', 'reverse_of': 'contact', 'capture': False},
                  {'name': 'released', 'hand': 'open', 'capture': True}]
        planned, _ = pickup_inspection_plan(events, point, inverse, forward)
        names = [e['name'] for e in planned]
        for name, height in (('inspection_low', .20), ('inspection_followed', .25)):
            event = next(e for e in planned if e['name'] == name)
            np.testing.assert_allclose(event['joints'][:3], [point[0] - .08, point[1] - .08, point[2] + height])
            self.assertTrue(event['capture'])
        self.assertLess(names.index('inspection_followed'), names.index('inspection_source_return'))
        self.assertEqual(next(e for e in planned if e['name'] == 'inspection_followed')['reverse_of'],
                         'inspection_high')
        self.assertLess(names.index('inspection_source_return'), names.index('source_release_position'))
        self.assertLess(names.index('inspection_source_return'), names.index('empty_pickup_open'))
        self.assertLess(names.index('empty_pickup_open'), names.index('source_release_position'))
        opening = next(e for e in planned if e['name'] == 'empty_pickup_open')
        self.assertEqual(opening['hand'], 'open')
        self.assertTrue(opening['requires_tabletop_target'])
        self.assertTrue(opening['wait_open'])
        self.assertLess(names.index('source_release_position'), names.index('released'))
        seen = {'contact'}
        for event in planned:
            if 'reverse_of' in event:
                self.assertIn(event['reverse_of'], seen)
            seen.add(event['name'])
        self.assertNotIn('transferred', names)

    def test_seeded_pickup_returns_to_source_before_opening_and_retraces_joint_samples(self):
        def forward(joints):
            transform = np.eye(4)
            transform[:3, 3] = joints[:3]
            return transform

        inverse = lambda target, seed: [*target[:3, 3], 0., 0., 0., 0.]
        start = {'joints': [0., .222, -.6082, 0., 0., 0., 0.], 'pose': [[0., .222, -.6082], [0.] * 3]}
        point = [.22875, .08125, -.497625]
        reference = {'completed': [{'name': 'contact', 'measured': {
            'joints': [.14875, .08125, -.457625, 0., 0., 0., 0.]}}]}
        events, _ = seeded_approach_plan(start, point, .009, .04, reference, 'pickup', inverse, forward)
        names = [event['name'] for event in events]
        for first, second in zip(('contact', 'closed', 'lifted', 'followed', 'source_release_position'),
                                 ('closed', 'lifted', 'followed', 'source_release_position', 'released')):
            self.assertLess(names.index(first), names.index(second))
        self.assertNotIn('transferred', names)
        seen = set()
        for event in events:
            if 'reverse_of' in event:
                self.assertIn(event['reverse_of'], seen)
            seen.add(event['name'])
        touch, _ = seeded_approach_plan(start, point, .009, .04, reference, 'touch', inverse, forward)
        self.assertFalse(any(event.get('hand') == 'close' for event in touch))

    def test_rotation_interpolation_handles_half_turn_and_seeded_turn(self):
        half_turn = np.diag([-1., -1., 1.])
        middle = interpolate_rotation(np.eye(3), half_turn, .5)
        np.testing.assert_allclose(middle @ middle, half_turn, atol=1e-8)
        np.testing.assert_allclose(middle.T @ middle, np.eye(3), atol=1e-8)

        def forward(joints):
            pose = np.eye(4)
            angle = joints[0]
            pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            return pose

        target = forward([.2] * 7)
        start = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}
        result = pose_path(start, target, lambda t, s: [np.arctan2(t[1, 0], t[0, 0])] + [0.] * 6, forward)
        self.assertTrue(result['continuous'])
        self.assertAlmostEqual(result['points'][-1]['joints'][0], .2)

    def test_continuous_seed_is_propagated_and_branch_jump_is_rejected(self):
        start = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}

        def forward(joints):
            pose = np.eye(4)
            pose[:3, 3] = joints[:3]
            return pose

        seeds = []

        def inverse(target, seed):
            seeds.append(list(seed))
            return [*target[:3, 3], 0., 0., 0., 0.]

        result = translation_path(start, [.03, 0, 0], inverse, forward)
        self.assertTrue(result['continuous'])
        self.assertAlmostEqual(seeds[1][0], .01)
        jump = translation_path(start, [.03, 0, 0], lambda t, s: [5.78] * 7, forward)
        self.assertFalse(jump['continuous'])
        self.assertEqual(jump['failed_step'], 1)
        self.assertEqual(jump['points'], [])

    def test_unreachable_target_and_interpolation_deviation_are_rejected(self):
        start = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}

        def forward(joints):
            pose = np.eye(4)
            pose[0, 3] = joints[0] * 2
            return pose

        missing = translation_path(start, [.01, 0, 0], lambda t, s: None, forward)
        self.assertFalse(missing['continuous'])
        wrong = translation_path(start, [.01, 0, 0], lambda t, s: [.01] * 7, forward)
        self.assertFalse(wrong['continuous'])
        self.assertIn('interpolation', wrong['reason'])
