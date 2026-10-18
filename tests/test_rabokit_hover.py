"""Check the settling wait that prevents sampling a still-moving arm."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rabokit_hover as hover


class HoverTests(unittest.TestCase):
    def test_pass_through_requires_arrival_but_does_not_claim_settling(self):
        def forward(joints):
            pose = np.eye(4)
            pose[:3, 3] = joints[:3]
            return pose
        elapsed = {}
        for settled in (False, True):
            clock, calls = [0.], [0]
            def state(_):
                calls[0] += 1
                x = 0. if calls[0] == 1 else .005 if calls[0] == 2 else .01
                last = (.004 if calls[0] % 2 else .006) if calls[0] < 8 else 0.
                return {'joints': [x, 0., 0., 0., 0., 0., last], 'pose': [[x, 0., 0.], [0.] * 3]}
            with patch.object(hover, 'measured', state), \
                    patch.object(hover.time, 'monotonic', lambda: clock[0]), \
                    patch.object(hover.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt)):
                result = hover.move_joint_pose(SimpleNamespace(move_joints=lambda *a, **k: True),
                    [.01, 0., 0., 0., 0., 0., 0.], forward, require_settled=settled)
            self.assertEqual(result['pose'][0][0], .01)
            self.assertEqual(result['settled_verified'], settled)
            elapsed[settled] = clock[0]
        self.assertLess(elapsed[False], .5)
        self.assertGreaterEqual(elapsed[True], 1.)

    def test_quiet_window_rejects_slow_drift_and_short_repeated_feedback(self):
        drifting = [(i * .1, [i * .0002] * 7) for i in range(7)]
        self.assertFalse(hover.quiet_joint_window(drifting))
        self.assertFalse(hover.quiet_joint_window([(0., [0.] * 7), (.2, [0.] * 7)]))
        self.assertTrue(hover.quiet_joint_window([(0., [0.] * 7), (.5, [.0001] * 7)]))

    def test_seeded_tracking_corrects_static_error_without_relaxing_arrival_tolerance(self):
        clock = [0.]
        commanded = [0.] * 7
        sent = []

        def forward(joints):
            pose = np.eye(4)
            pose[:3, 3] = joints[:3]
            return pose

        def state(_):
            joints = list(commanded)
            joints[0] -= .004 if sent else 0.
            return {'joints': joints, 'pose': [joints[:3], [0.] * 3]}

        def move(joints, **_):
            commanded[:] = joints
            sent.append(list(joints))
            return True

        with patch.object(hover, 'measured', state), \
                patch.object(hover.time, 'monotonic', lambda: clock[0]), \
                patch.object(hover.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt)):
            result = hover.move_joint_pose(SimpleNamespace(move_joints=move), [.01, 0., 0., 0., 0., 0., 0.], forward)
        self.assertEqual(len(sent), 2)
        self.assertLess(abs(result['pose'][0][0] - .01), .003)
        self.assertEqual(len(result['tracking_corrections']), 1)

    def test_seeded_joint_tracking_stops_on_measured_sideways_excursion(self):
        def forward(joints):
            pose = np.eye(4)
            pose[:3, 3] = joints[:3]
            return pose

        for settled in (True, False):
            states = iter([{'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]},
                           {'joints': [0., .04, 0., 0., 0., 0., 0.], 'pose': [[0., .04, 0.], [0.] * 3]}])
            stopped = []
            arm = SimpleNamespace(move_joints=lambda *a, **k: True, stop=lambda: stopped.append(True))
            with patch.object(hover, 'measured', lambda _: next(states)):
                with self.assertRaisesRegex(RuntimeError, '15 mm'):
                    hover.move_joint_pose(arm, [.01, 0., 0., 0., 0., 0., 0.], forward, require_settled=settled)
            self.assertEqual(stopped, [True])

    def test_seeded_joint_tracking_rejects_large_step_before_sending(self):
        initial = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}
        arm = SimpleNamespace(move_joints=lambda *a, **k: self.fail('unexpected motion command'))
        with patch.object(hover, 'measured', lambda _: initial):
            with self.assertRaisesRegex(RuntimeError, 'too far'):
                hover.move_joint_pose(arm, [1.] * 7, lambda _: np.eye(4))

    def test_pose_arrival_still_waits_for_joint_settling(self):
        clock = [0.]
        calls = [0]

        def state(_):
            calls[0] += 1
            # Redundant joints can still move while the tool is already in tolerance.
            angle = .01 * min(calls[0], 6)
            return {'joints': [angle] * 7, 'pose': [[.2, .1, -.2], [0, 0, 0]]}

        with patch.object(hover, 'measured', state), \
                patch.object(hover.time, 'monotonic', lambda: clock[0]), \
                patch.object(hover.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt)):
            result = hover.move_pose(SimpleNamespace(move_to=lambda *a, **k: True),
                                     (.2, .1, -.2), (0, 0, 0))
        self.assertEqual(result['joints'], [.06] * 7)
        self.assertGreaterEqual(clock[0], 1.)

    def test_command_rejection_does_not_wait_for_feedback(self):
        arm = SimpleNamespace(move_to=lambda *a, **k: False)
        with patch.object(hover, 'measured', side_effect=AssertionError('unexpected feedback read')):
            with self.assertRaisesRegex(RuntimeError, 'rejected'):
                hover.move_pose(arm, (.2, .1, -.2), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
