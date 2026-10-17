"""Check the settling wait that prevents sampling a still-moving arm."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rabokit_hover as hover


class HoverTests(unittest.TestCase):
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
