import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rabokit_right_empty_probe import check_plan, execute


class RightEmptyProbeTests(unittest.TestCase):
    def test_saved_route_small_steps_and_return(self):
        plan = json.loads((Path(__file__).resolve().parents[1]/'configs/right-empty-approach.json').read_text())
        self.assertLessEqual(check_plan(plan), .15+1e-9)
        self.assertEqual(list(reversed([plan['neutral'], *plan['points']]))[-1], plan['neutral'])

    def test_out_of_bounds_rejected(self):
        plan = dict(neutral=[0]*7, points=[[.1]*7], joint_limits_sdk=[[-.05, .05]]*7)
        with self.assertRaises(ValueError):
            check_plan(plan)

    def test_failed_command_is_not_a_completed_checkpoint(self):
        history = [[0]*7]
        with patch('rabokit_right_empty_probe.move', side_effect=[None, TimeoutError('blocked')]):
            with self.assertRaises(TimeoutError):
                execute(object(), [[.1]*7, [.2]*7], history)
        self.assertEqual(history, [[0]*7, [.1]*7])


if __name__ == '__main__':
    unittest.main()
