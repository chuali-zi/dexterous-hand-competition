from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rabokit_bimanual_empty_probe import execute_tracked, load_plan, return_completed


class BimanualEmptyTests(unittest.TestCase):
    def test_tracked_failure_retains_active_target_before_any_return(self):
        report, history = {}, [[0] * 7]
        with patch(
            "rabokit_bimanual_empty_probe.move_joint_pose",
            side_effect=[dict(tracking_corrections=[]), TimeoutError("did not settle")],
        ):
            with self.assertRaises(TimeoutError):
                execute_tracked(
                    object(),
                    [[0.1] * 7, [0.2] * 7],
                    history,
                    lambda q: None,
                    report,
                    "left",
                )
        self.assertEqual(history, [[0] * 7, [0.1] * 7])
        self.assertEqual(
            report["active_motion"], dict(side="left", point=2, target_joints=[0.2] * 7)
        )

    def test_stored_plan_loads_and_extends_existing_right_route(self):
        p = load_plan(
            Path(__file__).resolve().parents[1] / "configs/bimanual-empty-approach.json"
        )
        self.assertEqual(len(p["right"]["points"]), 54)
        self.assertEqual(len(p["left"]["points"]), 98)

    def test_left_return_failure_does_not_start_right_return(self):
        robot = SimpleNamespace(left_arm=Mock(), right_arm=Mock())
        history = dict(left=[[0] * 7], right=[[0] * 7])
        with patch(
            "rabokit_bimanual_empty_probe.measured", return_value=dict(joints=[0] * 7)
        ), patch(
            "rabokit_bimanual_empty_probe.execute", side_effect=TimeoutError("blocked")
        ):
            with self.assertRaises(TimeoutError):
                return_completed(robot, history)
        robot.right_arm.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
