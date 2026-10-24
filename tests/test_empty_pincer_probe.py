import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rabokit_empty_pincer_probe import run_probe
from rabokit_hand import HAND_JOINTS


def neutral():
    arms = {}
    for side, prefix in (("left", "lh_"), ("right", "rh_")):
        positions = {f"arm_{side}_J{i}": 0.0 for i in range(1, 8)}
        positions.update({n.replace("lh_", prefix): 0.0 for n in HAND_JOINTS})
        arms["arm_" + side] = dict(position=positions)
    return dict(stamp=[1, 0], arms=arms)


class EmptyPincerTests(unittest.TestCase):
    def run_case(self, close, snapshot, command, force):
        with tempfile.TemporaryDirectory() as temp:
            return run_probe(Path(temp), close, force, snapshot, command)

    def test_default_observation_does_not_setup_force_or_publish_hand_commands(self):
        command, force = Mock(), Mock()
        result = self.run_case(False, Mock(return_value=neutral()), command, force)
        self.assertTrue(result["sequence_completed"])
        command.assert_not_called()
        force.assert_not_called()

    def test_non_neutral_arm_is_rejected_before_hand_commands(self):
        before = neutral()
        before["arms"]["arm_right"]["position"]["arm_right_J3"] = 0.04
        command, force = Mock(), Mock()
        result = self.run_case(True, Mock(return_value=before), command, force)
        self.assertIn("requires the saved neutral", result["error"])
        command.assert_not_called()
        force.assert_not_called()

    def test_failed_closure_still_resets_and_checks_arm_drift(self):
        command = Mock(side_effect=[{}, RuntimeError("closure failed"), {}])
        result = self.run_case(
            True, Mock(return_value=neutral()), command, Mock(return_value=True)
        )
        self.assertIn("closure failed", result["error"])
        self.assertFalse(result["sequence_completed"])
        self.assertTrue(result["neutral_restored"])
        self.assertEqual(command.call_args.args[0], "neutral")
        self.assertFalse(result["physical_grasp_verified"])

    def test_arm_drift_after_reset_is_not_reported_as_recovered(self):
        after = copy.deepcopy(neutral())
        after["arms"]["arm_left"]["position"]["arm_left_J1"] = 0.006
        result = self.run_case(
            True,
            Mock(side_effect=[neutral(), neutral(), after]),
            Mock(),
            Mock(return_value=True),
        )
        self.assertFalse(result["neutral_restored"])
        self.assertIn("recovery_error", result)


if __name__ == "__main__":
    unittest.main()
