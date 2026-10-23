from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rabokit_hand import HAND_JOINTS, preparation_feedback, targets


class BimanualHandTargetsTests(unittest.TestCase):
    def test_tilt_shapes_keep_other_fingers_open_and_change_only_four_joints(self):
        pre, goal = targets("bimanual_preshape"), targets("bimanual_grasp")
        self.assertEqual(set(pre), set(HAND_JOINTS))
        self.assertEqual(
            {n for n in goal if pre[n] != goal[n]},
            {"lh_thumb_cmc_pitch", "lh_thumb_ip", "lh_index_mcp_pitch", "lh_index_dip"},
        )
        for n in (
            "lh_thumb_cmc_yaw",
            "lh_middle_mcp_pitch",
            "lh_ring_mcp_pitch",
            "lh_pinky_mcp_pitch",
        ):
            self.assertEqual(pre[n], goal[n])
        self.assertEqual(pre["lh_thumb_cmc_yaw"], 0.8)
        for finger in ("middle", "ring", "pinky"):
            self.assertEqual(pre["lh_" + finger + "_mcp_pitch"], -0.082)
            self.assertEqual(pre["lh_" + finger + "_dip"], -0.0837)
        self.assertEqual(
            (goal["lh_thumb_cmc_pitch"], goal["lh_thumb_ip"]), (0.35, 1.18)
        )
        self.assertFalse(
            preparation_feedback({}, "bimanual_grasp")["goal_arrival_required"]
        )

    def test_model_checked_shapes_do_not_silently_mix_with_other_thumb_profiles(self):
        self.assertEqual(targets("neutral"), dict.fromkeys(HAND_JOINTS, 0.0))
        for pose in ("bimanual_preshape", "bimanual_grasp", "neutral"):
            with self.assertRaises(ValueError):
                targets(pose, thumb_yaw=1.15)
            with self.assertRaises(ValueError):
                targets(pose, pad_grasp=True)


if __name__ == "__main__":
    unittest.main()
