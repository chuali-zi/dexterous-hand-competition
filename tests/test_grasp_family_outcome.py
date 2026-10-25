"""Outcome checks must distinguish real transport from stationary relative poses."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_grasp_families import outcome


def carried_rows():
    return [
        dict(
            seconds=t,
            left_root_position=[0, 0, 0.1 + dz],
            target_position=[0, 0, 0.025 + dz],
            target_in_hand_m=[0, 0, -0.075],
            target_in_hand_quaternion_wxyz=[1, 0, 0, 0],
            bottom_m=dz,
            target_contacts=[["target", "lh_thumb_distal"]],
        )
        for t, dz in [(6.5, 0.0), (8.5, 0.04), (9.5, 0.04), (10.5, 0.04), (11.5, 0.04)]
    ]


class GraspOutcomeTests(unittest.TestCase):
    def test_rigid_carry_and_hold_has_a_local_result_only(self):
        result = outcome(carried_rows())
        self.assertTrue(result["local_lift_hold_pass"])
        self.assertFalse(result["native_grasp_verified"])

    def test_hand_rises_but_stationary_table_object_is_not_a_grasp(self):
        rows = carried_rows()
        for r in rows[1:]:
            r.update(
                target_position=[0, 0, 0.025],
                target_in_hand_m=[0, 0, -0.115],
                bottom_m=0.0,
                target_contacts=[["table", "target"]],
            )
        result = outcome(rows)
        self.assertAlmostEqual(result["target_relative_drift_mm"], 0.0)
        self.assertFalse(result["local_lift_hold_pass"])

    def test_height_without_hand_contact_or_with_external_bracing_is_not_a_grasp(self):
        for contacts in ([], [["neighbor_1", "target"], ["target", "lh_thumb_distal"]]):
            rows = carried_rows()
            for r in rows[1:]:
                r["target_contacts"] = contacts
            self.assertFalse(outcome(rows)["local_lift_hold_pass"])

    def test_slip_or_rotation_during_hold_fails(self):
        rows = carried_rows()
        rows[-1]["target_in_hand_m"][0] = 0.01
        self.assertFalse(outcome(rows)["local_lift_hold_pass"])
        rotated = deepcopy(carried_rows())
        rotated[-1]["target_in_hand_quaternion_wxyz"] = [0.98480775, 0.17364818, 0, 0]
        self.assertFalse(outcome(rotated)["local_lift_hold_pass"])


if __name__ == "__main__":
    unittest.main()
