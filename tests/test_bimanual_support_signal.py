"""Support-history checks do not establish native grasp success."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bimanual_support_signal import left_blocking_loss, support_evidence


def history(start, mcp, dip):
    return [
        dict(
            seconds=start + i * 0.04,
            servo_effort={"rh_ring_mcp_pitch": mcp, "rh_ring_dip": dip},
        )
        for i in range(6)
    ]


class SupportSignalTests(unittest.TestCase):
    def test_blocked_to_free_hint_needs_both_fingers_and_low_current_speed(self):
        baseline = [
            dict(
                position=dict(lh_thumb_cmc_pitch=0.31, lh_index_mcp_pitch=0.18),
                servo_effort=dict(lh_thumb_cmc_pitch=0.8, lh_index_mcp_pitch=0.8),
            )
            for _ in range(5)
        ]
        targets = dict(lh_thumb_cmc_pitch=0.35, lh_index_mcp_pitch=0.22)
        current = dict(
            position=dict(targets),
            velocity=dict.fromkeys(targets, 0),
            effort=dict.fromkeys(targets, 0),
        )
        self.assertTrue(left_blocking_loss(baseline, current, targets)["lost_hint"])
        current["velocity"]["lh_thumb_cmc_pitch"] = 0.2
        self.assertFalse(left_blocking_loss(baseline, current, targets)["lost_hint"])
        current["velocity"]["lh_thumb_cmc_pitch"] = 0
        current["position"]["lh_index_mcp_pitch"] = 0.18
        self.assertFalse(left_blocking_loss(baseline, current, targets)["lost_hint"])

    def test_missing_or_unblocked_reference_does_not_produce_a_loss_verdict(self):
        targets = dict(lh_thumb_cmc_pitch=0.35, lh_index_mcp_pitch=0.22)
        current = dict(
            position=dict(targets),
            velocity=dict.fromkeys(targets, 0),
            effort=dict.fromkeys(targets, 0),
        )
        baseline = [
            dict(position=dict(targets), servo_effort=dict.fromkeys(targets, 0.001))
            for _ in range(5)
        ]
        self.assertFalse(
            left_blocking_loss(baseline, current, targets)["reference_usable"]
        )
        baseline[0]["position"]["lh_thumb_cmc_pitch"] = None
        self.assertFalse(left_blocking_loss(baseline, current, targets)["lost_hint"])

    def test_sustained_two_channel_change_uses_the_earlier_baseline(self):
        before = history(6.1, 0.2, -0.1)
        self.assertTrue(support_evidence(before, history(9.3, 0.25, -0.08))["ready"])
        self.assertFalse(support_evidence(before, history(9.3, 0.2, -0.1))["ready"])
        self.assertFalse(support_evidence(before, history(9.3, 0.25, -0.1))["ready"])

    def test_one_impact_sample_and_object_fields_cannot_approve_lifting(self):
        before = history(6.1, 0.0, 0.0)
        recent = history(9.3, 0.0, 0.0)
        recent[-1]["servo_effort"] = {"rh_ring_mcp_pitch": 0.5, "rh_ring_dip": 0.2}
        for row in recent:
            row.update(contacts=[["target", "rh_ring_distal"]], bottom_m=1.0)
        self.assertFalse(support_evidence(before, recent)["ready"])

    def test_missing_short_future_or_nonfinite_history_is_not_support(self):
        before = history(6.1, 0.0, 0.0)
        recent = history(9.3, 0.1, 0.1)
        self.assertFalse(support_evidence([], recent)["ready"])
        self.assertFalse(support_evidence(before, recent[:3])["ready"])
        self.assertFalse(support_evidence(history(10.0, 0.0, 0.0), recent)["ready"])
        missing = history(9.3, 0.1, 0.1)
        missing[2]["servo_effort"].pop("rh_ring_dip")
        self.assertFalse(support_evidence(before, missing)["ready"])
        recent[2]["servo_effort"]["rh_ring_dip"] = float("nan")
        self.assertFalse(support_evidence(before, recent)["ready"])


if __name__ == "__main__":
    unittest.main()
