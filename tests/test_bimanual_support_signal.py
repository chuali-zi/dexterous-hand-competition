"""Support-history checks do not establish native grasp success."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bimanual_support_signal import support_evidence


def history(start, mcp, dip):
    return [
        dict(
            seconds=start + i * 0.04,
            servo_effort={"rh_ring_mcp_pitch": mcp, "rh_ring_dip": dip},
        )
        for i in range(6)
    ]


class SupportSignalTests(unittest.TestCase):
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
