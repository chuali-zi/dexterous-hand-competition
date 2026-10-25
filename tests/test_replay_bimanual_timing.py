"""Local observation timing and phase labels, without platform or mesh assets."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bimanual_support_signal import support_evidence
from replay_bimanual_support import carry_metrics, sample_due


class ReplayTimingTests(unittest.TestCase):
    def test_support_window_survives_accumulated_two_ms_clock_error(self):
        rows = []
        seconds = 0.0
        while seconds < 22.302:
            if sample_due(seconds, len(rows)):
                rows.append(
                    dict(
                        seconds=seconds,
                        servo_effort=dict(rh_ring_mcp_pitch=0.1, rh_ring_dip=0.02),
                    )
                )
            seconds += 0.002
        # The old next_sample = t + .04 drifted to 42 ms intervals here,
        # leaving only four observations and rejecting a sustained signal.
        recent = [r for r in rows if seconds - 0.2 <= r["seconds"] < seconds]
        baseline = [
            dict(
                seconds=i * 0.04,
                servo_effort=dict(rh_ring_mcp_pitch=0.0, rh_ring_dip=0.0),
            )
            for i in range(5)
        ]
        self.assertEqual(len(recent), 5)
        self.assertTrue(support_evidence(baseline, recent)["ready"])
        self.assertAlmostEqual(rows[-1]["seconds"], 22.28, places=8)

    def test_contact_only_has_no_carry_or_hold_measurements(self):
        result = carry_metrics([], contact_only=True)
        self.assertFalse(result.pop("carry_metrics_available"))
        self.assertTrue(all(value is None for value in result.values()))

    def test_payload_schedule_keeps_existing_metrics(self):
        rows = [
            dict(
                seconds=t,
                bottom_m=z,
                target_position=[0.0, y, z],
                contacts=contacts,
            )
            for t, z, y, contacts in (
                (12.0, 0.01, 0.0, [["table", "target"]]),
                (16.0, 0.02, 0.01, []),
                (19.0, 0.02, 0.014, [["neighbor_1", "target"]]),
            )
        ]
        result = carry_metrics(rows, contact_only=False)
        self.assertTrue(result["carry_metrics_available"])
        self.assertAlmostEqual(result["minimum_carry_bottom_mm"], 10.0)
        self.assertAlmostEqual(result["carry_table_fraction"], 1 / 3)
        self.assertAlmostEqual(result["carry_neighbor_fraction"], 1 / 3)
        self.assertAlmostEqual(result["hold_drift_mm"], 4.0)


if __name__ == "__main__":
    unittest.main()
