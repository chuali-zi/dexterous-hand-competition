import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rabokit_bimanual_contact import (
    check_source,
    compare_support,
    hand_history,
    run_contact,
)


class ContactTests(unittest.TestCase):
    def test_effort_window_ends_at_rgb_stamp_and_excludes_future_feedback(self):
        rows = [
            dict(
                stamp=[10, i * 100_000_000],
                position=dict(rh_ring_mcp_pitch=0),
                effort=dict(rh_ring_mcp_pitch=float("nan"), rh_ring_dip=0.1),
            )
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "robot-frames.json").write_text(
                json.dumps(dict(arm_feedback_samples=dict(arm_right=rows)))
            )
            window = hand_history(folder, 10.6)
        self.assertEqual(
            [round(r["seconds"], 1) for r in window], [10.2, 10.3, 10.4, 10.5, 10.6]
        )
        self.assertIsNone(window[0]["servo_effort"]["rh_ring_mcp_pitch"])

    def test_rgb_reference_must_be_visible_and_close_before_contact(self):
        config = dict(source_surface_sdk_m=[0, 0, 0], source_tolerance_m=0.005)
        for observation, accepted in [
            (dict(valid=False), False),
            (dict(valid=True, point=[0.006, 0, 0]), False),
            (dict(valid=True, point=[0.001, 0, 0]), True),
        ]:
            with patch(
                "rabokit_bimanual_contact.target_observation", return_value=observation
            ):
                self.assertEqual(check_source(Path("."), config)["accepted"], accepted)

    def test_native_missing_effort_cannot_become_a_positive_support_hint(self):
        def rows(start, value):
            return [
                dict(
                    seconds=start + 0.05 * i,
                    servo_effort=dict(rh_ring_mcp_pitch=value, rh_ring_dip=value),
                )
                for i in range(9)
            ]

        self.assertTrue(compare_support(rows(0, 0.01), rows(2, 0.2))["ready"])
        self.assertFalse(compare_support(rows(0, 0.01), rows(2, None))["ready"])

    def test_partial_insert_failure_recovers_both_prepared_arm_positions_and_neutral_hand(
        self,
    ):
        cfg = json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/bimanual-contact.json"
            ).read_text()
        )
        seeds = {
            s: cfg["stages"][n]["start_joints"]
            for s, n in [("left", "descend"), ("right", "insert")]
        }
        left, right = Mock(), Mock()
        state = {id(left): list(seeds["left"]), id(right): list(seeds["right"])}
        robot = SimpleNamespace(
            left_arm=left, right_arm=right, left_hand=Mock(), right_hand=Mock()
        )
        insert_failed = False

        def execute(arm, points, history, budget=120):
            nonlocal insert_failed
            if arm is right and not insert_failed:
                state[id(arm)] = list(points[0])
                history.append(list(points[0]))
                insert_failed = True
                raise TimeoutError("partial insert")
            for q in points:
                state[id(arm)] = list(q)
                history.append(list(q))

        def snapshot(*args, **kwargs):
            return dict(left_target_error_rad=0.0, right_target_error_rad=0.0)

        report = {}
        runs = Path(__file__).resolve().parents[1] / "runs"
        with tempfile.TemporaryDirectory(dir=runs) as temp, patch(
            "rabokit_bimanual_contact.measured",
            side_effect=lambda a: dict(joints=state[id(a)]),
        ), patch(
            "rabokit_bimanual_contact.hand_command",
            return_value=dict(hand_arrival_verified=True),
        ) as hand, patch(
            "rabokit_bimanual_contact.capture_contact",
            return_value=dict(
                right_hand_history=[],
                left_target_error_rad=0.0,
                right_target_error_rad=0.0,
                feedback_alignment_valid=True,
            ),
        ):
            with self.assertRaises(TimeoutError):
                run_contact(
                    robot,
                    Path(temp),
                    copy.deepcopy(cfg),
                    execute,
                    snapshot,
                    lambda s, q: q,
                    report,
                )
        self.assertTrue(report["recovered"])
        self.assertFalse(report["sequence_completed"])
        self.assertEqual(state[id(left)], seeds["left"])
        self.assertEqual(state[id(right)], seeds["right"])
        self.assertEqual(hand.call_args.args[0], "neutral")


if __name__ == "__main__":
    unittest.main()
