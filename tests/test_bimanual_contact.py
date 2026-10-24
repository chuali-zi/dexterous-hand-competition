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
            left = [
                dict(
                    stamp=r["stamp"],
                    position=dict(lh_thumb_ip=1.1),
                    effort=dict(lh_thumb_ip=0.2),
                )
                for r in rows
            ]
            (folder / "robot-frames.json").write_text(
                json.dumps(dict(arm_feedback_samples=dict(arm_right=rows)))
            )
            window = hand_history(folder, 10.6)
            (folder / "robot-frames.json").write_text(
                json.dumps(
                    dict(arm_feedback_samples=dict(arm_right=rows, arm_left=left))
                )
            )
            left_window = hand_history(folder, 10.6, side="left")
        self.assertEqual(
            [round(r["seconds"], 1) for r in window], [10.2, 10.3, 10.4, 10.5, 10.6]
        )
        self.assertIsNone(window[0]["servo_effort"]["rh_ring_mcp_pitch"])
        self.assertEqual(left_window[-1]["position"]["lh_thumb_ip"], 1.1)
        self.assertEqual(left_window[-1]["servo_effort"]["lh_thumb_ip"], 0.2)

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
        self.run_recovery_case(None)

    def test_rgb_exclusion_recovers_without_attempting_right_insertion(self):
        self.run_recovery_case("tilted")

    def test_closed_rgb_exclusion_recovers_before_any_tilting(self):
        self.run_recovery_case("closed")

    def test_latest_load_loss_blocks_insertion_even_when_rgb_does_not_exclude(self):
        self.run_recovery_case("released")

    def test_inserted_rgb_failure_keeps_trace_through_inserted_snapshot(self):
        self.run_recovery_case("inserted")

    def run_recovery_case(self, excluded):
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
            left_arm=left,
            right_arm=right,
            left_hand=Mock(),
            right_hand=Mock(),
            stop=Mock(),
        )
        insert_failed = False

        def execute(arm, points, history, budget=120):
            nonlocal insert_failed
            if arm is right and not insert_failed:
                insert_failed = True
                if excluded != "inserted":
                    state[id(arm)] = list(points[0])
                    history.append(list(points[0]))
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
            "rabokit_bimanual_contact.start_tilt_recording", return_value=None
        ), patch(
            "rabokit_bimanual_contact.finish_tilt_recording", return_value=None
        ) as finish, patch(
            "rabokit_bimanual_contact.closing_rgb_evidence", return_value={}
        ), patch(
            "rabokit_bimanual_contact.capture_contact",
            side_effect=lambda root, label, *args: dict(
                stamp=[
                    {"contact_closed": 1, "contact_tilted": 2, "contact_inserted": 3}[
                        label
                    ],
                    0,
                ],
                right_hand_history=[],
                left_target_error_rad=0.0,
                right_target_error_rad=0.0,
                feedback_alignment_valid=True,
                left_hand_history=(
                    [
                        dict(
                            seconds=1 + i * 0.1,
                            position=dict(
                                lh_thumb_cmc_pitch=0.31, lh_index_mcp_pitch=0.18
                            ),
                            servo_effort=dict(
                                lh_thumb_cmc_pitch=0.8, lh_index_mcp_pitch=0.8
                            ),
                        )
                        for i in range(5)
                    ]
                    if excluded == "released"
                    else []
                ),
                left_fingers=dict(
                    position=dict(lh_thumb_cmc_pitch=0.35, lh_index_mcp_pitch=0.218),
                    velocity=dict(lh_thumb_cmc_pitch=0, lh_index_mcp_pitch=0),
                    effort=dict(lh_thumb_cmc_pitch=0, lh_index_mcp_pitch=0),
                ),
                nominal_tilt_visibility=dict(
                    excluded=label == "contact_" + str(excluded)
                ),
            ),
        ):
            with self.assertRaises(RuntimeError if excluded else TimeoutError):
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
        self.assertEqual(insert_failed, excluded in (None, "inserted"))
        if excluded == "closed":
            self.assertEqual(report["completed_points"]["tilt"], 0)
        if excluded == "inserted":
            self.assertEqual(finish.call_args.args[2], 3.0)
            self.assertIn("finished_monotonic", report["stage_timing"]["insert"])


if __name__ == "__main__":
    unittest.main()
