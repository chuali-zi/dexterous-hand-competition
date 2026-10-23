"""Mounting inputs must use observed feedback and the correct SDK basis/signs."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_arm_mount import (
    mounting_from_feedback,
    signed_joint_ranges,
    RIGHT_HAND_BASE,
)
from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT, LEFT_HAND_BASE


def example():
    r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    return dict(
        stamp=[1, 20],
        world_transforms={
            LEFT_ROOT: [r, [2.0, 3.2, 4.0]],
            RIGHT_ROOT: [r, [2.0, 2.8, 4.0]],
            LEFT_HAND_BASE: [r, [2.0, 3.5, 4.0]],
            RIGHT_HAND_BASE: [r, [2.0, 2.5, 4.0]],
        },
        arms={
            side: {f"{side}_J{i}": 0.1 * i for i in range(1, 8)}
            for side in ("arm_left", "arm_right")
        },
    )


class MountTests(unittest.TestCase):
    def test_full_capture_and_its_world_transform_reduction_agree(self):
        record = example()
        half = 2**-0.5

        def frame(parent, xyz, quat):
            return dict(parent=parent, xyz=xyz, xyzw=quat)

        frames = {
            LEFT_ROOT: frame("world", [2.0, 3.2, 4.0], [0.0, 0.0, half, half]),
            RIGHT_ROOT: frame("world", [2.0, 2.8, 4.0], [0.0, 0.0, half, half]),
            LEFT_HAND_BASE: frame(LEFT_ROOT, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            RIGHT_HAND_BASE: frame(RIGHT_ROOT, [-0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        }
        full = dict(
            joint_aligned_sample=dict(
                stamp=record["stamp"],
                frames=frames,
                arms={side: dict(position=q) for side, q in record["arms"].items()},
            )
        )

        def forward(side, q):
            return np.eye(4)

        def convert(side, q):
            return q

        reduced = mounting_from_feedback(record, forward, convert)
        actual = mounting_from_feedback(full, forward, convert)
        for side in ("left", "right"):
            np.testing.assert_allclose(
                actual["sides"][side]["tool_to_hand"],
                reduced["sides"][side]["tool_to_hand"],
                atol=1e-14,
            )

    def test_native_angles_are_converted_before_fk_and_world_basis_is_removed(self):
        calls = []

        def forward(side, joints):
            calls.append((side, joints))
            result = np.eye(4)
            result[0, 3] = joints[1]
            return result

        result = mounting_from_feedback(
            example(), forward, lambda side, q: [-v for v in q]
        )
        self.assertAlmostEqual(calls[1][1][1], -0.2)
        left = np.array(result["sides"]["left"]["tool_to_hand"])
        right = np.array(result["sides"]["right"]["tool_to_hand"])
        np.testing.assert_allclose(left[:3, :3], np.eye(3))
        np.testing.assert_allclose(left[:3, 3], [0.7, 0.0, 0.0])
        np.testing.assert_allclose(right[:3, 3], [-0.3, 0.0, 0.0])

    def test_missing_feedback_or_zero_stamp_does_not_become_a_zero_pose_assumption(
        self,
    ):
        record = example()
        record["arms"]["arm_right"].pop("arm_right_J2")
        with self.assertRaisesRegex(ValueError, "right arm feedback"):
            mounting_from_feedback(record, lambda side, q: np.eye(4), lambda side, q: q)
        record = example()
        record["stamp"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "nonzero"):
            mounting_from_feedback(record, lambda side, q: np.eye(4), lambda side, q: q)

    def test_negative_sign_swaps_range_endpoints(self):
        ranges = [[-2.28, 3.85], [-3.3, 0.17]] + [[-1.0, 1.0]] * 5
        converted = signed_joint_ranges(ranges, [1, -1, -1, -1, -1, -1, -1])
        self.assertEqual(converted[:2], [[-2.28, 3.85], [-0.17, 3.3]])


if __name__ == "__main__":
    unittest.main()
