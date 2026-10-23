"""Numerical tool-to-hand mounting calibration from aligned robot-only feedback.

Imports the vendor's numerical SDK only in main; never constructs a robot or
publishes commands. One stationary sample does not validate the whole arm path.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from rabokit_calibration_analysis import (
    LEFT_ROOT,
    RIGHT_ROOT,
    LEFT_HAND_BASE,
)
from rabokit_robot_frames import mount_input_from_sample

RIGHT_HAND_BASE = "rcd72e2_link_88a3e0c332"


def mounting_from_feedback(record, forward, native_to_sdk):
    """Accept a complete capture or its explicit world-transform/arm reduction."""
    if "joint_aligned_sample" in record:
        record = mount_input_from_sample(record["joint_aligned_sample"])
    if record["stamp"] == [0, 0]:
        raise ValueError("Expected a nonzero aligned feedback stamp")
    transforms = record["world_transforms"]
    basis = np.asarray(transforms[LEFT_ROOT][0], dtype=float)
    origin = (np.asarray(transforms[LEFT_ROOT][1]) + transforms[RIGHT_ROOT][1]) / 2
    result = dict(stamp=record["stamp"], motion_commanded=False, sides={})
    for side, hand in (("left", LEFT_HAND_BASE), ("right", RIGHT_HAND_BASE)):
        values = record["arms"]["arm_" + side]
        native = [values.get(f"arm_{side}_J{i}", float("nan")) for i in range(1, 8)]
        if not np.isfinite(native).all():
            raise ValueError(f"Missing or nonfinite {side} arm feedback")
        joints = native_to_sdk(side, native)
        hand_rotation, hand_position = transforms[hand]
        hand_sdk = np.eye(4)
        hand_sdk[:3, :3] = basis.T @ hand_rotation
        hand_sdk[:3, 3] = basis.T @ (np.asarray(hand_position) - origin)
        tool_sdk = np.asarray(forward(side, joints), dtype=float)
        mount = np.linalg.inv(tool_sdk) @ hand_sdk
        if not np.isfinite(mount).all():
            raise ValueError("Nonfinite mounting transform")
        result["sides"][side] = dict(
            native_joints=native, sdk_joints=list(joints), tool_to_hand=mount.tolist()
        )
    return result


def signed_joint_ranges(native_ranges, signs):
    """The order of bounds reverses on negative native-to-SDK axes."""
    if (
        len(native_ranges) != 7
        or len(signs) != 7
        or any(s not in (-1, 1) for s in signs)
    ):
        raise ValueError("Expected seven joint ranges and unit signs")
    return [
        sorted([sign * lo, sign * hi]) for (lo, hi), sign in zip(native_ranges, signs)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--scene-limits", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    if args.sdk_root:
        sys.path.insert(0, str(args.sdk_root.resolve()))
    import rabo_robolib
    from _robolib_core.arms import _lbot_kinematics as k

    record = json.loads(args.input.read_text(encoding="utf-8"))
    # Public API documents this sign conversion as self-inverse for feedback.
    result = mounting_from_feedback(record, k.forward_kinematics, k.to_sim_joints)
    result.update(sdk_version=rabo_robolib.__version__, scope=__doc__)
    scene = (
        json.loads(args.scene_limits.read_text(encoding="utf-8"))
        if args.scene_limits
        else None
    )
    for side, row in result["sides"].items():
        row["native_to_sdk_sign"] = [k.sim_joint_sign(side, i) for i in range(7)]
        row["sdk_preset_joint_limits"] = k.real_joint_limits(side)
        if scene:
            row["saved_scene_joint_limits_sdk"] = signed_joint_ranges(
                scene["native_ranges"][side], row["native_to_sdk_sign"]
            )
            row["limits_scope"] = (
                "Read-only saved-scene bounds and SDK presets are reported separately; this does not change SDK validation."
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
