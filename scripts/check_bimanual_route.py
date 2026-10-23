"""Check saved local hand-root trajectories using public numerical SDK FK/IK only.

Never constructs an SDK robot or ROS node. A continuous numerical route does not
verify whole-arm collision, native tracking, grasping, or competition score.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rabokit_joint_path import centered_inverse, pose_path


def joint_bounds(preset, scene, margin):
    preset, scene = np.asarray(preset), np.asarray(scene)
    if (
        preset.shape != (7, 2)
        or scene.shape != (7, 2)
        or not math.isfinite(margin)
        or margin < 0
    ):
        raise ValueError(
            "Expected seven SDK-coordinate ranges and a nonnegative margin"
        )
    bounds = np.column_stack(
        (
            np.maximum(preset[:, 0], scene[:, 0]) + margin,
            np.minimum(preset[:, 1], scene[:, 1]) - margin,
        )
    )
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Empty or nonfinite joint range intersection")
    return bounds


def hand_pose(row, side):
    pose = np.eye(4)
    pose[:3, 3] = row[side + "_root_position"]
    pose[:3, :3] = Rotation.from_quat(
        np.asarray(row[side + "_root_quaternion"])[[1, 2, 3, 0]]
    ).as_matrix()
    return pose


def check_side(rows, mount, seed, bounds, forward):
    if len(rows) < 2 or any(
        a["seconds"] >= b["seconds"] for a, b in zip(rows, rows[1:])
    ):
        raise ValueError("Expected at least two strictly increasing trajectory times")
    inverse_mount = np.linalg.inv(np.asarray(mount))
    stats = {}

    def inverse(target, q):
        return centered_inverse(target, q, forward, bounds, stats)

    goal = rows[0]["hand_pose"] @ inverse_mount
    q = inverse(goal, seed)
    if q is None:
        return dict(
            continuous=False, reason="Initial pose has no nearby solution", segments=[]
        )
    initial = list(q)
    segments = []
    # Retain the same 5-row selection used in the prior offline checks.
    selected = rows[::5]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    for row in selected[1:]:
        fk = np.asarray(forward(q))
        path = pose_path(
            dict(joints=q, pose=[fk[:3, 3].tolist(), [0, 0, 0]]),
            row["hand_pose"] @ inverse_mount,
            inverse,
            forward,
            spacing=0.005,
        )
        segments.append(dict(seconds=row["seconds"], **path))
        if not path["continuous"]:
            break
        q = path["points"][-1]["joints"]
    points = [initial] + [p["joints"] for s in segments for p in s["points"]]
    return dict(
        initial_joints=initial,
        continuous=all(s["continuous"] for s in segments),
        segments=segments,
        stats=stats,
        retained_joint_points=len(points),
        minimum_inset_margin_rad=float(
            min(
                np.minimum(np.asarray(q) - bounds[:, 0], bounds[:, 1] - q).min()
                for q in points
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "mounts", "seeds", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--margin", type=float, default=0.02)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    if args.sdk_root:
        sys.path.insert(0, str(args.sdk_root.resolve()))
    import rabo_robolib
    from _robolib_core.arms import _lbot_kinematics as k

    rows = json.loads(args.trace.read_text())["rows"]
    mounts = json.loads(args.mounts.read_text())["sides"]
    seeds = json.loads(args.seeds.read_text())
    arms = {}
    started = time.monotonic()
    for side in ("left", "right"):
        preset = np.asarray(k.real_joint_limits(side))
        scene = np.asarray(mounts[side]["saved_scene_joint_limits_sdk"])
        bounds = joint_bounds(preset, scene, args.margin)
        reduced = [
            dict(seconds=r["seconds"], hand_pose=hand_pose(r, side)) for r in rows
        ]
        arm = check_side(
            reduced,
            mounts[side]["tool_to_hand"],
            seeds[side],
            bounds,
            lambda q: k.forward_kinematics(side, q),
        )
        points = (
            [arm["initial_joints"]]
            + [p["joints"] for s in arm["segments"] for p in s["points"]]
            if "initial_joints" in arm
            else []
        )
        for name, limits in (("sdk_preset", preset), ("saved_scene", scene)):
            arm["minimum_" + name + "_margin_rad"] = (
                float(
                    min(
                        np.minimum(np.asarray(q) - limits[:, 0], limits[:, 1] - q).min()
                        for q in points
                    )
                )
                if points
                else None
            )
        arms[side] = arm
        print(
            side,
            arm["continuous"],
            len(arm["segments"]),
            arm["minimum_sdk_preset_margin_rad"],
            flush=True,
        )
    result = dict(
        scope=__doc__,
        sdk_version=rabo_robolib.__version__,
        arms=arms,
        margin_rad=args.margin,
        continuous=all(a["continuous"] for a in arms.values()),
        motion_commanded=False,
        whole_arm_collision_checked=False,
        neutral_to_start_checked=False,
        wall_seconds=time.monotonic() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["continuous"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
