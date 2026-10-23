"""Move the empty right hand through a locally checked route and return (sim only)."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from rabokit_calibration_probe import measured, move


def check_plan(plan):
    points = [plan["neutral"], *plan["points"]]
    bounds = plan["joint_limits_sdk"]
    if len(bounds) != 7 or not plan["points"]:
        raise ValueError("Expected a nonempty seven-joint route")
    for q in points:
        if len(q) != 7 or any(
            not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(q, bounds)
        ):
            raise ValueError("Nonfinite or out-of-range route")
    maximum = max(
        max(abs(a - b) for a, b in zip(q, r)) for q, r in zip(points, points[1:])
    )
    if maximum > 0.15 + 1e-9:
        raise ValueError("Route exceeds 0.15 rad per control point")
    return maximum


def joint_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def native_angles(sample, side, convert):
    values = sample["arms"]["arm_" + side]["position"]
    return list(convert(side, [values[f"arm_{side}_J{i}"] for i in range(1, 8)]))


def execute(arm, points, history, budget=120):
    deadline = time.monotonic() + budget
    for index, goal in enumerate(points):
        if time.monotonic() > deadline:
            raise TimeoutError("Route time budget exceeded")
        move(arm, goal)
        history.append(list(goal))
        if index % 10 == 0 or index == len(points) - 1:
            print("RIGHT_EMPTY_POINT", index + 1, "/", len(points), flush=True)


def snapshot(root, label, expected, left_start, convert, rgb=False):
    folder = root / label
    folder.mkdir()
    scripts = Path(__file__).resolve().parent
    commands = [
        [
            sys.executable,
            str(scripts / "rabokit_robot_frames.py"),
            "--arm-feedback",
            "--seconds",
            "2",
            "--output",
            str(folder / "robot-frames.json"),
        ]
    ]
    if rgb:
        commands.append(
            [
                sys.executable,
                str(scripts / "rabokit_stereo.py"),
                "--capture-only",
                "--seconds",
                "1",
                "--output",
                str(folder / "camera"),
            ]
        )
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        (folder / (Path(command[1]).stem + ".log")).write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(
                f"{label}: {Path(command[1]).name} exited {result.returncode}"
            )
    sample = json.loads((folder / "robot-frames.json").read_text())[
        "joint_aligned_sample"
    ]
    right, left = [native_angles(sample, side, convert) for side in ("right", "left")]
    return dict(
        stamp=sample["stamp"],
        right_joints_sdk=right,
        left_joints_sdk=left,
        right_target_error_rad=joint_error(right, expected),
        left_drift_rad=joint_error(left, left_start),
        right_fingers={
            n: v
            for n, v in sample["arms"]["arm_right"]["position"].items()
            if n.startswith("rh_")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/right-empty-approach.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--motion", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    maximum = check_plan(plan)
    if not args.motion:
        print(
            json.dumps(
                dict(
                    motion_commanded=False,
                    points=len(plan["points"]),
                    maximum_joint_step_rad=maximum,
                )
            )
        )
        return
    if not os.getenv("ROS2_NAMESPACE") or args.output is None or args.output.exists():
        parser.error(
            "Motion requires the simulation terminal and a new --output directory"
        )
    from rabo_robolib import LbotDualArm
    from _robolib_core.arms import _lbot_kinematics as k

    args.output.mkdir(parents=True)
    robot = LbotDualArm(mode="sim")
    started = time.monotonic()
    history, report = [], dict(
        target_reached=False,
        returned_verified=False,
        physical_grasp_verified=False,
        official_score=None,
        whole_arm_collision_verified=False,
        errors=[],
    )
    arm = robot.right_arm
    left_start = start = None
    try:
        time.sleep(1)
        start, left_start = measured(arm)["joints"], measured(robot.left_arm)["joints"]
        if (
            joint_error(start, plan["neutral"]) > 0.02
            or max(abs(v) for v in left_start) > 0.02
        ):
            raise RuntimeError("Expected both arms at the retained neutral pose")
        initial = snapshot(
            args.output, "start", start, left_start, k.to_sim_joints, rgb=True
        )
        report["start"] = initial
        if initial["right_target_error_rad"] > 0.01 or initial["left_drift_rad"] > 0.01:
            raise RuntimeError("SDK and native initial arm feedback disagree")
        if (
            not initial["right_fingers"]
            or max(abs(v) for v in initial["right_fingers"].values()) > 0.03
        ):
            raise RuntimeError(
                "Right hand is not empty/open at its retained initial posture"
            )
        if not arm.enable():
            raise RuntimeError("Right arm enable failed")
        history.append(start)
        execute(arm, plan["points"], history)
        report["preinsert"] = snapshot(
            args.output, "preinsert", plan["points"][-1], left_start, k.to_sim_joints
        )
        report["target_reached"] = (
            report["preinsert"]["right_target_error_rad"] < 0.01
            and report["preinsert"]["left_drift_rad"] < 0.01
        )
        if not report["target_reached"]:
            raise RuntimeError(
                "Fresh native feedback did not verify the preinsert posture"
            )
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            if history:
                arm.stop()
                # A failed command may have stopped between checkpoints. Rejoin only
                # the last short segment, then retrace the measured completed route.
                if joint_error(measured(arm)["joints"], history[-1]) > 0.3:
                    raise RuntimeError(
                        "Return skipped: arm is too far from its last checkpoint"
                    )
                returned = []
                execute(arm, list(reversed(history)), returned)
                report["restored"] = snapshot(
                    args.output,
                    "restored",
                    start,
                    left_start,
                    k.to_sim_joints,
                    rgb=True,
                )
                report["returned_verified"] = (
                    report["restored"]["right_target_error_rad"] < 0.01
                    and report["restored"]["left_drift_rad"] < 0.01
                )
                from rabokit_pickup_vision import neighboring_layout, saved_scene

                report["layout"] = neighboring_layout(
                    saved_scene(args.output / "start"),
                    saved_scene(args.output / "restored"),
                )
        except Exception as exc:
            report["errors"].append(f"Return: {type(exc).__name__}: {exc}")
        finally:
            for cleanup in (robot.stop, robot.shutdown):
                try:
                    cleanup()
                except Exception as exc:
                    report["errors"].append(f"Cleanup: {type(exc).__name__}: {exc}")
        report.update(
            wall_seconds=time.monotonic() - started,
            completed_outbound_points=max(0, len(history) - 1),
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("RIGHT_EMPTY_REPORT " + json.dumps(report, allow_nan=False), flush=True)
    if (
        report["errors"]
        or not report["target_reached"]
        or not report["returned_verified"]
        or not report.get("layout", {}).get("consistent")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
