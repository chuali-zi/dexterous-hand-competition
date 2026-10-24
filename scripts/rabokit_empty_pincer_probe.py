"""Observe the neutral robot; explicitly compare an empty left pincer and reset.

Fixed simulation mode, no arm trajectory or object transforms. This control
distinguishes empty-hand blocking from target contact; it never proves a grasp.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from rabokit_calibration_probe import hand_command
from rabokit_hand import HAND_JOINTS


def take_snapshot(root, label):
    folder = root / label
    folder.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("rabokit_robot_frames.py")),
            "--arm-feedback",
            "--seconds",
            "1",
            "--output",
            str(folder / "robot-frames.json"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    (folder / "capture.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{label}: robot observation exited {result.returncode}")
    sample = json.loads((folder / "robot-frames.json").read_text())[
        "joint_aligned_sample"
    ]
    if sample is None:
        raise RuntimeError(f"{label}: no aligned robot feedback")
    return dict(stamp=sample["stamp"], arms=sample["arms"])


def joint_values(snapshot, fingers=False):
    values = {}
    for side in ("left", "right"):
        positions = snapshot["arms"]["arm_" + side]["position"]
        names = (
            [n.replace("lh_", "lh_" if side == "left" else "rh_") for n in HAND_JOINTS]
            if fingers
            else [f"arm_{side}_J{i}" for i in range(1, 8)]
        )
        for name in names:
            value = positions.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing or nonfinite robot feedback: {name}")
            values[name] = value
    return values


def run_probe(
    root, close=False, setup_force=None, snapshot=take_snapshot, command=hand_command
):
    report = dict(
        arm_motion_commanded=False,
        hand_motion_requested=close,
        physical_grasp_verified=False,
        sequence_completed=False,
        neutral_restored=False,
        observations={},
    )
    started = time.monotonic()
    hand_started = False
    try:
        before = snapshot(root, "before")
        report["observations"]["before"] = before
        arms = joint_values(before)
        if close:
            if (
                max(
                    abs(v)
                    for v in [*arms.values(), *joint_values(before, True).values()]
                )
                >= 0.03
            ):
                raise RuntimeError(
                    "Empty control requires the saved neutral arm/hand posture"
                )
            if setup_force is None or not setup_force():
                raise RuntimeError("Simulation left-hand force setup rejected")
            report["requested_force_limit_normalized"] = [1.0] * 6
            hand_started = True
            command("bimanual_preshape", wait_arrival=True, ramp_seconds=2.0)
            command(
                "bimanual_grasp",
                ramp_seconds=2.0,
                trace_output=root / "empty-close.json",
            )
            report["observations"]["closed"] = snapshot(root, "closed")
        report["sequence_completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if hand_started:
            try:
                command("neutral", wait_arrival=True, ramp_seconds=2.0)
                after = snapshot(root, "restored")
                report["observations"]["restored"] = after
                report["maximum_arm_drift_rad"] = max(
                    abs(v - arms[n]) for n, v in joint_values(after).items()
                )
                report["neutral_restored"] = (
                    report["maximum_arm_drift_rad"] < 0.005
                    and max(abs(v) for v in joint_values(after, True).values()) < 0.03
                )
                if not report["neutral_restored"]:
                    raise RuntimeError(
                        "Neutral hand or unchanged arm feedback not verified"
                    )
            except Exception as error:
                report["recovery_error"] = f"{type(error).__name__}: {error}"
        report["wall_seconds"] = time.monotonic() - started
        (root / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2) + "\n"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--close-empty-hand", action="store_true")
    args = parser.parse_args()
    if not os.getenv("ROS2_NAMESPACE"):
        parser.error("Run only in the Rabokit simulation terminal")
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    robots = []

    def setup_force():
        from rabo_robolib import LbotDualArm

        robots.append(LbotDualArm(mode="sim"))
        return robots[0].left_hand.set_force([1.0] * 6)

    try:
        report = run_probe(args.output, args.close_empty_hand, setup_force)
    finally:
        for robot in robots:
            robot.shutdown()
    print("EMPTY_PINCER_REPORT " + json.dumps(report, allow_nan=False), flush=True)
    return int(
        "error" in report
        or "recovery_error" in report
        or (args.close_empty_hand and not report["neutral_restored"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
