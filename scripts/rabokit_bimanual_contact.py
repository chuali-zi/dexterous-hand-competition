"""Low bimanual contact diagnostic and recovery; no payload lift or object TF."""

import json
import math
from pathlib import Path
import subprocess
import sys


from bimanual_support_signal import support_evidence
from rabokit_calibration_probe import hand_command, measured
from rabokit_joint_path import centered_inverse, pose_path
from rabokit_pickup_vision import target_observation
from rabokit_right_empty_probe import joint_error, native_angles


def check_source(folder, config):
    observation = target_observation(folder)
    error = (
        math.dist(observation["point"], config["source_surface_sdk_m"])
        if observation["valid"]
        else None
    )
    return dict(
        observation=observation,
        error_m=error,
        accepted=error is not None and error <= config["source_tolerance_m"],
    )


def plan_contact(config, seeds, forward):
    import numpy as np
    from scipy.spatial.transform import Rotation

    joints = {s: list(q) for s, q in seeds.items()}
    result = {}
    for name, side in [("descend", "left"), ("tilt", "left"), ("insert", "right")]:
        hand = config["hand_goals"][name]
        target = np.eye(4)
        target[:3, 3] = hand["position"]
        target[:3, :3] = Rotation.from_quat(
            np.array(hand["quaternion_wxyz"])[[1, 2, 3, 0]]
        ).as_matrix()
        target = target @ np.linalg.inv(np.array(config["tool_to_hand"][side]))

        def fk(q):
            return forward(side, q)

        def inverse(t, q):
            return centered_inverse(t, q, fk, config["joint_limits_sdk"][side])

        start = np.asarray(fk(joints[side]))
        path = pose_path(
            dict(joints=joints[side], pose=[start[:3, 3].tolist(), [0, 0, 0]]),
            target,
            inverse,
            fk,
            spacing=0.005,
        )
        if not path["continuous"]:
            raise RuntimeError(f"{name}: no continuous local contact path")
        result[name] = dict(side=side, start_joints=joints[side], **path)
        joints[side] = path["points"][-1]["joints"]
    return result


def hand_history(folder, end_seconds=None):
    frames = json.loads((folder / "robot-frames.json").read_text())
    rows = frames["arm_feedback_samples"]["arm_right"]
    last = (
        end_seconds
        if end_seconds is not None
        else max(r["stamp"][0] + r["stamp"][1] * 1e-9 for r in rows)
    )
    channels = ("rh_ring_mcp_pitch", "rh_ring_dip")

    def number(value):
        return (
            value if isinstance(value, (float, int)) and math.isfinite(value) else None
        )

    return [
        dict(
            seconds=r["stamp"][0] + r["stamp"][1] * 1e-9,
            position={n: number(r["position"].get(n)) for n in channels},
            servo_effort={n: number(r["effort"].get(n)) for n in channels},
        )
        for r in rows
        if last - 0.4 <= r["stamp"][0] + r["stamp"][1] * 1e-9 <= last
    ]


def compare_support(baseline, loaded):
    # Missing/nonfinite native efforts remain null in the saved observations;
    # omit them for the existing rule, which rejects missing channels.
    def usable(rows):
        return [
            dict(
                seconds=r["seconds"],
                servo_effort={
                    n: v for n, v in r["servo_effort"].items() if v is not None
                },
            )
            for r in rows
        ]

    return dict(
        **support_evidence(usable(baseline), usable(loaded)),
        scope="Experimental unchanged local thresholds on native effort; not a calibrated grasp label",
    )


def capture_contact(root, label, goals, convert, snapshot):
    folder = root / label
    folder.mkdir()
    frame_command = [
        sys.executable,
        str(Path(__file__).with_name("rabokit_robot_frames.py")),
        "--arm-feedback",
        "--seconds",
        "8",
        "--output",
        str(folder / "robot-frames.json"),
    ]
    camera_command = [
        sys.executable,
        str(Path(__file__).with_name("rabokit_stereo.py")),
        "--capture-only",
        "--side-camera",
        "--seconds",
        "6",
        "--output",
        str(folder / "camera"),
    ]
    processes = []
    try:
        for command in (frame_command, camera_command):
            processes.append(
                subprocess.Popen(
                    command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            )
        for name, process in zip(("robot-frames", "camera"), processes):
            stdout, stderr = process.communicate(timeout=15)
            (folder / (name + ".log")).write_text(stdout + stderr)
            if process.returncode:
                raise RuntimeError(
                    f"{label}: {name} capture exited {process.returncode}"
                )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=3)
    frames = json.loads((folder / "robot-frames.json").read_text())
    sample = frames["joint_aligned_sample"]
    if sample is None:
        raise RuntimeError("No common robot/arm feedback stamp")
    row = dict(stamp=sample["stamp"])
    for side in goals:
        angles = native_angles(sample, side, convert)
        row[side + "_joints_sdk"] = angles
        row[side + "_target_error_rad"] = joint_error(angles, goals[side])
    from rabokit_bricks import detect, saturated_color
    from hand_competition.closed_loop.artifacts import read_ppm

    row["green_rgb"] = {}
    for camera in ("base_cam", "base_cam2", "base_cam3"):
        width, height, pixels = read_ppm(folder / "camera" / camera / "frame.ppm")
        found = detect(width, height, pixels)
        row["green_rgb"][camera] = dict(
            saturated_pixel_count=sum(
                saturated_color(*pixels[i : i + 3]) == (0, 255, 0)
                for i in range(0, len(pixels), 3)
            ),
            components=[
                c
                for c in found["candidates"] + found["rejected_components"]
                if c["color"] == "green"
            ],
        )
    row["stereo_target"] = target_observation(folder)
    info = json.loads((folder / "camera/capture.json").read_text())["camera_messages"][
        "base_cam"
    ]["stamp"]
    rgb_seconds = info["sec"] + info["nanosec"] * 1e-9
    row["rgb_stamp"] = [info["sec"], info["nanosec"]]
    row["right_hand_history"] = hand_history(folder, rgb_seconds)
    history = row["right_hand_history"]
    row["rgb_effort_gap_seconds"] = (
        rgb_seconds - history[-1]["seconds"] if history else None
    )
    row["feedback_alignment_valid"] = bool(
        len(history) >= 5
        and row["rgb_effort_gap_seconds"] <= 0.04
        and history[-1]["seconds"] - history[0]["seconds"] >= 0.16
    )
    return row


def run_contact(robot, root, config, executor, snapshot, convert, report):
    seeds = {
        s: measured(getattr(robot, s + "_arm"))["joints"] for s in ("left", "right")
    }
    report.update(
        sequence_completed=False,
        recovered=True,
        payload_lift_commanded=False,
        observations={},
    )
    paths = config["stages"]
    for name in ("descend", "insert"):
        stage = paths[name]
        if (
            max(abs(a - b) for a, b in zip(seeds[stage["side"]], stage["start_joints"]))
            > 0.02
        ):
            raise RuntimeError(
                "Compiled contact start differs from the measured prepared posture"
            )
    (root / "contact-plan.json").write_text(json.dumps(paths, indent=2) + "\n")
    history = {name: [] for name in paths}
    goals = {s: list(q) for s, q in seeds.items()}
    hand_started = False

    def move_stage(name):
        report["active_step"] = name
        stage = paths[name]
        side = stage["side"]
        arm = getattr(robot, side + "_arm")
        history[name].append(measured(arm)["joints"])
        print("BIMANUAL_CONTACT_STAGE " + name, flush=True)
        executor(arm, [p["joints"] for p in stage["points"]], history[name])
        goals[side] = stage["points"][-1]["joints"]

    def undo(name):
        if not history[name]:
            return
        side = paths[name]["side"]
        arm = getattr(robot, side + "_arm")
        arm.stop()
        if (
            max(abs(a - b) for a, b in zip(measured(arm)["joints"], history[name][-1]))
            > 0.3
        ):
            raise RuntimeError(f"{name}: recovery too far from completed checkpoint")
        executor(arm, list(reversed(history[name])), [])
        goals[side] = history[name][0]

    try:
        if not (
            robot.left_hand.set_force([1.0] * 6)
            and robot.right_hand.set_force([1.0] * 6)
        ):
            raise RuntimeError("Simulation hand force setup rejected")
        report["force_limit_normalized"] = [1.0] * 6
        hand_started = True
        report["recovered"] = False
        report["active_step"] = "preshape"
        report["preshape"] = hand_command(
            "bimanual_preshape", wait_arrival=True, ramp_seconds=2.0
        )
        move_stage("descend")
        report["active_step"] = "close"
        hand_command(
            "bimanual_grasp", ramp_seconds=2.0, trace_output=root / "contact-close.json"
        )
        move_stage("tilt")
        report["active_step"] = "capture_tilted"
        report["observations"]["tilted"] = capture_contact(
            root, "contact_tilted", goals, convert, snapshot
        )
        if any(
            report["observations"]["tilted"][s + "_target_error_rad"] >= 0.01
            for s in goals
        ):
            raise RuntimeError("Tilted arm posture drifted during contact observation")
        move_stage("insert")
        report["active_step"] = "capture_inserted"
        report["observations"]["inserted"] = capture_contact(
            root, "contact_inserted", goals, convert, snapshot
        )
        if any(
            report["observations"]["inserted"][s + "_target_error_rad"] >= 0.01
            for s in goals
        ):
            raise RuntimeError(
                "Inserted arm posture drifted during contact observation"
            )
        report["support_pattern"] = compare_support(
            report["observations"]["tilted"]["right_hand_history"],
            report["observations"]["inserted"]["right_hand_history"],
        )
        if not all(
            r["feedback_alignment_valid"] for r in report["observations"].values()
        ):
            report["support_pattern"] = dict(
                ready=False, reason="incomplete_rgb_effort_alignment"
            )
        report["sequence_completed"] = True
        report.pop("active_step", None)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["failed_step"] = report.get("active_step")
        report["completed_points"] = {n: max(0, len(h) - 1) for n, h in history.items()}
        failed_goals = dict(goals)
        name = report["failed_step"]
        if name in paths:
            stage = paths[name]
            index = min(report["completed_points"][name], len(stage["points"]) - 1)
            report["failed_target_joints"] = stage["points"][index]["joints"]
            failed_goals[stage["side"]] = report["failed_target_joints"]
        try:
            report["failed_sdk"] = {
                s: measured(getattr(robot, s + "_arm")) for s in goals
            }
            robot.stop()
            report["failed_native"] = snapshot(
                root, "contact_failed", failed_goals, convert
            )
        except Exception as diagnostic_error:
            report["diagnostic_error"] = str(diagnostic_error)
        raise
    finally:
        # Retract the supporting palm while the thumb/index keep the low tilt,
        # undo the tilt, reopen, and only then raise the empty left hand.
        try:
            undo("insert")
            undo("tilt")
            if hand_started:
                hand_command("bimanual_preshape", ramp_seconds=2.0)
            undo("descend")
            if hand_started:
                report["neutral"] = hand_command(
                    "neutral", wait_arrival=True, ramp_seconds=2.0
                )
            report["recovery_snapshot"] = snapshot(
                root, "contact_recovered", seeds, convert
            )
            if any(
                report["recovery_snapshot"][s + "_target_error_rad"] >= 0.01
                for s in seeds
            ):
                raise RuntimeError("Contact recovery arm position unverified")
            report["recovered"] = True
        except Exception as error:
            report["recovery_error"] = f"{type(error).__name__}: {error}"
            raise
