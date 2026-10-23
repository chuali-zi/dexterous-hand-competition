"""Empty bimanual preparation and reverse return, fixed simulation mode only."""

import argparse
import base64
import gzip
import json
import math
import os
from pathlib import Path
import time

from rabokit_calibration_probe import measured
from rabokit_hover import move_joint_pose
from rabokit_right_empty_probe import check_plan, execute, joint_error, snapshot


def export_report(folder):
    """Export completed saved data before stopping the simulation resets terminals."""
    exit_file = Path(str(folder) + ".exit")
    record = dict(
        report=json.loads((folder / "report.json").read_text()),
        exit=exit_file.read_text().strip() if exit_file.exists() else None,
    )
    payload = base64.b64encode(
        gzip.compress(json.dumps(record, allow_nan=False).encode())
    ).decode()
    print("BIMANUAL_EMPTY_EXPORT " + payload, flush=True)


def load_plan(path):
    record = json.loads(path.read_text())
    right = json.loads((path.parent / record["right_plan"]).read_text())
    profile = json.loads((path.parent / record["support_profile"]).read_text())
    right["points"] += profile["entry_from_native59_preinsert"]["joint_waypoints_sdk"]
    plans = dict(right=right, left=record["left"])
    for plan in plans.values():
        check_plan(plan)
    return plans


def dual_snapshot(root, label, goals, convert, rgb=False):
    row = snapshot(root, label, goals["right"], goals["left"], convert, rgb)
    row["left_target_error_rad"] = row.pop("left_drift_rad")
    frames = json.loads((root / label / "robot-frames.json").read_text())
    for side, prefix in [("left", "lh_"), ("right", "rh_")]:
        row[side + "_fingers"] = {
            n: v
            for n, v in frames["joint_aligned_sample"]["arms"]["arm_" + side][
                "position"
            ].items()
            if n.startswith(prefix)
        }
    if label == "both_ready":
        channels = ("rh_ring_mcp_pitch", "rh_ring_dip")
        samples = frames["arm_feedback_samples"]["arm_right"]
        last = max(s["stamp"][0] + s["stamp"][1] * 1e-9 for s in samples)
        row["right_unloaded_history"] = [
            dict(
                stamp=s["stamp"],
                position={n: s["position"].get(n) for n in channels},
                effort={n: s["effort"].get(n) for n in channels},
            )
            for s in samples
            if s["stamp"][0] + s["stamp"][1] * 1e-9 >= last - 0.5
        ]
    return row


def arrived(row):
    return all(row[side + "_target_error_rad"] < 0.01 for side in ("left", "right"))


def open_hands(row):
    return all(
        row[side + "_fingers"]
        and all(
            math.isfinite(v) and abs(v) < 0.03 for v in row[side + "_fingers"].values()
        )
        for side in ("left", "right")
    )


def execute_tracked(arm, points, history, forward, report, side, budget=120):
    """Reuse the previously native-tested tool/joint tracking, recording failures."""
    deadline = time.monotonic() + budget
    for index, goal in enumerate(points):
        report["active_motion"] = dict(side=side, point=index + 1, target_joints=goal)
        if time.monotonic() > deadline:
            raise TimeoutError("Route time budget exceeded")
        state = move_joint_pose(arm, goal, forward, require_settled=False)
        history.append(list(goal))
        if state["tracking_corrections"]:
            report.setdefault("tracking_corrections", []).append(
                dict(
                    **report["active_motion"], corrections=state["tracking_corrections"]
                )
            )
        if index % 10 == 0 or index == len(points) - 1:
            print("BIMANUAL_EMPTY_POINT", side, index + 1, "/", len(points), flush=True)
    report.pop("active_motion", None)


def return_completed(robot, history, executor=None):
    # The left hand entered after the right. If its return fails, hold both arms
    # instead of moving the right through an unverified left-arm posture.
    for side in ("left", "right"):
        if not history[side]:
            continue
        arm = getattr(robot, side + "_arm")
        arm.stop()
        if joint_error(measured(arm)["joints"], history[side][-1]) > 0.3:
            raise RuntimeError(f"{side}: too far from last completed checkpoint")
        (executor or execute)(arm, list(reversed(history[side])), [], budget=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/bimanual-empty-approach.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--contact-plan",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/bimanual-contact.json",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--motion", action="store_true")
    action.add_argument(
        "--contact",
        action="store_true",
        help="Execute the compiled low contact diagnostic and recover; no payload lift",
    )
    action.add_argument(
        "--export",
        action="store_true",
        help="Print saved report as compact data; no SDK/ROS initialization",
    )
    args = parser.parse_args()
    if args.export:
        if args.output is None:
            parser.error("--export requires --output of an existing saved run")
        export_report(args.output)
        return
    plans = load_plan(args.plan)
    contact_config = json.loads(args.contact_plan.read_text())
    for stage in contact_config["stages"].values():
        check_plan(
            dict(
                neutral=stage["start_joints"],
                points=[p["joints"] for p in stage["points"]],
                joint_limits_sdk=contact_config["joint_limits_sdk"][stage["side"]],
            )
        )
    if not (args.motion or args.contact):
        print(
            json.dumps(
                dict(
                    motion_commanded=False,
                    contact_stages={
                        n: len(s["points"]) for n, s in contact_config["stages"].items()
                    },
                    sides={
                        s: dict(
                            points=len(p["points"]),
                            maximum_joint_step_rad=check_plan(p),
                        )
                        for s, p in plans.items()
                    },
                )
            )
        )
        return
    if not os.getenv("ROS2_NAMESPACE") or args.output is None or args.output.exists():
        parser.error(
            "Motion requires the simulation terminal and a new output directory"
        )
    from rabo_robolib import LbotDualArm
    from _robolib_core.arms import _lbot_kinematics as k

    args.output.mkdir(parents=True)
    robot = LbotDualArm(mode="sim")
    history = dict(left=[], right=[])
    report = dict(
        target_reached=False,
        returned_verified=False,
        errors=[],
        physical_grasp_verified=False,
        whole_arm_collision_verified=False,
        official_score=None,
    )
    started = time.monotonic()
    start = None
    entry_return_allowed = True
    report["mode"] = "low_contact" if args.contact else "empty_preparation"

    def executor(arm, points, completed, budget=120):
        side = "left" if arm is robot.left_arm else "right"
        execute_tracked(
            arm,
            points,
            completed,
            lambda q: k.forward_kinematics(side, q),
            report,
            side,
            budget,
        )

    try:
        time.sleep(1)
        start = {s: measured(getattr(robot, s + "_arm"))["joints"] for s in plans}
        if any(joint_error(q, plans[s]["neutral"]) > 0.02 for s, q in start.items()):
            raise RuntimeError("Expected both arms at retained neutral poses")
        goals = {s: list(q) for s, q in start.items()}
        report["start"] = dual_snapshot(
            args.output, "start", goals, k.to_sim_joints, rgb=True
        )
        if not arrived(report["start"]) or not open_hands(report["start"]):
            raise RuntimeError(
                "Native initial feedback disagrees or either hand is not open"
            )
        if args.contact:
            from rabokit_bimanual_contact import check_source, run_contact

            report["source_check"] = check_source(args.output / "start", contact_config)
            if not report["source_check"]["accepted"]:
                raise RuntimeError(
                    "Fresh RGB target differs from the locally checked contact scene"
                )
        for side, label in [("right", "right_ready"), ("left", "both_ready")]:
            arm = getattr(robot, side + "_arm")
            if not arm.enable():
                raise RuntimeError(f"{side} enable failed")
            history[side].append(start[side])
            print("BIMANUAL_EMPTY_STAGE " + side, flush=True)
            executor(arm, plans[side]["points"], history[side], budget=120)
            goals[side] = plans[side]["points"][-1]
            report[label] = dual_snapshot(args.output, label, goals, k.to_sim_joints)
            if not arrived(report[label]) or not open_hands(report[label]):
                raise RuntimeError(
                    f"{label}: native arm target or open fingers unverified"
                )
        report["preparation_reached"] = True
        if args.contact:
            entry_return_allowed = False
            report["contact"] = {}
            try:
                run_contact(
                    robot,
                    args.output,
                    contact_config,
                    executor,
                    dual_snapshot,
                    k.to_sim_joints,
                    report["contact"],
                )
            finally:
                entry_return_allowed = report["contact"].get("recovered", False)
            report["target_reached"] = report["contact"]["sequence_completed"]
        else:
            report["target_reached"] = True
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        if "active_motion" in report:
            report["failed_motion"] = dict(report["active_motion"])
            try:
                report["failed_motion"]["sdk"] = {
                    s: measured(getattr(robot, s + "_arm")) for s in plans
                }
                failed_goals = dict(goals)
                failed_goals[report["active_motion"]["side"]] = report["active_motion"][
                    "target_joints"
                ]
                report["failed_motion"]["native"] = dual_snapshot(
                    args.output, "failed_motion", failed_goals, k.to_sim_joints
                )
            except Exception as diagnostic_error:
                report["failed_motion"]["diagnostic_error"] = str(diagnostic_error)
    finally:
        try:
            if not entry_return_allowed:
                raise RuntimeError(
                    "Contact recovery unverified; holding both arms instead of returning through entry"
                )
            return_completed(robot, history, executor)
            if any(history.values()):
                report["restored"] = dual_snapshot(
                    args.output, "restored", start, k.to_sim_joints, rgb=True
                )
                report["returned_verified"] = arrived(
                    report["restored"]
                ) and open_hands(report["restored"])
                from rabokit_pickup_vision import neighboring_layout, saved_scene

                report["layout"] = neighboring_layout(
                    saved_scene(args.output / "start"),
                    saved_scene(args.output / "restored"),
                )
                if args.contact:
                    report["neighbor_layout"] = neighboring_layout(
                        saved_scene(args.output / "start"),
                        saved_scene(args.output / "restored"),
                        exclude_green=True,
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
            completed_outbound_points={
                s: max(0, len(h) - 1) for s, h in history.items()
            },
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            "BIMANUAL_EMPTY_REPORT " + json.dumps(report, allow_nan=False), flush=True
        )
    if (
        report["errors"]
        or not report["target_reached"]
        or not report["returned_verified"]
        or (not args.contact and not report.get("layout", {}).get("consistent"))
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
