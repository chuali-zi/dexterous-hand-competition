"""Compare local left-hand grasp structures by actual lift and hold, never ROS.

Uses a retained model without changing mass, friction or actuators. The idle
right hand is parked 0.5 m higher to avoid its table-exterior neutral posture
intersecting the model's infinite table. No whole-arm collision is represented.
Object state is an offline outcome measurement, never a native policy input.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rabokit_hand import HAND_JOINTS
from replay_hand_dynamics import cube_bottom_z


def outcome(rows):
    """Research thresholds for a 40 mm lift followed by a three-second hold."""
    before = min(rows, key=lambda r: abs(r["seconds"] - 6.5))
    held = [r for r in rows if r["seconds"] >= 8.5 - 1e-8]
    start = held[0]
    relative = np.array([r["target_in_hand_m"] for r in held])
    drift = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
    rotations = Rotation.from_quat(
        np.array([r["target_in_hand_quaternion_wxyz"] for r in held])[:, [1, 2, 3, 0]]
    )
    angle = float(np.max((rotations[0].inv() * rotations).magnitude()))
    table = any(any("table" in p for p in r["target_contacts"]) for r in held)
    neighbors = any(
        any(any(n.startswith("neighbor_") for n in p) for p in r["target_contacts"])
        for r in held
    )
    hand_fraction = sum(
        any(any(n.startswith("lh_") for n in p) for p in r["target_contacts"])
        for r in held
    ) / len(held)
    bottom = min(r["bottom_m"] for r in held)
    hand_rise = start["left_root_position"][2] - before["left_root_position"][2]
    target_rise = start["target_position"][2] - before["target_position"][2]
    sustained = held[-1]["seconds"] - held[0]["seconds"] >= 2.98
    passed = (
        sustained
        and bottom >= 0.01
        and hand_rise >= 0.035
        and target_rise >= 0.03
        and drift <= 0.005
        and angle <= np.deg2rad(10)
        and not table
        and not neighbors
        and hand_fraction >= 0.95
    )
    return dict(
        local_lift_hold_pass=bool(passed),
        minimum_hold_clearance_mm=1000 * bottom,
        actual_hand_rise_mm=1000 * hand_rise,
        actual_target_rise_mm=1000 * target_rise,
        target_relative_drift_mm=1000 * drift,
        target_relative_rotation_deg=float(np.rad2deg(angle)),
        hold_duration_s=held[-1]["seconds"] - held[0]["seconds"],
        hold_table_contact=table,
        hold_neighbor_contact=neighbors,
        hold_hand_contact_fraction=hand_fraction,
        held_contact_parts=sorted(
            {n for r in held for p in r["target_contacts"] for n in p if n != "target"}
        ),
        native_grasp_verified=False,
    )


def replay(model_path, candidate):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    goal = np.asarray(candidate["hand_root_position"], float)
    quat = np.asarray(candidate["hand_root_quaternion_wxyz"], float)
    offset = np.asarray(candidate.get("approach_offset_m", [0, 0, 0.08]), float)
    if (
        goal.shape != (3,)
        or quat.shape != (4,)
        or offset.shape != (3,)
        or not all(np.isfinite(a).all() for a in (goal, quat, offset))
        or abs(np.linalg.norm(quat) - 1) > 1e-6
    ):
        raise ValueError(
            "Expected finite root pose, unit quaternion and approach offset"
        )
    pre, closed = candidate["preshape_targets"], candidate["closure_targets"]
    for targets in (pre, closed):
        if set(targets) != set(HAND_JOINTS):
            raise ValueError("Expected all eleven left-hand joint targets")
        for name, value in targets.items():
            low, high = model.jnt_range[model.joint(name).id]
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid joint target: {name}")
    la = model.jnt_qposadr[model.joint("hand_root_free").id]
    ra = model.jnt_qposadr[model.joint("right_hand_root_free").id]
    ta = model.jnt_qposadr[model.joint("target_free").id]
    lm = model.body_mocapid[model.body("root_reference").id]
    rm = model.body_mocapid[model.body("right_root_reference").id]
    qa = {n: model.jnt_qposadr[model.joint(n).id] for n in pre}
    ai = {n: model.actuator(n).id for n in pre}
    data.qpos[la : la + 3] = goal + offset
    data.qpos[la + 3 : la + 7] = quat
    data.mocap_pos[lm] = goal + offset
    data.mocap_quat[lm] = quat
    data.qpos[ra + 2] += 0.5
    data.mocap_pos[rm] = data.qpos[ra : ra + 3]
    data.mocap_quat[rm] = data.qpos[ra + 3 : ra + 7]
    for n, v in pre.items():
        data.qpos[qa[n]] = v
        data.ctrl[ai[n]] = v
    neighbors = {
        model.geom(g).name: model.jnt_qposadr[
            model.joint(model.geom(g).name + "_free").id
        ]
        for g in range(model.ngeom)
        if model.geom(g).name.startswith("neighbor_")
    }
    original = {n: data.qpos[a : a + 3].copy() for n, a in neighbors.items()}
    maximum_neighbor_shift = {n: 0.0 for n in neighbors}
    mujoco.mj_forward(model, data)
    initial_contacts = [
        dict(pair=[model.geom(int(g)).name for g in c.geom], distance_m=float(c.dist))
        for c in data.contact
        if any(model.geom(int(g)).name.startswith("lh_") for g in c.geom)
    ]
    # A preshape already overlapping anything is not a valid free-space start.
    if any(c["distance_m"] < -0.0005 for c in initial_contacts):
        return dict(
            candidate=candidate,
            summary=dict(
                local_lift_hold_pass=False,
                reason="preshape_initial_collision",
                initial_contacts=initial_contacts,
                native_grasp_verified=False,
            ),
            rows=[],
        )
    rows, unexpected = [], {}
    floor = float(model.geom("table").pos[2])
    start = time.monotonic()
    while data.time < 11.5:
        t = float(data.time)
        approach = float(np.clip((t - 0.5) / 3, 0, 1))
        close = float(np.clip((t - 3.5) / 2, 0, 1))
        lift = 0.04 * float(np.clip((t - 6.5) / 2, 0, 1))
        data.mocap_pos[lm] = goal + (1 - approach) * offset + [0, 0, lift]
        for n in pre:
            data.ctrl[ai[n]] = pre[n] + close * (closed[n] - pre[n])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Nonfinite model state")
        for c in data.contact:
            pair = tuple(model.geom(int(g)).name for g in c.geom)
            if "target" not in pair and any(n.startswith("lh_") for n in pair):
                unexpected[pair] = min(unexpected.get(pair, 0.0), float(c.dist))
        for n, a in neighbors.items():
            maximum_neighbor_shift[n] = max(
                maximum_neighbor_shift[n],
                float(np.linalg.norm(data.qpos[a : a + 2] - original[n][:2])),
            )
        if data.time + 1e-9 >= len(rows) * 0.04 or data.time >= 11.5:
            target_pos, target_quat = data.qpos[ta : ta + 3], data.qpos[ta + 3 : ta + 7]
            root_pos, root_quat = data.qpos[la : la + 3], data.qpos[la + 3 : la + 7]
            root_rot = Rotation.from_quat(root_quat[[1, 2, 3, 0]])
            rel_rot = root_rot.inv() * Rotation.from_quat(target_quat[[1, 2, 3, 0]])
            contacts, forces = [], []
            for index, c in enumerate(data.contact):
                pair = [model.geom(int(g)).name for g in c.geom]
                if "target" not in pair:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                world_force = c.frame.reshape(3, 3).T @ force[:3]
                world_force *= 1 if pair[1] == "target" else -1
                contacts.append(pair)
                forces.append(dict(pair=pair, force_on_target_n=world_force.tolist()))
            rows.append(
                dict(
                    seconds=float(data.time),
                    target_position=target_pos.tolist(),
                    target_quaternion=target_quat.tolist(),
                    left_root_position=root_pos.tolist(),
                    left_root_quaternion=root_quat.tolist(),
                    commanded_root_position=data.mocap_pos[lm].tolist(),
                    target_in_hand_m=root_rot.inv()
                    .apply(target_pos - root_pos)
                    .tolist(),
                    target_in_hand_quaternion_wxyz=rel_rot.as_quat()[
                        [3, 0, 1, 2]
                    ].tolist(),
                    bottom_m=cube_bottom_z(target_pos, target_quat) - floor,
                    target_contacts=contacts,
                    target_forces=forces,
                    q={n: float(data.qpos[a]) for n, a in qa.items()},
                    effort={n: float(data.actuator_force[a]) for n, a in ai.items()},
                )
            )
    summary = outcome(rows)
    summary.update(
        wall_seconds=time.monotonic() - start,
        neighbor_maximum_horizontal_shift_mm={
            n: 1000 * v for n, v in maximum_neighbor_shift.items()
        },
        unintended_hand_contacts=[
            dict(pair=p, minimum_distance_m=v) for p, v in unexpected.items()
        ],
        initial_contacts=initial_contacts,
        loaded_model_parameters_changed=False,
        idle_right_root_raise_m=0.5,
    )
    return dict(
        candidate=candidate,
        summary=summary,
        rows=rows,
        feedback_timing="Positions are post-step; contact/actuator forces are cached from the beginning of the last 2 ms step.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    result = replay(args.model, json.loads(args.candidate.read_text(encoding="utf-8")))
    result.update(
        scope=__doc__, source_model=str(args.model), candidate_file=str(args.candidate)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
