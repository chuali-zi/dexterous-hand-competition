#!/usr/bin/env python3
"""Replay a local two-hand tilt/support/carry/place hypothesis; never ROS/SDK.

Initial cubes come from retained RGB. Supplied support poses are offline design
inputs. Object states and solver forces below are diagnostic outputs, not policy
inputs or native success labels. Whole-arm dynamics/collision are not included.
The default is timed; --support-feedback adds an experimental pre-lift decision
from actuator history, not a trained multimodal policy or a grasp-success label.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from local_bimanual_proxy import build_bimanual_model
from replay_hand_dynamics import read, wxyz, cube_bottom_z
from rabokit_hand import HAND_JOINTS, bimanual_targets
from bimanual_support_signal import support_evidence


def replay(
    trial,
    left_assets,
    right_assets,
    right_config,
    gap=0.002,
    engage=0.012,
    release=False,
    support_feedback=False,
):
    import mujoco

    m, xml, rvalues, mapping = build_bimanual_model(trial, left_assets, right_assets)
    d0 = mujoco.MjData(m)
    la = int(m.jnt_qposadr[m.joint("hand_root_free").id])
    ra = int(m.jnt_qposadr[m.joint("right_hand_root_free").id])
    ta = int(m.jnt_qposadr[m.joint("target_free").id])
    lm = int(m.body_mocapid[m.body("root_reference").id])
    rm = int(m.body_mocapid[m.body("right_root_reference").id])
    floor = float(m.geom("table").pos[2])
    lbp = d0.qpos[la : la + 3].copy() + [0, 0.007, -0.005]
    lbr = Rotation.from_quat(d0.qpos[la + 3 : la + 7][[1, 2, 3, 0]]).as_matrix()
    center = d0.qpos[ta : ta + 3].copy()
    pre, goal = bimanual_targets()

    pivot = center + [0.025, 0, -0.025]
    name = "local-bimanual-cycle" if release else "local-bimanual-hold"
    rp = np.array(right_config["right_root_goal"]) - [0, 0, gap - 0.002]
    rquat = np.array(right_config["right_root_quaternion"])
    direction = np.array(
        right_config.get(
            "approach_direction_sdk",
            Rotation.from_euler("z", right_config["yaw_degrees"], degrees=True).apply(
                [1, 0, 0]
            ),
        )
    )
    qa = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in (*HAND_JOINTS, *rvalues)}
    ai = {n: m.actuator(n).id for n in qa}
    neighbors = {
        m.geom(g).name: int(m.jnt_qposadr[m.joint(m.geom(g).name + "_free").id])
        for g in range(m.ngeom)
        if m.geom(g).name.startswith("neighbor_")
    }
    d = mujoco.MjData(m)
    d.qpos[la : la + 3] = lbp + [0, 0, 0.08]
    d.qpos[la + 3 : la + 7] = wxyz(lbr)
    d.mocap_pos[lm] = d.qpos[la : la + 3]
    d.mocap_quat[lm] = wxyz(lbr)
    d.qpos[ra : ra + 3] = rp - 0.1 * direction
    d.qpos[ra + 3 : ra + 7] = rquat
    d.mocap_pos[rm] = d.qpos[ra : ra + 3]
    d.mocap_quat[rm] = rquat
    for n, q in {**pre, **rvalues}.items():
        d.qpos[qa[n]] = q
        d.ctrl[ai[n]] = q
    mujoco.mj_forward(m, d)
    initial_neighbors = {n: d.qpos[a : a + 3].copy() for n, a in neighbors.items()}
    rows = []
    other = {}
    next_sample = 0
    started = time.monotonic()
    duration = 29 if release else 19
    decision = None
    engagement_stop = None

    def evidence_at(stamp):
        # Pass only measured effort history, without object/contact outputs.
        observations = [
            dict(seconds=r["seconds"], servo_effort=r["servo_effort"]) for r in rows
        ]
        before = [r for r in observations if 6.1 <= r["seconds"] < 6.5]
        recent = [r for r in observations if stamp - 0.2 <= r["seconds"] < stamp]
        return support_evidence(before, recent)

    while d.time < duration:
        t = d.time
        if support_feedback and engagement_stop is None and 8.5 <= t < 9.5:
            signal = evidence_at(t)
            if signal["ready"]:
                engagement_stop = dict(
                    seconds=float(t),
                    engagement_m=engage * float(np.clip(t - 8.5, 0, 1)),
                    **signal,
                )
        if support_feedback and decision is None and t >= 9.5:
            decision = dict(seconds=float(t), **evidence_at(t))
        action_time = (
            min(t, 9.5) if decision is not None and not decision["ready"] else t
        )
        down = 0.08 * (1 - float(np.clip((action_time - 0.5) / 3, 0, 1)))
        close = float(np.clip((action_time - 3.5) / 2, 0, 1))
        angle = 0.6 * float(np.clip((action_time - 5.5) / 1, 0, 1))
        lr = Rotation.from_euler("y", angle).as_matrix()
        insert = float(np.clip((action_time - 6.5) / 2, 0, 1))
        contact = engage * float(np.clip((action_time - 8.5) / 1, 0, 1))
        if engagement_stop is not None:
            contact = min(contact, engagement_stop["engagement_m"])
        lift = 0.08 * float(np.clip((action_time - 9.5) / 2, 0, 1))
        side = -0.04 * float(np.clip((action_time - 14) / 2, 0, 1))
        # Preserve the right hand's previous release clearance after reducing
        # its upward engagement; repeating the full descent would hit the table.
        descent = 0.075 - (
            engage - engagement_stop["engagement_m"] if engagement_stop else 0.0
        )
        lowering = (
            descent * float(np.clip((action_time - 19) / 2, 0, 1)) if release else 0.0
        )
        opening = float(np.clip((action_time - 21) / 1, 0, 1)) if release else 0.0
        withdraw = float(np.clip((action_time - 22) / 2, 0, 1)) if release else 0.0
        settle = (
            0.015 * float(np.clip((action_time - 22) / 1, 0, 1)) if release else 0.0
        )
        clear = 0.06 * float(np.clip((action_time - 24) / 1, 0, 1)) if release else 0.0
        d.mocap_pos[lm] = (
            lr @ (lbp - pivot) + pivot + [0, side, down + lift - lowering + clear]
        )
        d.mocap_quat[lm] = wxyz(lr @ lbr)
        d.mocap_pos[rm] = (
            rp
            - 0.1 * (1 - insert + withdraw) * direction
            + [0, side, contact + lift - lowering - settle + clear]
        )
        d.mocap_quat[rm] = rquat
        for n in HAND_JOINTS:
            d.ctrl[ai[n]] = pre[n] + close * (1 - opening) * (goal[n] - pre[n])
        mujoco.mj_step(m, d)
        for c in d.contact:
            pair = tuple(m.geom(int(g)).name for g in c.geom)
            if "target" not in pair and any(n.startswith(("lh_", "rh_")) for n in pair):
                other[pair] = min(other.get(pair, 0), float(c.dist))
        if t >= next_sample:
            contacts = []
            forces = []
            for i, c in enumerate(d.contact):
                pair = [m.geom(int(g)).name for g in c.geom]
                if "target" not in pair:
                    continue
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                on = (
                    c.frame.reshape(3, 3).T @ f[:3] * (1 if pair[1] == "target" else -1)
                )
                contacts.append(pair)
                forces.append(dict(pair=pair, force_on_target=on.tolist()))
            rows.append(
                dict(
                    seconds=float(t),
                    bottom_m=cube_bottom_z(d.qpos[ta : ta + 3], d.qpos[ta + 3 : ta + 7])
                    - floor,
                    target_position=d.qpos[ta : ta + 3].tolist(),
                    target_quaternion=d.qpos[ta + 3 : ta + 7].tolist(),
                    left_root_position=d.qpos[la : la + 3].tolist(),
                    left_root_quaternion=d.qpos[la + 3 : la + 7].tolist(),
                    right_root_position=d.qpos[ra : ra + 3].tolist(),
                    right_root_quaternion=d.qpos[ra + 3 : ra + 7].tolist(),
                    q={n: float(d.qpos[a]) for n, a in qa.items()},
                    velocity={
                        n: float(d.qvel[m.jnt_dofadr[m.joint(n).id]]) for n in qa
                    },
                    servo_effort={n: float(d.actuator_force[ai[n]]) for n in qa},
                    command={n: float(d.ctrl[ai[n]]) for n in qa},
                    contacts=contacts,
                    forces=forces,
                    neighbor_positions={
                        n: d.qpos[a : a + 3].tolist() for n, a in neighbors.items()
                    },
                )
            )
            next_sample = t + 0.04
    late = [v for v in rows if v["seconds"] > duration - 1]
    summary = dict(
        name=name,
        release=release,
        gap_m=gap,
        engagement_m=engage,
        support_feedback=support_feedback,
        support_decision=decision,
        engagement_stop=engagement_stop,
        right_root_goal_sdk_m=rp.tolist(),
        right_root_quaternion_wxyz=rquat.tolist(),
        right_config=right_config,
        minimum_late_bottom_mm=1000 * min(v["bottom_m"] for v in late),
        last1_table_fraction=sum(any("table" in p for p in v["contacts"]) for v in late)
        / len(late),
        last1_neighbor_fraction=sum(
            any(any(n.startswith("neighbor_") for n in p) for p in v["contacts"])
            for v in late
        )
        / len(late),
        target_y_move_mm=1000
        * (rows[-1]["target_position"][1] - rows[0]["target_position"][1]),
        other_contacts=[dict(pair=p, gap_m=v) for p, v in other.items()],
        neighbor_horizontal_displacement_mm={
            n: 1000
            * float(np.linalg.norm(d.qpos[a : a + 2] - initial_neighbors[n][:2]))
            for n, a in neighbors.items()
        },
        wall_seconds=time.monotonic() - started,
    )

    hold = [v for v in rows if 16 <= v["seconds"] <= 19]
    carry = [v for v in rows if 11.5 <= v["seconds"] <= 19]
    final = [v for v in rows if duration - 3 <= v["seconds"]]
    summary.update(
        minimum_carry_bottom_mm=1000 * min(v["bottom_m"] for v in carry),
        carry_table_fraction=sum(
            any("table" in p for p in v["contacts"]) for v in carry
        )
        / len(carry),
        carry_neighbor_fraction=sum(
            any(any(n.startswith("neighbor_") for n in p) for p in v["contacts"])
            for v in carry
        )
        / len(carry),
        hold_drift_mm=1000
        * float(
            np.linalg.norm(
                np.array(hold[-1]["target_position"]) - hold[0]["target_position"]
            )
        ),
        final_drift_mm=1000
        * float(
            np.linalg.norm(
                np.array(final[-1]["target_position"]) - final[0]["target_position"]
            )
        ),
        final_contacts=rows[-1]["contacts"],
        self_or_interhand_contact=any(
            all(n.startswith(("lh_", "rh_")) for n in p) for p in other
        ),
    )
    return (
        dict(
            scope=__doc__,
            summary=summary,
            right_mapping=mapping,
            feedback_timing="Position/velocity are post-step; actuator force and contacts are cached from the beginning of the last 2 ms step. Command is the last applied model control.",
            rows=rows,
        ),
        xml,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial", "left-assets", "right-assets", "support-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--support-feedback",
        action="store_true",
        help="Stop upward engagement on the experimental support pattern and require it before lifting; otherwise hold at the low pose",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Lower, open and withdraw both hands after the carry",
    )
    parser.add_argument(
        "--gap", type=float, default=0.002, help="Initial pad gap, metres"
    )
    parser.add_argument(
        "--engage",
        type=float,
        default=0.012,
        help="Right support vertical engagement, metres",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".xml").exists():
        parser.error("Use a new output path")
    result, xml = replay(
        args.trial,
        args.left_assets,
        args.right_assets,
        read(args.support_config),
        args.gap,
        args.engage,
        args.release,
        args.support_feedback,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, allow_nan=False) + "\n")
    with args.output.with_suffix(".xml").open("x", encoding="utf-8") as stream:
        stream.write(xml)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
