"""Fixed-shape O6 table pushing trial; no grasp, policy fitting, or ROS.

The table deliberately carries the brick. Object state is used only to score
the completed trajectory. This is a local movement test, not competition points.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def command_offset(seconds, withdraw_height=.080, push_sign=-1, push_distance_m=.120, lift_before_retreat=False):
    """One descent, horizontal push, 20 mm retreat, lift, then 3 s observation."""
    if seconds < .5:
        return np.array([-.020 * push_sign, 0., .080]), 'settle'
    if seconds < 3:
        return np.array([-.020 * push_sign, 0., .080 * (3 - seconds) / 2.5]), 'descend'
    if seconds < 4:
        return np.array([-.020 * push_sign * (4 - seconds), 0., 0.]), 'approach'
    if seconds < 8:
        return np.array([push_distance_m * push_sign * (seconds - 4) / 4, 0., 0.]), 'push'
    if lift_before_retreat:
        if seconds < 10:
            return np.array([push_distance_m * push_sign, 0., withdraw_height * (seconds - 8) / 2]), 'withdraw'
        if seconds < 11:
            return np.array([push_sign * (push_distance_m - .020 * (seconds - 10)), 0., withdraw_height]), 'retreat'
        return np.array([(push_distance_m - .020) * push_sign, 0., withdraw_height]), 'released_observation'
    if seconds < 9:
        return np.array([push_sign * (push_distance_m - .020 * (seconds - 8)), 0., 0.]), 'retreat'
    if seconds < 11:
        return np.array([(push_distance_m - .020) * push_sign, 0., withdraw_height * (seconds - 9) / 2]), 'withdraw'
    return np.array([(push_distance_m - .020) * push_sign, 0., withdraw_height]), 'released_observation'


def run(model_path, source, mode, output, withdraw_height=.080, target_body='target', push_sign=-1,
        push_distance_m=.120, lift_before_retreat=False):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    candidate = json.loads(source.read_text())
    goal = np.array(candidate['hand_root_position'])
    if push_sign not in (-1, 1) or (target_body != 'target' and not target_body.startswith('neighbor_')):
        raise ValueError('Expected a retained brick body and push_sign of -1 or +1')
    if not np.isfinite(push_distance_m) or not .020 < push_distance_m <= .120:
        raise ValueError('Expected push distance in (0.020, 0.120] metres')
    if mode in ('center_single', 'center_two'):
        goal[2] -= .020
    quat = np.array(candidate['hand_root_quaternion_wxyz'])
    targets = dict(candidate['hand_targets'])
    if mode == 'center_two':
        targets.update(lh_middle_mcp_pitch=0., lh_middle_dip=0.)
    roots = {side: int(model.jnt_qposadr[model.joint(name).id]) for side, name in
             [('left', 'hand_root_free'), ('right', 'right_hand_root_free')]}
    mocaps = {side: int(model.body_mocapid[model.body(name).id]) for side, name in
              [('left', 'root_reference'), ('right', 'right_root_reference')]}
    la, ra = roots['left'], roots['right']
    initial_offset, _ = command_offset(0., withdraw_height, push_sign, push_distance_m, lift_before_retreat)
    data.qpos[la:la+3] = goal + initial_offset
    data.qpos[la+3:la+7] = quat
    data.mocap_pos[mocaps['left']] = goal + initial_offset
    data.mocap_quat[mocaps['left']] = quat
    # Same documented parking correction as earlier local trials; not a native arm pose.
    data.qpos[ra+2] += .5
    data.mocap_pos[mocaps['right']] = data.qpos[ra:ra+3]
    data.mocap_quat[mocaps['right']] = data.qpos[ra+3:ra+7]
    for name, value in targets.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        data.ctrl[model.actuator(name).id] = value
    mujoco.mj_forward(model, data)
    target = model.body(target_body).id
    target_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == target}
    neighbor_ids = {model.body(i).name: i for i in range(model.nbody)
                    if (model.body(i).name == 'target' or model.body(i).name.startswith('neighbor_')) and i != target}
    initial = data.xpos[target].copy()
    neighbor_start = {name: data.xpos[i].copy() for name, i in neighbor_ids.items()}
    rows, unintended = [], {}
    neighbor_max = {name: 0. for name in neighbor_ids}
    started = time.monotonic()
    for step in range(round(14 / model.opt.timestep) + 1):
        offset, phase = command_offset(float(data.time), withdraw_height, push_sign, push_distance_m, lift_before_retreat)
        data.mocap_pos[mocaps['left']] = goal + offset
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError('Nonfinite simulation state')
        for contact in data.contact:
            pair = tuple(model.geom(int(g)).name for g in contact.geom)
            if any(name.startswith('lh_') for name in pair) and not any(int(g) in target_geoms for g in contact.geom):
                unintended[pair] = min(unintended.get(pair, 0.), float(contact.dist))
        for name, body in neighbor_ids.items():
            neighbor_max[name] = max(neighbor_max[name], float(np.linalg.norm(
                data.xpos[body][:2] - neighbor_start[name][:2])))
        if step % 10 == 0:
            contacts = [[model.geom(int(g)).name for g in c.geom] for c in data.contact
                        if any(int(g) in target_geoms for g in c.geom)]
            rows.append(dict(seconds=float(data.time), phase=phase,
                             command_root_position=data.mocap_pos[mocaps['left']].tolist(),
                             actual_root_position=data.qpos[la:la+3].tolist(),
                             target_position=data.xpos[target].tolist(),
                             target_quaternion_wxyz=data.xquat[target].tolist(),
                             qpos=data.qpos.tolist(),
                             target_contacts=contacts))
    held = [r for r in rows if r['seconds'] >= 11]
    positions = np.array([r['target_position'] for r in held])
    drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    rotations = Rotation.from_quat(np.array([r['target_quaternion_wxyz'] for r in held])[:, [1, 2, 3, 0]])
    angular_drift = float(np.rad2deg((rotations[0].inv() * rotations).magnitude().max()))
    final = np.array(rows[-1]['target_position'])
    hand_clear = all(not any(any(n.startswith(('lh_', 'rh_')) for n in c)
                            for c in r['target_contacts']) for r in held)
    table_supported = all(any('table' in c for c in r['target_contacts']) for r in held)
    displacement = final - initial
    travel_error = abs(push_sign * displacement[0] - push_distance_m)
    max_neighbor = max(neighbor_max.values())
    release_order = 'lift_then_retreat' if lift_before_retreat else 'retreat_then_lift'
    summary = dict(mode=mode, target_body=target_body, push_sign=push_sign, push_distance_m=push_distance_m,
                   release_order=release_order,
                   axial_travel_error_mm=1000*travel_error, maximum_axial_travel_error_mm=10.,
                   local_slide_release_pass=bool(travel_error <= .010 and drift <= .002
                   and angular_drift <= 2 and hand_clear and table_supported and max_neighbor <= .002
                   and held[-1]['seconds'] - held[0]['seconds'] >= 2.98 and not unintended),
                   displacement_mm=(1000 * displacement).tolist(), released_drift_mm=1000*drift,
                   released_rotation_drift_deg=angular_drift, released_duration_s=held[-1]['seconds']-held[0]['seconds'],
                   released_hand_contact_absent=hand_clear, released_table_contact_present=table_supported,
                   maximum_neighbor_xy_shift_mm=1000*max_neighbor,
                   neighbor_xy_shift_mm={name: 1000 * value for name, value in neighbor_max.items()},
                   unintended_hand_contacts=[dict(pair=list(pair), overlap_mm=-1000*value) for pair, value in unintended.items()],
                   final_target_position=final.tolist(), final_target_quaternion_wxyz=rows[-1]['target_quaternion_wxyz'],
                   wall_seconds=time.monotonic()-started, simulation_seconds=float(data.time),
                   physics_parameters_changed=False, native_execution_verified=False, competition_score_verified=False)
    release_schedule = (f'2 s lift {1000*withdraw_height:g} mm; 1 s retreat {-20*push_sign:+g} mm X' if lift_before_retreat else
                        f'1 s retreat {-20*push_sign:+g} mm X; 2 s lift {1000*withdraw_height:g} mm')
    result = dict(summary=summary, candidate=dict(hand_root_position=goal.tolist(), hand_root_quaternion_wxyz=quat.tolist(),
                  hand_targets=targets, idle_right_raise_m=.5, source=str(source), withdraw_height_m=withdraw_height,
                  target_body=target_body, push_sign=push_sign, push_distance_m=push_distance_m, release_order=release_order,
                  schedule=f'0.5 s settle; 2.5 s descend; 1 s approach; 4 s push {1000*push_distance_m*push_sign:+g} mm X; {release_schedule}; 3 s released observation'),
                  limitations=['Fixed-root reference trajectory, full arm path not included.',
                               'Retained approximate model and infinite table; finite table and destination region not checked.',
                               'No object true state used to choose commands; measurements are offline evaluation only.'], rows=rows)
    if 'contact_joints_sdk' in candidate:
        result['candidate']['contact_joints_sdk'] = candidate['contact_joints_sdk']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('runs/bimanual-contact-68/local-calibrated-contact.xml'))
    parser.add_argument('--source', type=Path, default=Path('runs/grasp-access-72/pusher/candidate.json'))
    parser.add_argument('--mode', choices=['high_single', 'center_single', 'center_two', 'source'], required=True,
                        help='source preserves the supplied root height and hand targets exactly')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--withdraw-height', type=float, default=.080, help='Post-push withdrawal rise in metres; approach rise stays 80 mm')
    parser.add_argument('--target-body', default='target', help='Existing brick body to score; all other bricks stay as neighbors')
    parser.add_argument('--push-sign', type=int, choices=[-1, 1], default=-1)
    parser.add_argument('--push-distance', type=float, default=.120, help='Push distance in metres; scored within 10 mm axial travel error')
    parser.add_argument('--lift-before-retreat', action='store_true', help='Withdraw vertically during 8-10 s, then retreat horizontally during 10-11 s')
    args = parser.parse_args()
    sys.path.insert(0, str(Path('runs/mujoco-runtime').resolve()))
    if not 0 < args.withdraw_height <= .080:
        parser.error('--withdraw-height must be in (0, 0.080] metres')
    summary = run(args.model, args.source, args.mode, args.output, args.withdraw_height, args.target_body, args.push_sign,
                  args.push_distance, args.lift_before_retreat)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['local_slide_release_pass'] else 1)
