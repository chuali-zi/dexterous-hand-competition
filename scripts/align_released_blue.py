"""Local, fixed side-face correction after actual blue release on two yellows."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import try_blue_upper_release as blue

g = blue.g


def configure(source):
    record = json.loads((source / 'result.json').read_text())
    summary = record['summary']
    target = summary['target']
    assessment = blue.seven.assess(source, placed_payload=target)
    if (not assessment['local_seven_brick_final_structure_verified']
            or not summary.get('local_released_bridge_verified', False)
            or not summary.get('held_only_base_contacts', False)
            or summary['held_duration_s'] < 2.98 or summary['held_drift_mm'] > 2
            or summary['held_angular_drift_degrees'] > 2
            or summary['final_cube_face_tilt_degrees'] > 2):
        raise ValueError('Alignment requires an actual stable released blue preserving seven bricks')
    state = np.load(source / 'integration-state.npz', allow_pickle=False)
    check = json.loads((source / 'precheck.json').read_text())
    plan = json.loads((source / 'plan.json').read_text())
    def restore():
        g.mj.mj_setState(g.m, g.d, state['state'], int(state['spec']))
        g.mj.mj_forward(g.m, g.d)
    g.SOURCE, g.RECORD, g.STATE, g.CHECK = source, record, state, check
    g.TARGET, g.SUPPORTS, g.CONTINUOUS_SOURCE = target, plan['supports'], True
    g.restore = restore
    restore()


def prepare(axis, finger_height=.120):
    if not np.isfinite(finger_height) or not .105 <= finger_height <= .135:
        raise ValueError('Pusher fingertip height must be 105 to 135 mm above the table')
    g.restore()
    starts = {side: (g.d.mocap_pos[i].copy(), g.d.mocap_quat[i].copy())
              for side, i in [('left', g.p.t.LM), ('right', g.p.t.RM)]}
    controls = {g.m.actuator(i).name: float(g.d.ctrl[i]) for i in range(g.m.nu)}
    goals = g.p.t.CANDIDATE['hand_targets']
    target = g.d.xpos[g.m.body(g.TARGET).id].copy()
    rotation = Rotation.from_quat(g.d.xquat[g.m.body(g.TARGET).id][[1, 2, 3, 0]])
    corners = rotation.apply(g.p.CORNERS) + target
    pair = np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS], axis=0)
    component = 0 if axis == 'x' else 1
    error = target[component] - pair[component]
    distance = abs(float(error))
    if not .001 <= distance <= .025:
        raise ValueError('Alignment requires an axis error between 1 and 25 mm')
    normal = np.zeros(3)
    normal[component] = np.sign(error)
    tangent = np.array([-normal[1], normal[0], 0.])
    hand_rotation = Rotation.from_euler('z', 90 if axis == 'y' else 0, degrees=True) * Rotation.from_quat(g.p.t.QUAT[[1, 2, 3, 0]])
    quat = hand_rotation.as_quat()[[3, 0, 1, 2]]
    for name, value in goals.items():
        g.d.qpos[g.m.jnt_qposadr[g.m.joint(name).id]] = value
    g.d.qpos[g.p.t.LA:g.p.t.LA + 3] = 0
    g.d.qpos[g.p.t.LA + 3:g.p.t.LA + 7] = quat
    g.mj.mj_forward(g.m, g.d)
    vertices = np.concatenate([g.a.g.mesh_vertices('lh_' + name)
        for name in ('index_distal', 'index_proximal', 'middle_distal', 'middle_proximal')])
    root = normal * ((corners @ normal).max() - (vertices @ normal).min() - .0005)
    root += tangent * (target @ tangent - ((vertices @ tangent).min() + (vertices @ tangent).max()) / 2)
    root[2] = g.a.g.FLOOR + finger_height - vertices[:, 2].min()
    start, startq = starts['left']
    turn = Slerp([2., 5.], Rotation.from_quat(np.array([startq, quat])[:, [1, 2, 3, 0]]))
    above = root + .020 * normal + [0, 0, .050]
    g.DURATION = 18.
    def command(t):
        if t < 2:
            pos, phase = start, 'prepare_pusher'
        elif t < 5:
            f = (t - 2) / 3
            pos, phase = start * (1 - f) + above * f, 'above_target'
        elif t < 7:
            pos, phase = root + .020 * normal + [0, 0, .050 * (7 - t) / 2], 'descend'
        elif t < 8:
            pos, phase = root + .020 * (8 - t) * normal, 'approach'
        elif t < 12:
            pos, phase = root - distance * (t - 8) / 4 * normal, 'align_blue'
        elif t < 13:
            pos, phase = root + (-distance + .020 * (t - 12)) * normal, 'retreat'
        elif t < 15:
            pos, phase = root + (-distance + .020) * normal + [0, 0, .050 * (t - 13) / 2], 'withdraw'
        else:
            pos, phase = root + (-distance + .020) * normal + [0, 0, .050], 'released_hold'
        ctrl = controls.copy()
        for name, value in goals.items():
            ctrl[name] += (value - ctrl[name]) * np.clip(t / 2, 0, 1)
        poses = dict(left=(pos, turn(np.clip(t, 2, 5)).as_quat()[[3, 0, 1, 2]]), right=starts['right'])
        return poses, ctrl, phase
    g.restore()
    return command, dict(source=str(g.SOURCE), target=g.TARGET, supports=g.SUPPORTS,
        mode='align-released-blue', axis=axis, distance_m=distance, normal=normal.tolist(),
        contact_root_m=root.tolist(), finger_height_m=finger_height, pair_center_m=pair.tolist(), duration_s=g.DURATION, hold_start_s=15.,
        source_sequence_continuity_verified=True, online_object_feedback=False,
        native_verified=False, whole_arm_collision_checked=False, diagnostic_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/blue-release-119/early-guide-clear'))
    parser.add_argument('--axis', choices=('x', 'y'), required=True)
    parser.add_argument('--finger-height-mm', type=float, default=120.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    configure(args.source)
    command, plan = prepare(args.axis, args.finger_height_mm / 1000)
    args.out.mkdir(parents=True, exist_ok=False)
    check = g.precheck(command)
    (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
    (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
    passed = check['passed']
    brief = dict(precheck_passed=passed, geometry_failures=len(check['geometry_failures']),
                 stages={s: len(v) for s, v in check['stages'].items()})
    if passed and args.run:
        g.run(command, plan, args.out, protected_payload='neighbor_5',
              additional_protected_payloads=('neighbor_1', 'neighbor_3', *blue.third.feed.BASES), payload_floor_m=.090)
        brief['dynamics'] = blue.evaluate(args.out, released=True)
        passed = brief['dynamics']['local_eight_brick_final_structure_verified']
    print(json.dumps(brief, indent=2))
    raise SystemExit(0 if passed else 1)
