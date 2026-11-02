"""Fork one measured fixed-jaw trajectory to diagnose local friction sensitivity.

This is a counterfactual model experiment, never a competition/acquisition pass.
Replay the original approach once, fork its full integration state before lift,
and vary only the friction cone and impedance ratio. No scene file is edited.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import try_two_hand_pinch as pinch

g = pinch.g
m, d, mj = g.m, g.d, g.mj


def motion_metrics(rows):
    held = [row for row in rows if row['elapsed_s'] >= 23 - 1e-8]
    early = [row for row in rows if 20.1 <= row['elapsed_s'] <= 20.5]
    result = {}
    if held:
        rotation = Rotation.from_quat(np.array([row['quaternion_wxyz'] for row in held])[:, [1, 2, 3, 0]])
        result['held_angular_drift_degrees'] = float(np.degrees((rotation[0].inv() * rotation).magnitude().max()))
        result['final_face_tilt_degrees'] = float(np.degrees(np.arccos(np.clip(np.max(np.abs(rotation[-1].as_matrix()[2, :])), 0, 1))))
    if early:
        result['early_maximum_contact_finger_tracking_error_rad'] = max(
            abs(v['actual_rad'] - v['command_rad']) for row in early for v in row['fingers'].values())
        result['early_maximum_contact_finger_actuator_force'] = max(
            abs(v['actuator_force']) for row in early for v in row['fingers'].values())
        result['early_maximum_root_tracking_error_mm'] = max(
            max(row['root_tracking_error_mm'].values()) for row in early)
        slips = [c['tangential_speed_mm_s'] for row in early for c in row['contacts']
                 if any(n.startswith(('lh_', 'rh_')) for n in c['pair'])]
        result['early_mean_hand_contact_tangential_speed_mm_s'] = float(np.mean(slips)) if slips else None
    return result


def snapshot(t, controls, poses, normal):
    target = m.body(pinch.feed.TARGET).id
    contacts = []
    for ci, contact in enumerate(d.contact):
        names = [m.geom(int(i)).name for i in contact.geom]
        if pinch.feed.TARGET not in names:
            continue
        force = np.zeros(6)
        mj.mj_contactForce(m, d, ci, force)
        vector = contact.frame.reshape(3, 3).T @ force[:3]
        vector *= 1 if names[1] == pinch.feed.TARGET else -1
        velocities = []
        for geom in contact.geom:
            jp = np.zeros((3, m.nv))
            jr = np.zeros_like(jp)
            mj.mj_jac(m, d, jp, jr, contact.pos, int(m.geom_bodyid[geom]))
            velocities.append(jp @ d.qvel)
        relative = velocities[1] - velocities[0]
        n = contact.frame[:3]
        slip = relative - np.dot(relative, n) * n
        contacts.append(dict(pair=names, force_xyz_n=vector.tolist(),
                             tangential_speed_mm_s=float(np.linalg.norm(slip) * 1000)))
    rotation = Rotation.from_quat(d.xquat[target][[1, 2, 3, 0]])
    bottom = (rotation.apply(g.p.CORNERS) + d.xpos[target])[:, 2].min() - g.a.g.FLOOR
    fingers = {}
    for side, prefix, selected in [('left', 'lh', ('index', 'middle')),
                                    ('right', 'rh', ('ring', 'pinky'))]:
        for finger in selected:
            for suffix in ('mcp_pitch', 'dip'):
                name = f'{prefix}_{finger}_{suffix}'
                actuator = m.actuator(name).id
                q = float(d.qpos[m.jnt_qposadr[m.joint(name).id]])
                fingers[name] = dict(command_rad=controls[name], actual_rad=q,
                                     actuator_force=float(d.actuator_force[actuator]))
    root_error = {side: float(np.linalg.norm(d.qpos[address:address + 3] - poses[side][0]) * 1000)
                  for side, address in [('left', g.p.t.LA), ('right', g.p.t.RA)]}
    return dict(elapsed_s=t, position_m=d.xpos[target].tolist(), quaternion_wxyz=d.xquat[target].tolist(),
                bottom_mm=float(bottom * 1000), contacts=contacts, fingers=fingers,
                root_tracking_error_mm=root_error,
                inward_force_n={prefix: sum(float(np.dot(c['force_xyz_n'], -sign * normal))
                    for c in contacts if any(n.startswith(prefix) for n in c['pair']))
                    for prefix, sign in [('lh_', 1), ('rh_', -1)]},
                hand_up_force_n=sum(c['force_xyz_n'][2] for c in contacts
                                   if any(n.startswith(('lh_', 'rh_')) for n in c['pair'])))


def run(reference, out):
    original_plan = json.loads((reference / 'plan.json').read_text())
    original_check = json.loads((reference / 'precheck.json').read_text())
    original = json.loads((reference / 'result.json').read_text())
    if not original_check['passed'] or original['summary']['abort']:
        raise ValueError('Expected the completed, checked original trajectory')
    report = json.loads(Path(original_plan['candidate_source']).read_text())
    candidates = [c for c in report['candidates'] if c['endpoint_candidate']]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one checked contact candidate')
    source = Path(report['source'])
    pinch.a.configure_continuation(source, pinch.feed.TARGET, southern_receiver=True)
    pinch.feed.configure(source)
    left, right = pinch.feed.seeds()
    g.CHECK = {**g.CHECK, 'stages': {'left': [dict(final_joints_sdk=left)],
                                   'right': [dict(final_joints_sdk=right)]}}
    command, plan = pinch.prepare(candidates[0], original_plan['nominal_penetration_per_hand_m'] * 1000)
    for key, value in plan.items():
        if original_plan[key] != value:
            raise ValueError(f'Command plan differs from checked reference: {key}')
    normal = np.array(plan['normal'])
    options = dict(cone=int(m.opt.cone), impratio=float(m.opt.impratio))
    if options != dict(cone=int(mj.mjtCone.mjCONE_PYRAMIDAL), impratio=1.):
        raise ValueError('Original model settings differ from the experiment baseline')
    out.mkdir(parents=True, exist_ok=False)
    pinch.a.restore()
    start = float(d.time)
    protected = {m.body(i).name: d.xpos[i].copy() for i in range(m.nbody)
                 if (m.body(i).name == 'target' or m.body(i).name.startswith('neighbor_'))
                 and m.body(i).name != pinch.feed.TARGET}

    def step():
        elapsed = float(d.time) - start
        poses, controls, _ = command(elapsed)
        for side, index in [('left', g.p.t.LM), ('right', g.p.t.RM)]:
            d.mocap_pos[index], d.mocap_quat[index] = poses[side]
        for name, value in controls.items():
            d.ctrl[m.actuator(name).id] = value
        mj.mj_step(m, d)
        mj.mj_forward(m, d)
        if not np.isfinite(d.qpos).all():
            raise RuntimeError('Nonfinite dynamics')
        return poses, controls

    started = time.monotonic()
    for _ in range(round(20 / m.opt.timestep)):
        step()
    prefix_wall = time.monotonic() - started
    previous = min(original['rows'], key=lambda row: abs(row['elapsed_s'] - 20))
    prefix_error = max(float(np.max(np.abs(d.qpos - previous['qpos']))),
                       float(np.max(np.abs(d.qvel - previous['qvel']))))
    if abs(previous['elapsed_s'] - (float(d.time) - start)) > 1e-8 or prefix_error > 1e-10:
        raise ValueError(f'Original prefix did not reproduce: {prefix_error}')
    spec = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(m, spec))
    mj.mj_getState(m, d, state, spec)
    np.savez(out / 'before-lift.npz', state=state, spec=int(spec), mujoco_version=mj.__version__)
    branches = {}
    try:
        for name, cone, ratio in [('original', mj.mjtCone.mjCONE_PYRAMIDAL, 1.),
                                   ('elliptic', mj.mjtCone.mjCONE_ELLIPTIC, 1.),
                                   ('elliptic-impratio100', mj.mjtCone.mjCONE_ELLIPTIC, 100.)]:
            m.opt.cone, m.opt.impratio = cone, ratio
            mj.mj_setState(m, d, state, spec)
            mj.mj_forward(m, d)
            rows = []
            shifts = {n: 0. for n in protected}
            abort = None
            started = time.monotonic()
            for index in range(round(6 / m.opt.timestep)):
                poses, controls = step()
                for n, xyz in protected.items():
                    shifts[n] = max(shifts[n], float(np.linalg.norm(d.xpos[m.body(n).id] - xyz) * 1000))
                for c in d.contact:
                    ns = [m.geom(int(i)).name for i in c.geom]
                    if any(n.startswith(('lh_', 'rh_')) for n in ns) and pinch.feed.TARGET not in ns and c.dist < -.0005:
                        abort = 'unexpected_hand_contact'
                if max(shifts.values()) > 2:
                    abort = 'other_brick_moved'
                if index % 10 == 9 or abort:
                    rows.append(snapshot(float(d.time) - start, controls, poses, normal))
                    if rows[-1]['bottom_mm'] < -20:
                        abort = 'target_below_limit'
                if abort:
                    break
            wall = time.monotonic() - started
            held = [row for row in rows if row['elapsed_s'] >= 23 - 1e-8]
            positions = np.array([row['position_m'] for row in held])
            early = [row for row in rows if 20.1 <= row['elapsed_s'] <= 20.5]
            summary = dict(abort=abort, cone=int(cone), impratio=ratio, wall_seconds=wall,
                simulated_seconds=float(d.time) - start - 20, maximum_bottom_mm=max(row['bottom_mm'] for row in rows),
                early_up_force_range_n=[min(row['hand_up_force_n'] for row in early), max(row['hand_up_force_n'] for row in early)] if early else None,
                held_minimum_bottom_mm=min((row['bottom_mm'] for row in held), default=None),
                held_drift_mm=float(np.linalg.norm(positions - positions[0], axis=1).max() * 1000) if held else None,
                held_only_hands=bool(held and all(row['contacts'] and all(any(n.startswith(('lh_', 'rh_')) for n in c['pair']) for c in row['contacts']) for row in held)),
                held_minimum_up_force_n=min((row['hand_up_force_n'] for row in held), default=None),
                other_bricks_max_shift_mm=shifts, diagnostic_only=True, acquisition_verified=False,
                native_verified=False, full_pyramid_verified=False)
            summary.update(motion_metrics(rows))
            if name == 'original':
                summary['original_final_qpos_qvel_max_error'] = max(
                    float(np.max(np.abs(d.qpos - original['rows'][-1]['qpos']))),
                    float(np.max(np.abs(d.qvel - original['rows'][-1]['qvel']))))
                if summary['original_final_qpos_qvel_max_error'] > 1e-10:
                    raise ValueError('Unmodified diagnostic branch failed to reproduce original dynamics')
            branches[name] = summary
            (out / (name + '.json')).write_text(json.dumps(dict(summary=summary, rows=rows), allow_nan=False))
            print(json.dumps(dict(branch=name, summary=summary)), flush=True)
    finally:
        m.opt.cone, m.opt.impratio = options['cone'], options['impratio']
        g.restore()
    result = dict(reference=str(reference), source=str(source), diagnostic_only=True,
        prefix_simulated_seconds=20., prefix_wall_seconds=prefix_wall, prefix_replay_error=prefix_error,
        original_options=options, restored_options=dict(cone=int(m.opt.cone), impratio=float(m.opt.impratio)),
        branches=branches, native_verified=False, full_pyramid_verified=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('runs/third-flat-113/opposed-strong-clear'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.out)
