"""Two fixed pushes separated by settling, using retained local teaching.

Requires local rounds 77/78 assets and SDK cache. No platform or live object
feedback. The second waypoint is calibrated from an earlier local trial;
this is not an independently validated native teaching procedure.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path[:0] = ['runs/edge-transfer-78/bimanual', 'runs/edge-transfer-78/bimanual-routes']
import precheck_bimanual as p
import check_routes as routes
from try_table_slide import command_offset

TEACHING = Path('runs/inward-transfer-77/backside/red-backside-lift-first.json')
OLD = json.loads(TEACHING.read_text())
CANDIDATE = OLD['candidate']
ROOT = np.array(CANDIDATE['hand_root_position'])
SHIFT = np.array(OLD['summary']['displacement_mm']) / 1000
# Retain original finger height; the taught shift is horizontal only.
SHIFT[2] = 0
QUAT = np.array(CANDIDATE['hand_root_quaternion_wxyz'])
REGION = np.array([[.365364, .455568], [-.164170, .370613]])
INITIAL_POSE = 'nominal'
m, mj = p.m, p.mujoco
d = mj.MjData(m)
LA, RA = p.roots['left'], p.roots['right']
LM = int(m.body_mocapid[m.body('root_reference').id])
RM = int(m.body_mocapid[m.body('right_root_reference').id])


def command(t):
    if t <= 14:
        offset, phase = command_offset(t, .050, 1, .080, True)
        return ROOT + offset, 'first_' + phase
    start = ROOT + command_offset(14, .050, 1, .080, True)[0]
    finish = ROOT + SHIFT + command_offset(0, .050, 1, .030, True)[0]
    if t < 17:
        return start + (finish - start) * (t - 14) / 3, 'reposition_above_settled_brick'
    offset, phase = command_offset(t - 17, .050, 1, .030, True)
    return ROOT + SHIFT + offset, 'second_' + phase


def initialize():
    mj.mj_resetData(m, d)
    if INITIAL_POSE != 'nominal':
        sign = 1 if INITIAL_POSE == 'plus' else -1
        address = int(m.jnt_qposadr[m.joint('neighbor_9_free').id])
        d.qpos[address:address+2] += sign * np.array([.002, .002])
        d.qpos[address+3:address+7] = Rotation.from_euler('z', sign*3, degrees=True).as_quat()[[3,0,1,2]]
    d.qpos[RA:RA+7] = OLD['rows'][0]['qpos'][RA:RA+7]
    d.mocap_pos[RM] = d.qpos[RA:RA+3]
    d.mocap_quat[RM] = d.qpos[RA+3:RA+7]
    d.qpos[LA:LA+3] = command(0)[0]
    d.qpos[LA+3:LA+7] = QUAT
    d.mocap_quat[LM] = QUAT
    for name, value in CANDIDATE['hand_targets'].items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
        d.ctrl[m.actuator(name).id] = value
    mj.mj_forward(m, d)


def precheck():
    initialize()
    bad = []
    for t in np.linspace(0, 31, 621):
        d.qpos[LA:LA+3] = command(float(t))[0]
        mj.mj_forward(m, d)
        for c in d.contact:
            pair = [m.geom(int(g)).name for g in c.geom]
            if any(n.startswith(('lh_', 'rh_')) for n in pair) and 'neighbor_9' not in pair:
                bad.append(dict(t=float(t), pair=pair, distance_m=float(c.dist)))
    ctx = routes.context()['left']
    ctx['limits'] = ctx['intersection'] + [.020, -.020]
    ctx['inset'] = .020
    ctx['root'][:3, 3] = ROOT
    ctx['root'][:3, :3] = Rotation.from_quat(QUAT[[1, 2, 3, 0]]).as_matrix()
    q = CANDIDATE['contact_joints_sdk']
    stages = []
    if not bad:
        for t in [0, .5, 3, 4, 8, 10, 11, 14, 17, 17.5, 20, 21, 25, 27, 28, 31]:
            pos, phase = command(t)
            result = routes.stage('left', ctx, q, phase, (pos-ROOT).tolist())
            stages.append(result)
            if not result['continuous']:
                break
            q = result['final_joints_sdk']
    return dict(passed=not bad and len(stages) == 16 and all(s['continuous'] for s in stages),
                unexpected_contacts=bad, stages=stages, whole_arm_collision_checked=False,
                scope='Nominal hand geometry and SDK route; target contacts excluded from static screen')


def run(out, check_only=False):
    out.mkdir(parents=True, exist_ok=False)
    check = precheck()
    (out/'precheck.json').write_text(json.dumps(check, allow_nan=False))
    if check_only or not check['passed']:
        print(json.dumps(dict(precheck_passed=check['passed'], contacts=len(check['unexpected_contacts']),
                              stages=len(check['stages']), dynamics_run=False)))
        return check['passed']
    initialize()
    target = m.body('neighbor_9').id
    neighbors = {m.body(i).name: i for i in range(m.nbody)
                 if (m.body(i).name == 'target' or m.body(i).name.startswith('neighbor_')) and i != target}
    initial = {name: d.xpos[i].copy() for name, i in neighbors.items()}
    shifts = {name: 0. for name in neighbors}
    rows, bad = [], []
    started = time.monotonic()
    abort = None
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(31/m.opt.timestep)+1):
            pos, phase = command(float(d.time))
            d.mocap_pos[LM] = pos
            mj.mj_step(m, d)
            mj.mj_forward(m, d)
            for name, i in neighbors.items():
                shifts[name] = max(shifts[name], float(np.linalg.norm(d.xpos[i][:2]-initial[name][:2])))
            contacts = []
            for c in d.contact:
                pair = [m.geom(int(g)).name for g in c.geom]
                if 'neighbor_9' in pair:
                    contacts.append(pair)
                elif any(n.startswith(('lh_', 'rh_')) for n in pair):
                    bad.append(dict(t=float(d.time), pair=pair, distance_m=float(c.dist)))
                    if c.dist < -.0005:
                        abort = 'unexpected_hand_collision'
            if not np.isfinite(d.qpos).all():
                raise RuntimeError('Nonfinite simulation state')
            if step % 10 == 0 or abort:
                row = dict(t=float(d.time), phase=phase, position=d.xpos[target].tolist(),
                           quaternion=d.xquat[target].tolist(), contacts=contacts,
                           qpos=d.qpos.tolist(), qvel=d.qvel.tolist())
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
            if abort:
                break
    held = [r for r in rows if r['t'] >= 28.002-1e-8]
    summary = dict(abort=abort, local_reposition_verified=False, wall_seconds=time.monotonic()-started,
                   initial_pose=INITIAL_POSE,
                   simulation_seconds=float(d.time), unintended_contacts=bad,
                   neighbor_max_shift_mm={k: v*1000 for k, v in shifts.items()},
                   teaching_source=str(TEACHING), fixed_second_xy_shift_m=SHIFT.tolist(),
                   native_verified=False, official_region_verified=False, full_pyramid_verified=False)
    if held:
        xyz = np.array([r['position'] for r in held])
        rot = Rotation.from_quat(np.array([r['quaternion'] for r in held])[:, [1, 2, 3, 0]])
        drift = float(np.linalg.norm(xyz-xyz[0], axis=1).max())
        turn = float(np.rad2deg((rot[0].inv()*rot).magnitude().max()))
        corners = np.array([[x,y,z] for x in [-.025,.025] for y in [-.025,.025] for z in [-.025,.025]])
        inside = all(np.all((pts[:, :2] >= REGION[:, 0]) & (pts[:, :2] <= REGION[:, 1]))
                     for pts in [rr.apply(corners)+pos for rr, pos in zip(rot, xyz)])
        only_table = all(r['contacts'] and all(set(c) == {'table','neighbor_9'} for c in r['contacts']) for r in held)
        duration = held[-1]['t']-held[0]['t']
        summary.update(final_position=xyz[-1].tolist(), released_drift_mm=drift*1000,
                       released_angle_drift_deg=turn, released_duration_s=duration,
                       entire_footprint_in_analysis_rectangle=inside, released_only_table_contact=only_table)
        summary['local_reposition_verified'] = bool(not abort and not bad and inside and only_table
                                                    and duration >= 2.98 and drift <= .002 and turn <= 2
                                                    and max(shifts.values()) <= .002)
    (out/'result.json').write_text(json.dumps(dict(summary=summary, rows=rows), allow_nan=False))
    print(json.dumps(summary, indent=2))
    return summary['local_reposition_verified']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--initial-pose', choices=['nominal', 'plus', 'minus'], default='nominal',
                        help='Local sensitivity only: plus/minus 2 mm in X,Y and 3 degrees yaw; commands unchanged')
    args = parser.parse_args()
    INITIAL_POSE = args.initial_pose
    raise SystemExit(0 if run(args.out, args.check_only) else 1)
