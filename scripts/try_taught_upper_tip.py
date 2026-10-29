"""Local fixed upper-side push: test tipping before designing a pickup.

Requires retained round-90 state and local model/SDK assets. Object truth is
used for offline geometry, diagnostics and aborts, never online correction.
This does not prove acquisition, upper placement, or native execution.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import check_upper_tip as g

p, m, mj, d = g.p, g.m, g.mj, g.d
TARGET = 'neighbor_1'
DURATION = 18.


def prepare(hold=0.):
    global DURATION
    DURATION = 18. + hold
    report = g.check(TARGET)
    candidate = report['results'][0]
    if not candidate['endpoint']['accepted'] or candidate['non_target_contacts']:
        raise ValueError('Nominal contact endpoint rejected')
    g.restore()
    start = d.mocap_pos[p.t.LM].copy()
    root = np.array(candidate['root_position_m'])
    rotations = Rotation.from_quat(np.array([d.mocap_quat[p.t.LM], candidate['root_quaternion_wxyz']])[:, [1, 2, 3, 0]])
    turn = Slerp([3., 7.], rotations)
    def command(seconds):
        elapsed = seconds
        if seconds >= 13: seconds = max(13., seconds-hold)
        if seconds < 3:
            pos = start + [0, 0, .05 * seconds / 3]; phase = 'rise'
        elif seconds < 7:
            f = (seconds - 3) / 4
            pos = (start + [0, 0, .05]) * (1-f) + (root + [.02, 0, .07]) * f; phase = 'turn'
        elif seconds < 10:
            pos = root + [.02, 0, .07 * (10-seconds)/3]; phase = 'descend'
        elif seconds < 11:
            pos = root + [.02 * (11-seconds), 0, 0]; phase = 'approach'
        elif seconds < 13:
            pos = root + [-.02 * (seconds-11)/2, 0, 0]; phase = 'push'
        elif seconds < 14:
            pos = root + [-.02 + .03 * (seconds-13), 0, 0]; phase = 'retreat'
        elif seconds < 15:
            pos = root + [.01, 0, .05 * (seconds-14)]; phase = 'withdraw'
        else:
            pos = root + [.01, 0, .05]; phase = 'released'
        if hold and 13 <= elapsed < 13+hold: phase = 'hold_tilt'
        quat = turn(np.clip(seconds, 3, 7)).as_quat()[[3, 0, 1, 2]]
        return pos, quat, phase
    return command, report


def precheck(command):
    started = time.monotonic(); failures = []
    count = round(DURATION*4)
    for seconds in np.linspace(0, DURATION, round(DURATION*20)+1):
        g.restore(); pos, quat, phase = command(seconds)
        d.qpos[p.t.LA:p.t.LA+3] = pos; d.qpos[p.t.LA+3:p.t.LA+7] = quat
        mj.mj_forward(m, d)
        for c in d.contact:
            names = [m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_', 'rh_')) for n in names) and TARGET not in names:
                failures.append(dict(t=float(seconds), phase=phase, pair=names, distance_m=float(c.dist)))
    ctx = p.t.routes.context()['left']; ctx['limits'] = ctx['intersection'] + [.020, -.020]; ctx['inset'] = .020
    q = json.loads(Path('runs/fourth-base-89/aligned-low/precheck.json').read_text())['stages'][-1]['result']['final_joints_sdk']
    stages = []
    # Confirm the retained seed matches this source, before claiming continuity.
    g.restore(); actual = np.asarray(p.t.routes.sdk.forward_kinematics('left', q)) @ ctx['mount']
    initial_error = float(np.linalg.norm(actual[:3, 3] - d.mocap_pos[p.t.LM]))
    initial_angle = float((Rotation.from_matrix(actual[:3, :3]).inv() * Rotation.from_quat(d.mocap_quat[p.t.LM][[1, 2, 3, 0]])).magnitude())
    if not failures and initial_error < 1e-5 and initial_angle < 1e-4:
        for seconds in np.linspace(0, DURATION, count+1)[1:]:
            pos, quat, phase = command(seconds)
            ctx['root'][:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            result = p.t.routes.stage('left', ctx, q, phase, (pos-ctx['root'][:3, 3]).tolist())
            stages.append(dict(t=float(seconds), result=result))
            if not result['continuous']: break
            q = result['final_joints_sdk']
    return dict(passed=not failures and len(stages)==count and all(s['result']['continuous'] for s in stages),
                geometry_failures=failures, stages=stages, initial_seed_error_m=initial_error,
                initial_seed_angle_rad=initial_angle, wall_seconds=time.monotonic()-started,
                whole_arm_collision_checked=False, native_verified=False)


def run(command, out):
    g.restore(); start_time = float(d.time)
    names = [m.body(i).name for i in range(m.nbody) if m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')]
    initial = {n:d.xpos[m.body(n).id].copy() for n in names}
    shifts = {n:0. for n in names if n!=TARGET}; abort = None; rows = []; started = time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            pos, quat, phase = command(float(d.time)-start_time)
            d.mocap_pos[p.t.LM] = pos; d.mocap_quat[p.t.LM] = quat
            mj.mj_step(m,d); mj.mj_forward(m,d)
            for n in shifts:
                shifts[n] = max(shifts[n], float(np.linalg.norm(d.xpos[m.body(n).id]-initial[n])))
                if shifts[n] > .002: abort = 'other_brick_moved'
            contacts = []
            for ci,c in enumerate(d.contact):
                ns = [m.geom(int(i)).name for i in c.geom]
                if any(n.startswith(('lh_', 'rh_')) for n in ns) and TARGET not in ns and c.dist < -.0005:
                    abort = 'unexpected_hand_contact'
                if TARGET in ns:
                    f = np.zeros(6); mj.mj_contactForce(m,d,ci,f)
                    vector = c.frame.reshape(3,3).T @ f[:3] * (1 if ns[1]==TARGET else -1)
                    contacts.append(dict(pair=ns, force_xyz_n=vector.tolist()))
            body = m.body(TARGET).id; rotation = Rotation.from_quat(d.xquat[body][[1,2,3,0]])
            corners = rotation.apply(p.CORNERS)+d.xpos[body]
            if not np.isfinite(d.qpos).all(): raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row = dict(elapsed_s=float(d.time)-start_time, phase=phase, position_m=d.xpos[body].tolist(),
                           quaternion_wxyz=d.xquat[body].tolist(), contacts=contacts,
                           corner_heights_m=(corners[:,2]-g.FLOOR).tolist(),
                           tilt_degrees=float(np.degrees(np.arccos(np.clip(rotation.as_matrix()[2,2],-1,1)))))
                rows.append(row); stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort: break
    summary = dict(abort=abort, duration_s=float(d.time)-start_time, wall_seconds=time.monotonic()-started,
                   target=TARGET, translation_mm=((d.xpos[m.body(TARGET).id]-initial[TARGET])*1000).tolist(),
                   maximum_tilt_degrees=max(r['tilt_degrees'] for r in rows),
                   maximum_lowest_corner_height_mm=1000*max(min(r['corner_heights_m']) for r in rows),
                   maximum_center_rise_mm=1000*max(r['position_m'][2]-initial[TARGET][2] for r in rows),
                   final_tilt_degrees=rows[-1]['tilt_degrees'],
                   other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},
                   acquisition_verified=False, upper_placement_verified=False, native_verified=False,
                   whole_arm_collision_checked=False)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--run',action='store_true')
    parser.add_argument('--hold',type=float,choices=[0.,3.],default=0.,help='Hold the fixed push endpoint before retreat')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    command, geometry = prepare(args.hold); checked = precheck(command)
    checked['fixed_endpoint_hold_s'] = args.hold
    (args.out/'precheck.json').write_text(json.dumps(dict(geometry=geometry,**checked),allow_nan=False))
    brief={k:v for k,v in checked.items() if k not in ['geometry_failures','stages']}
    brief.update(geometry_failure_count=len(checked['geometry_failures']),stage_count=len(checked['stages']))
    if args.run and checked['passed']: brief['dynamics']=run(command,args.out)
    print(json.dumps(brief,indent=2)); raise SystemExit(0 if checked['passed'] and not brief.get('dynamics',{}).get('abort') else 1)
