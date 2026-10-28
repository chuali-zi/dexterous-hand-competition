"""Fixed local table sequence: retained red placement, blue parking, next red.

Retained assets/SDK required. Frozen geometry screens do not predict contact
dynamics. Temporary parking is not an authenticated construction area.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import try_taught_table_reposition as t

m, mj, d = t.m, t.mj, t.d
SOURCE = Path('runs/taught-reposition-83/two-push')
PREVIOUS = json.loads((SOURCE/'result.json').read_text())
PRECHECK = json.loads((SOURCE/'precheck.json').read_text())
BASE = m.body('neighbor_9').pos.copy()
DISTANCES = (.080, .080, .030)
NEXT_RED = False
CENTER_NEXT_RED = False
BLUE_SOURCE = Path('runs/base-lane-84/short-last-push')


def push_command(local, root, distances=DISTANCES, approach_height=.080):
    """Two 80mm pushes and one 30mm push with fixed taught resets."""
    index = min(int(local // 17), 2)
    start = index * 17
    if index and local-start < 3:
        previous = root+(index-1)*t.SHIFT+t.command_offset(14,.050,1,distances[index-1],True)[0]
        following = root+index*t.SHIFT+t.command_offset(0,.050,1,distances[index],True)[0]
        following[2] += approach_height-.080
        return previous+(following-previous)*(local-start)/3, 'reposition'
    elapsed = local if index == 0 else local-start-3
    offset=t.command_offset(elapsed,.050,1,distances[index],True)[0]
    if elapsed<=3:offset[2]*=approach_height/.080
    return root+index*t.SHIFT+offset, f'push_{index+1}'


def command(seconds):
    if seconds <= 31:
        pos, phase = t.command(seconds)
        return pos, phase, 'neighbor_9'
    operations=[(31.,'neighbor_7',t.command(31)[0],DISTANCES)]
    if NEXT_RED:
        blue_root=t.ROOT+(m.body('neighbor_7').pos-BASE)*[1,1,0]
        operations.append((85.,'neighbor_6',push_command(51,blue_root)[0],(.080,.080,.080)))
    for start, name, previous, distances in operations:
        approach_height=.050 if name=='neighbor_6' else .080
        root = t.ROOT+(m.body(name).pos-BASE)*[1,1,0]
        if name=='neighbor_6' and CENTER_NEXT_RED:
            # Recorded first-stroke contacts: index near centre, middle about
            # +19mm. Move the pair by -10mm; do not change force or physics.
            root[1]-=.010
        local = seconds-start
        if local < 0:
            continue
        if local < 3:
            high = push_command(0,root,distances,approach_height)[0]
            return previous+(high-previous)*local/3, 'change_target', None
        if local <= 54:
            pos, phase = push_command(local-3,root,distances,approach_height)
            return pos, name+'_'+phase, name
    raise ValueError('Outside the fixed sequence')


def check():
    # Start with the observed preceding end state. Only nominal hand root moves;
    # Active-target contacts are excluded; other hand contacts are rejected.
    # Next-red mode uses the ACTUAL previously parked blue state, not a guessed
    # displaced state. Its dynamics still replays the complete prefix.
    reference=json.loads((BLUE_SOURCE/'result.json').read_text()) if NEXT_RED else PREVIOUS
    if NEXT_RED and not reference['summary']['local_blue_parking_stable']:
        raise ValueError('Blue reference did not finish stable')
    d.qpos[:] = reference['rows'][-1]['qpos']
    failures = []
    start=85 if NEXT_RED else 31
    for seconds in np.linspace(start,start+54,1081):
        pos, phase, target = command(float(seconds))
        d.qpos[t.LA:t.LA+3] = pos
        d.qpos[t.LA+3:t.LA+7] = t.QUAT
        mj.mj_forward(m,d)
        for c in d.contact:
            pair = [m.geom(int(g)).name for g in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in pair) and target not in pair:
                failures.append(dict(t=float(seconds), phase=phase, pair=pair, distance_m=float(c.dist)))
    context = t.routes.context()['left']
    context['root'][:3,3] = t.ROOT
    context['root'][:3,:3] = Rotation.from_quat(t.QUAT[[1,2,3,0]]).as_matrix()
    context['limits'] = context['intersection']+[.020,-.020]
    context['inset'] = .020
    q = (json.loads((BLUE_SOURCE/'precheck.json').read_text())['stages'][-1]['result']['final_joints_sdk']
         if NEXT_RED else PRECHECK['stages'][-1]['final_joints_sdk'])
    stages = []
    times = [34,34.5,37,38,42,44,45,48,51,54,54.5,57,58,62,64,65,68,71,74,74.5,77,78,82,84,85]
    if NEXT_RED:times=[seconds+54 for seconds in times]
    if not failures:
        for seconds in times:
            pos,phase,_ = command(seconds)
            r = t.routes.stage('left',context,q,phase,(pos-t.ROOT).tolist())
            stages.append(dict(t=seconds,result=r))
            if not r['continuous']:
                break
            q = r['final_joints_sdk']
    return dict(passed=not failures and len(stages)==len(times) and all(s['result']['continuous'] for s in stages),
                geometry_failures=failures, stages=stages, dynamics_run=False,next_red=NEXT_RED,
                center_next_red=CENTER_NEXT_RED,
                whole_arm_collision_checked=False, official_region_verified=False)


def dynamics(out):
    t.initialize()
    names=['neighbor_9','neighbor_7']+(['neighbor_6'] if NEXT_RED else [])
    reference=json.loads((BLUE_SOURCE/'result.json').read_text()) if NEXT_RED else PREVIOUS
    end=139 if NEXT_RED else 85
    others={m.body(i).name:i for i in range(m.nbody)
            if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and m.body(i).name not in names}
    initial={n:d.xpos[i].copy() for n,i in others.items()}
    shifts={n:0. for n in others}
    rows=[];bad=[];abort=None;prefix=0.;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(end/m.opt.timestep)+1):
            pos,phase,target=command(float(d.time));d.mocap_pos[t.LM]=pos
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,i in others.items():shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[i][:2]-initial[n][:2])))
            contacts={n:[] for n in names}
            for c in d.contact:
                pair=[m.geom(int(g)).name for g in c.geom]
                for n in names:
                    if n in pair:contacts[n].append(pair)
                if any(n.startswith(('lh_','rh_')) for n in pair) and target not in pair:
                    bad.append(dict(t=float(d.time),pair=pair,distance_m=float(c.dist)))
                    if c.dist<-.0005:abort='unexpected_hand_contact'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==0 or abort:
                bricks={n:dict(position=d.xpos[m.body(n).id].tolist(),quaternion=d.xquat[m.body(n).id].tolist(),contacts=contacts[n]) for n in names}
                if len(rows)<len(reference['rows']):prefix=max(prefix,float(np.max(np.abs(d.qpos-reference['rows'][len(rows)]['qpos']))))
                row=dict(t=float(d.time),phase=phase,bricks=bricks,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if abort:break
    held=[r for r in rows if r['t']>=end-3+.002-1e-8]
    summary=dict(abort=abort,wall_seconds=time.monotonic()-started,simulation_seconds=float(d.time),
                 prefix_max_qpos_difference=prefix,unexpected_hand_contacts=bad,
                 other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},
                 local_blue_parking_stable=False,two_red_bricks_in_analysis_area=False,next_red=NEXT_RED,
                 center_next_red=CENTER_NEXT_RED,
                 lane_clearance_verified=False,native_verified=False,
                 official_region_verified=False,full_pyramid_verified=False)
    # End-of-run stillness cannot hide displacement of an already placed brick.
    previous_red=np.array(PREVIOUS['rows'][-1]['position'])
    red_shift=max((float(np.linalg.norm(np.array(row['bricks']['neighbor_9']['position'])[:2]-previous_red[:2]))
                   for row in rows if row['t']>=31.002-1e-8),default=0.)
    summary['placed_red_maximum_shift_mm']=red_shift*1000
    summary['placed_red_preserved']=red_shift<=.002
    if held:
        metrics={}
        for n in names:
            xyz=np.array([r['bricks'][n]['position'] for r in held])
            rot=Rotation.from_quat(np.array([r['bricks'][n]['quaternion'] for r in held])[:,[1,2,3,0]])
            corners=np.array([[x,y,z] for x in [-.025,.025] for y in [-.025,.025] for z in [-.025,.025]])
            all_corners=np.concatenate([rr.apply(corners)+pt for rr,pt in zip(rot,xyz)])
            metrics[n]=dict(final_position=xyz[-1].tolist(),position_drift_mm=float(np.linalg.norm(xyz-xyz[0],axis=1).max()*1000),
                            angle_drift_degrees=float(np.rad2deg((rot[0].inv()*rot).magnitude().max())),
                            footprint_minimum_m=all_corners[:,:2].min(0).tolist(),footprint_maximum_m=all_corners[:,:2].max(0).tolist(),
                            table_only=all(r['bricks'][n]['contacts'] and all(set(c)=={n,'table'} for c in r['bricks'][n]['contacts']) for r in held))
        summary.update(held_duration_s=held[-1]['t']-held[0]['t'],bricks=metrics)
        stable=all(r['position_drift_mm']<=2 and r['angle_drift_degrees']<=2 and r['table_only'] for r in metrics.values())
        summary['local_blue_parking_stable']=bool(not abort and not bad and stable and summary['held_duration_s']>=2.98 and max(shifts.values())<=.002)
        # Conservative axis-aligned corridor beside the already placed red.
        summary['blue_left_minus_red_right_mm']=(metrics['neighbor_7']['footprint_minimum_m'][0]-metrics['neighbor_9']['footprint_maximum_m'][0])*1000
        if NEXT_RED:
            inside=all(np.all(np.array(metrics[n]['footprint_minimum_m'])>=t.REGION[:,0]) and
                       np.all(np.array(metrics[n]['footprint_maximum_m'])<=t.REGION[:,1]) for n in ['neighbor_9','neighbor_6'])
            summary['two_red_bricks_in_analysis_area']=bool(summary['local_blue_parking_stable'] and inside and summary['placed_red_preserved'])
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False))
    print(json.dumps(summary,indent=2));return summary['two_red_bricks_in_analysis_area'] if NEXT_RED else summary['local_blue_parking_stable']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='store_true',help='Run local dynamics only if precheck passes')
    parser.add_argument('--next-red',action='store_true',help='Append red-6 transfer; requires retained stable blue trial')
    parser.add_argument('--center-next-red',action='store_true',help='Move red-6 pusher pair Y -10mm based on retained contact offsets')
    args=parser.parse_args()
    NEXT_RED=args.next_red
    CENTER_NEXT_RED=args.center_next_red
    if CENTER_NEXT_RED and not NEXT_RED:parser.error('--center-next-red requires --next-red')
    args.out.mkdir(parents=True,exist_ok=False)
    result=check()
    (args.out/'precheck.json').write_text(json.dumps(result,allow_nan=False))
    print(json.dumps(dict(passed=result['passed'],geometry_failures=len(result['geometry_failures']),
                         first_geometry_failure=result['geometry_failures'][:1],stages=len(result['stages']),
                         last_stage=result['stages'][-1]['t'] if result['stages'] else None),indent=2))
    passed=result['passed']
    if passed and args.run:passed=dynamics(args.out)
    raise SystemExit(0 if passed else 1)
