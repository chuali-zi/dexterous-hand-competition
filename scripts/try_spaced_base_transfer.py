"""Continuous local wide staging, blue parking, then second red transfer.

The read-only screen uses measured constituent end states as nominal planning
states. The dynamics itself starts from the original scene, with no resets.
Requires retained runs/77-85 assets. No ROS or Rabokit calls.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import try_wide_table_staging as w
import try_clear_base_lane as lane

t=w.t;m,mj,d=w.m,w.mj,w.d
lane.NEXT_RED=True;lane.CENTER_NEXT_RED=True
WIDE_DIR=Path('runs/wide-staging-85/transverse')
WIDE=json.loads((WIDE_DIR/'result.json').read_text())
WIDE_CHECK=json.loads((WIDE_DIR/'precheck.json').read_text())
BLUE=json.loads((lane.BLUE_SOURCE/'result.json').read_text())
if not WIDE['summary']['wide_staging_verified'] or not BLUE['summary']['local_blue_parking_stable']:
    raise ValueError('Required constituent trial did not finish stable')


def command(seconds):
    # Repeated physics timesteps can put the last command a few ulps past the
    # nominal endpoint. Hold the final pose instead of leaving the schedule.
    seconds=min(seconds,163.)
    if seconds<=51:
        pos,quat,phase=w.command(seconds)
        return pos,quat,phase,'neighbor_9'
    start=w.command(51)[0]
    if seconds<54:
        pos=start+[0,0,.050*(seconds-51)/3];angle=90.;phase='rise_before_blue';target=None
    elif seconds<58:
        f=(seconds-54)/4;end=lane.command(34)[0]
        pos=(start+[0,0,.050])*(1-f)+end*f;angle=90*(1-f);phase='return_orientation_above_table';target=None
    else:
        pos,phase,target=lane.command(seconds-24);angle=0.
    quat=(Rotation.from_euler('z',angle,degrees=True)*w.BASE_ROTATION).as_quat()[[3,0,1,2]]
    return pos,quat,phase,target


def precheck():
    bad=[]
    # Blue has not moved in the first section. After its nominal completion,
    # use the measured blue parking pose with the measured wide red position.
    for seconds in np.linspace(51,163,2241):
        d.qpos[:]=WIDE['rows'][-1]['qpos'] if seconds<=109 else BLUE['rows'][-1]['qpos']
        if seconds>109:
            a=int(m.jnt_qposadr[m.joint('neighbor_9_free').id])
            d.qpos[a:a+7]=WIDE['rows'][-1]['qpos'][a:a+7]
        pos,quat,phase,target=command(float(seconds));d.qpos[t.LA:t.LA+3]=pos;d.qpos[t.LA+3:t.LA+7]=quat
        mj.mj_forward(m,d)
        for c in d.contact:
            pair=[m.geom(int(g)).name for g in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in pair) and target not in pair:
                bad.append(dict(t=float(seconds),phase=phase,pair=pair,distance_m=float(c.dist)))
    ctx=t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020
    q=WIDE_CHECK['stages'][-1]['result']['final_joints_sdk'];stages=[]
    base=[34,34.5,37,38,42,44,45,48,51,54,54.5,57,58,62,64,65,68,71,74,74.5,77,78,82,84,85]
    times=list(np.linspace(51,58,29)[1:])+[v+24 for v in base if v>34]+[v+78 for v in base]
    if not bad:
        for seconds in times:
            pos,quat,phase,_=command(float(seconds));ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            r=t.routes.stage('left',ctx,q,phase,(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=float(seconds),result=r))
            if not r['continuous']:break
            q=r['final_joints_sdk']
    return dict(passed=not bad and len(stages)==len(times) and all(s['result']['continuous'] for s in stages),
                geometry_failures=bad,stages=stages,nominal_composite_geometry=True,whole_arm_collision_checked=False)


def run(out,until=163,resume=None):
    t.initialize();names=['neighbor_9','neighbor_7','neighbor_6']
    others={m.body(i).name:i for i in range(m.nbody)
            if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and m.body(i).name not in names}
    initial={n:d.xpos[i].copy() for n,i in others.items()};shifts={n:0. for n in others}
    rows=[];bad=[];abort=None;prefix=0.;prior_wall=0.
    if resume is not None:
        previous=json.loads((resume/'result.json').read_text())
        if previous['summary']['abort'] or previous['summary']['unexpected_hand_contacts']:
            raise ValueError('Cannot continue an interrupted physical plan with unexpected contacts')
        if previous['summary'].get('checkpoint_next_step_maximum_difference',float('inf'))>1e-10:
            raise ValueError('Checkpoint continuation equivalence has not been verified')
        checkpoint=np.load(resume/'integration-state.npz',allow_pickle=False)
        if str(checkpoint['mujoco_version'])!=mj.__version__ or len(checkpoint['state'])!=mj.mj_stateSize(m,int(checkpoint['spec'])):
            raise ValueError('Checkpoint runtime or model-state dimensions differ')
        rows=previous['rows'];shifts={n:v/1000 for n,v in previous['summary']['other_bricks_max_shift_mm'].items()}
        prefix=previous['summary']['prefix_max_qpos_difference']
        prior_wall=previous['summary'].get('total_wall_seconds',previous['summary']['wall_seconds'])
        mj.mj_setState(m,d,checkpoint['state'],int(checkpoint['spec']))
        if abs(d.time-rows[-1]['t'])>1e-8 or d.time>=until:
            raise ValueError('Checkpoint time does not match trace or requested continuation')
    started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
        for step in range(round(d.time/m.opt.timestep),round(until/m.opt.timestep)+1):
            pos,quat,phase,target=command(float(d.time));d.mocap_pos[t.LM]=pos;d.mocap_quat[t.LM]=quat
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
                if len(rows)<len(WIDE['rows']):prefix=max(prefix,float(np.max(np.abs(d.qpos-WIDE['rows'][len(rows)]['qpos']))))
                bricks={n:dict(position=d.xpos[m.body(n).id].tolist(),quaternion=d.xquat[m.body(n).id].tolist(),contacts=contacts[n]) for n in names}
                row=dict(t=float(d.time),phase=phase,bricks=bricks,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if abort:break
    summary=dict(abort=abort,wall_seconds=time.monotonic()-started,simulation_seconds=float(d.time),
                 prefix_max_qpos_difference=prefix,unexpected_hand_contacts=bad,other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},
                 two_red_staging_verified=False,base_row_verified=False,native_verified=False,official_region_verified=False,full_pyramid_verified=False)
    summary.update(total_wall_seconds=prior_wall+summary['wall_seconds'],requested_until_s=until,
                   continuation_source=str(resume) if resume else None)
    held=[r for r in rows if r['t']>=160.002-1e-8]
    if held:
        metrics={}
        for n in names:
            xyz=np.array([r['bricks'][n]['position'] for r in held]);rot=Rotation.from_quat(np.array([r['bricks'][n]['quaternion'] for r in held])[:,[1,2,3,0]])
            pts=np.concatenate([rr.apply(w.CORNERS)+p for rr,p in zip(rot,xyz)])
            metrics[n]=dict(final_position=xyz[-1].tolist(),final_yaw_degrees=float(rot[-1].as_euler('xyz',degrees=True)[2]),
                            drift_mm=float(np.linalg.norm(xyz-xyz[0],axis=1).max()*1000),angle_drift_degrees=float(np.rad2deg((rot[0].inv()*rot).magnitude().max())),
                            footprint_minimum_m=pts[:,:2].min(0).tolist(),footprint_maximum_m=pts[:,:2].max(0).tolist(),
                            inside_analysis_rectangle=bool(np.all(pts[:,:2]>=t.REGION[:,0]) and np.all(pts[:,:2]<=t.REGION[:,1])),
                            only_table=all(r['bricks'][n]['contacts'] and all(set(c)=={n,'table'} for c in r['bricks'][n]['contacts']) for r in held))
        first=np.array(WIDE['rows'][-1]['position'])
        moved=max(float(np.linalg.norm(np.array(r['bricks']['neighbor_9']['position'])[:2]-first[:2])) for r in rows if r['t']>=51.002-1e-8)
        duration=held[-1]['t']-held[0]['t'];summary.update(held_duration_s=duration,bricks=metrics,first_red_max_shift_mm=moved*1000)
        summary['two_red_staging_verified']=bool(not abort and not bad and duration>=2.98 and moved<=.002 and max(shifts.values())<=.002
            and all(v['only_table'] and v['drift_mm']<=2 and v['angle_drift_degrees']<=2 for v in metrics.values())
            and all(metrics[n]['inside_analysis_rectangle'] for n in ['neighbor_9','neighbor_6']))
    # Preserve full integration state for a later continuation without replaying
    # this 163-second prefix. Includes controls and solver warm-start state.
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    # Check that restoring the saved state reproduces the next physics step,
    # including a phase-boundary command. This does not extend the task trace.
    restored=mj.MjData(m);mj.mj_setState(m,restored,state,spec)
    pos,quat,_,_=command(min(float(d.time),163.))
    for data in [d,restored]:
        data.mocap_pos[t.LM]=pos;data.mocap_quat[t.LM]=quat;mj.mj_step(m,data);mj.mj_forward(m,data)
    original_next=np.empty_like(state);restored_next=np.empty_like(state)
    mj.mj_getState(m,d,original_next,spec);mj.mj_getState(m,restored,restored_next,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(original_next-restored_next)))
    mj.mj_setState(m,d,state,spec)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False))
    print(json.dumps(summary,indent=2))
    if until<163:return not abort and not bad and summary['checkpoint_next_step_maximum_difference']<=1e-10
    return summary['two_red_staging_verified']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--until',type=int,choices=[51,109,163],default=163)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--checked-route',type=Path,help='Reuse this retained run-specific precheck; local development only')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    check=json.loads(args.checked_route.read_text()) if args.checked_route else precheck()
    if args.checked_route and (not check.get('nominal_composite_geometry') or len(check.get('stages',[]))!=77):
        raise ValueError('Expected this route\'s complete numerical and nominal-geometry check')
    (args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    print(json.dumps(dict(passed=check['passed'],geometry_failures=len(check['geometry_failures']),stages=len(check['stages']),
                         last_time=check['stages'][-1]['t'] if check['stages'] else None)),flush=True)
    passed=check['passed']
    if passed and args.run:passed=run(args.out,args.until,args.resume_from)
    raise SystemExit(0 if passed else 1)
