"""Local fixed transverse push to leave clearance for the next base brick.

Uses retained rounds 77-84 assets, no platform/ROS. Prior settled pose is an
offline teaching reference, never current-state feedback during execution.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import try_taught_table_reposition as t

m,mj,d=t.m,t.mj,t.d
OLD=json.loads(Path('runs/taught-reposition-83/two-push/result.json').read_text())
PRECHECK=json.loads(Path('runs/taught-reposition-83/two-push/precheck.json').read_text())
SETTLED=OLD['rows'][-1]
ROTATION=Rotation.from_euler('z',90,degrees=True)
BASE_ROTATION=Rotation.from_quat(t.QUAT[[1,2,3,0]])
CORNERS=np.array([[x,y,z] for x in [-.025,.025] for y in [-.025,.025] for z in [-.025,.025]])


def contact_root():
    """Place two finger mesh faces at the recorded brick's +Y support plane."""
    t.initialize()
    d.qpos[t.LA:t.LA+3]=[0,0,t.ROOT[2]]
    d.qpos[t.LA+3:t.LA+7]=(ROTATION*BASE_ROTATION).as_quat()[[3,0,1,2]]
    mj.mj_forward(m,d)
    vertices=[]
    for name in ['lh_index_proximal','lh_index_distal','lh_middle_proximal','lh_middle_distal']:
        gid=m.geom(name).id;mesh=int(m.geom_dataid[gid]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
        vertices.extend(m.mesh_vert[a:a+n]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid])
    vertices=np.array(vertices)
    center=np.array(SETTLED['position'])
    points=Rotation.from_quat(np.array(SETTLED['quaternion'])[[1,2,3,0]]).apply(CORNERS)+center
    return np.array([center[0]-(vertices[:,0].min()+vertices[:,0].max())/2,
                     points[:,1].max()-vertices[:,1].min()-.0005,t.ROOT[2]])


ROOT=contact_root()
START=t.command(31)[0]


def command(seconds):
    if seconds<=31:
        pos,phase=t.command(seconds)
        return pos,t.QUAT,phase
    if seconds<34:
        pos=START+[0,0,.050*(seconds-31)/3];angle=0.;phase='rise_before_turn'
    elif seconds<38:
        f=(seconds-34)/4;pos=(START+[0,0,.050])*(1-f)+(ROOT+[0,.020,.100])*f;angle=90*f;phase='turn_above_table'
    elif seconds<41:
        pos=ROOT+[0,.020,.100*(41-seconds)/3];angle=90.;phase='descend'
    elif seconds<42:
        pos=ROOT+[0,.020*(42-seconds),0];angle=90.;phase='approach'
    elif seconds<45:
        pos=ROOT+[0,-.050*(seconds-42)/3,0];angle=90.;phase='transverse_push'
    elif seconds<46:
        pos=ROOT+[0,-.050+.020*(seconds-45),0];angle=90.;phase='retreat'
    elif seconds<48:
        pos=ROOT+[0,-.030,.050*(seconds-46)/2];angle=90.;phase='withdraw'
    else:
        pos=ROOT+[0,-.030,.050];angle=90.;phase='released'
    quat=(Rotation.from_euler('z',angle,degrees=True)*BASE_ROTATION).as_quat()[[3,0,1,2]]
    return pos,quat,phase


def precheck():
    d.qpos[:]=SETTLED['qpos'];bad=[]
    for seconds in np.linspace(31,51,401):
        pos,quat,phase=command(float(seconds));d.qpos[t.LA:t.LA+3]=pos;d.qpos[t.LA+3:t.LA+7]=quat
        mj.mj_forward(m,d)
        for c in d.contact:
            pair=[m.geom(int(g)).name for g in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in pair) and 'neighbor_9' not in pair:
                bad.append(dict(t=float(seconds),phase=phase,pair=pair,distance_m=float(c.dist)))
    ctx=t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020
    q=PRECHECK['stages'][-1]['final_joints_sdk'];stages=[]
    times=np.linspace(31,51,81)[1:]
    if not bad:
        for seconds in times:
            pos,quat,phase=command(float(seconds));ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            r=t.routes.stage('left',ctx,q,phase,(pos-ctx['root'][:3,3]).tolist())
            stages.append(dict(t=float(seconds),result=r))
            if not r['continuous']:break
            q=r['final_joints_sdk']
    return dict(passed=not bad and len(stages)==len(times) and all(s['result']['continuous'] for s in stages),
                geometry_failures=bad,stages=stages,contact_root_m=ROOT.tolist(),whole_arm_collision_checked=False)


def run(out):
    t.initialize();target=m.body('neighbor_9').id;neighbors={m.body(i).name:i for i in range(m.nbody)
        if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and i!=target}
    initial={n:d.xpos[i].copy() for n,i in neighbors.items()};shifts={n:0. for n in neighbors}
    rows=[];bad=[];abort=None;prefix=0.;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(51/m.opt.timestep)+1):
            pos,quat,phase=command(float(d.time));d.mocap_pos[t.LM]=pos;d.mocap_quat[t.LM]=quat
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,i in neighbors.items():shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[i][:2]-initial[n][:2])))
            contacts=[]
            for c in d.contact:
                pair=[m.geom(int(g)).name for g in c.geom]
                if 'neighbor_9' in pair:contacts.append(pair)
                elif any(n.startswith(('lh_','rh_')) for n in pair):
                    bad.append(dict(t=float(d.time),pair=pair,distance_m=float(c.dist)))
                    if c.dist<-.0005:abort='unexpected_hand_contact'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==0 or abort:
                if len(rows)<len(OLD['rows']):prefix=max(prefix,float(np.max(np.abs(d.qpos-OLD['rows'][len(rows)]['qpos']))))
                row=dict(t=float(d.time),phase=phase,position=d.xpos[target].tolist(),quaternion=d.xquat[target].tolist(),
                         contacts=contacts,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if abort:break
    held=[r for r in rows if r['t']>=48.002-1e-8]
    summary=dict(abort=abort,wall_seconds=time.monotonic()-started,simulation_seconds=float(d.time),
                 prefix_max_qpos_difference=prefix,unintended_hand_contacts=bad,neighbor_max_shift_mm={n:v*1000 for n,v in shifts.items()},
                 wide_staging_verified=False,native_verified=False,full_pyramid_verified=False,official_region_verified=False)
    if held:
        xyz=np.array([r['position'] for r in held]);rot=Rotation.from_quat(np.array([r['quaternion'] for r in held])[:,[1,2,3,0]])
        pts=np.concatenate([rr.apply(CORNERS)+p for rr,p in zip(rot,xyz)])
        drift=float(np.linalg.norm(xyz-xyz[0],axis=1).max());angle=float(np.rad2deg((rot[0].inv()*rot).magnitude().max()))
        only_table=all(r['contacts'] and all(set(c)=={'table','neighbor_9'} for c in r['contacts']) for r in held)
        distance=float(SETTLED['position'][1]-xyz[-1,1]);duration=held[-1]['t']-held[0]['t']
        summary.update(final_position=xyz[-1].tolist(),minus_y_displacement_mm=distance*1000,drift_mm=drift*1000,
                       angle_drift_degrees=angle,held_duration_s=duration,only_table=only_table,
                       footprint_minimum_m=pts[:,:2].min(0).tolist(),footprint_maximum_m=pts[:,:2].max(0).tolist())
        inside=np.all(pts[:,:2]>=t.REGION[:,0]) and np.all(pts[:,:2]<=t.REGION[:,1])
        summary['wide_staging_verified']=bool(not abort and not bad and only_table and duration>=2.98 and drift<=.002 and angle<=2
                                              and distance>=.030 and pts[:,1].max()<=.085 and inside and max(shifts.values())<=.002)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));print(json.dumps(summary,indent=2))
    return summary['wide_staging_verified']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    check=precheck();(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    print(json.dumps(dict(passed=check['passed'],geometry_failures=len(check['geometry_failures']),stages=len(check['stages']),
                         last_time=check['stages'][-1]['t'] if check['stages'] else None)),flush=True)
    passed=check['passed']
    if passed and args.run:passed=run(args.out)
    raise SystemExit(0 if passed else 1)
