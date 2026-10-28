"""Continue the retained aligned pair: park yellow 5, then transfer red 4.

Local retained MuJoCo/SDK assets required. Fixed teaching only; no platform
calls or live object-pose control. Temporary staging is not a full base row.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import try_close_staged_pair as pair
import try_clear_base_lane as lane

t,m,mj,d=pair.t,pair.m,pair.mj,pair.d
CORNERS=pair.CORNERS


def configure(target):
    global TARGET,SOURCE,RECORD,CHECK,STATE,START,START_POS,START_ROT,ROOT,DISTANCES,HEIGHT
    TARGET=target
    SOURCE=Path('runs/pair-alignment-87/final-close' if target=='neighbor_5' else 'runs/third-base-88/yellow')
    RECORD=json.loads((SOURCE/'result.json').read_text());CHECK=json.loads((SOURCE/'precheck.json').read_text())
    STATE=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
    s=RECORD['summary'];flag='local_pair_spacing_verified' if target=='neighbor_5' else 'target_staging_verified'
    if not s.get(flag) or s.get('checkpoint_next_step_maximum_difference',float('inf'))>1e-10:
        raise ValueError('Required source outcome or checkpoint is unverified')
    restore();START=float(d.time);START_POS=d.mocap_pos[t.LM].copy();START_ROT=Rotation.from_quat(d.mocap_quat[t.LM][[1,2,3,0]])
    ROOT=t.ROOT+(m.body(target).pos-m.body('neighbor_9').pos)*[1,1,0]
    DISTANCES=(.080,.080,.030) if target=='neighbor_5' else (.080,.080,.080)
    HEIGHT=.080 if target=='neighbor_5' else .050
    if target=='neighbor_4':ROOT[1]-=.010


def restore():
    if str(STATE['mujoco_version'])!=mj.__version__ or len(STATE['state'])!=mj.mj_stateSize(m,int(STATE['spec'])):
        raise ValueError('Checkpoint runtime or dimension mismatch')
    mj.mj_setState(m,d,STATE['state'],int(STATE['spec']))


def command(seconds):
    elapsed=max(0.,min(seconds-START,58.))
    rotation=Rotation.from_quat(t.QUAT[[1,2,3,0]])
    if elapsed<3:
        pos=START_POS+[0,0,.050*elapsed/3];rotation=START_ROT;phase='rise'
    elif elapsed<7:
        f=(elapsed-3)/4;high=lane.push_command(0,ROOT,DISTANCES,HEIGHT)[0]
        pos=(START_POS+[0,0,.050])*(1-f)+high*f
        rotation=Slerp([0,1],Rotation.from_quat([START_ROT.as_quat(),rotation.as_quat()]))([f])[0];phase='change_target'
    else:pos,phase=lane.push_command(elapsed-7,ROOT,DISTANCES,HEIGHT)
    return pos,rotation.as_quat()[[3,0,1,2]],phase


def precheck():
    bad=[]
    for seconds in np.linspace(START,START+58,1161):
        restore();pos,quat,phase=command(float(seconds));d.qpos[t.LA:t.LA+3]=pos;d.qpos[t.LA+3:t.LA+7]=quat;mj.mj_forward(m,d)
        for c in d.contact:
            names=[m.geom(int(g)).name for g in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in names) and TARGET not in names:
                bad.append(dict(t=float(seconds),phase=phase,pair=names,distance_m=float(c.dist)))
    ctx=t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020
    q=CHECK['stages'][-1]['result']['final_joints_sdk'];stages=[]
    times=list(np.linspace(0,7,29)[1:])+[7+v for v in [.5,3,4,8,10,11,14,17,20,20.5,23,24,28,30,31,34,37,40,40.5,43,44,48,50,51]]
    if not bad:
        for elapsed in times:
            pos,quat,phase=command(START+elapsed);ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            r=t.routes.stage('left',ctx,q,phase,(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=START+elapsed,result=r))
            if not r['continuous']:break
            q=r['final_joints_sdk']
    return dict(passed=not bad and len(stages)==len(times) and all(s['result']['continuous'] for s in stages),
                target=TARGET,geometry_failures=bad,stages=stages,whole_arm_collision_checked=False)


def run(out):
    restore();mj.mj_forward(m,d)
    names=[m.body(i).name for i in range(m.nbody) if m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')]
    initial={n:d.xpos[m.body(n).id].copy() for n in names};shifts={n:0. for n in names if n!=TARGET}
    rows=[];bad=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(58/m.opt.timestep)):
            pos,quat,phase=command(float(d.time));d.mocap_pos[t.LM]=pos;d.mocap_quat[t.LM]=quat
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n in shifts:shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-initial[n])))
            contacts={n:[] for n in names}
            for i,c in enumerate(d.contact):
                ns=[m.geom(int(g)).name for g in c.geom]
                if any(n.startswith(('lh_','rh_')) for n in ns) and TARGET not in ns:
                    bad.append(dict(t=float(d.time),pair=ns,distance_m=float(c.dist)))
                    if c.dist<-.0005:abort='unexpected_hand_contact'
                force=np.zeros(6);mj.mj_contactForce(m,d,i,force);vector=c.frame.reshape(3,3).T@force[:3]
                for n in contacts:
                    if n in ns:contacts[n].append(dict(pair=ns,force_xyz_n=(vector*(1 if ns[1]==n else -1)).tolist()))
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                bricks={n:dict(position=d.xpos[m.body(n).id].tolist(),quaternion=d.xquat[m.body(n).id].tolist(),contacts=contacts[n]) for n in names}
                row=dict(t=float(d.time),phase=phase,bricks=bricks,qpos=d.qpos.tolist(),qvel=d.qvel.tolist());rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if abort:break
    summary=dict(target=TARGET,source=str(SOURCE),abort=abort,wall_seconds=time.monotonic()-started,
                 simulation_duration_s=float(d.time)-START,unexpected_hand_contacts=bad,other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},
                 target_staging_verified=False,full_base_row_verified=False,native_verified=False,official_region_verified=False,full_pyramid_verified=False)
    held=[r for r in rows if r['t']>=START+55-1e-8];metrics={}
    if held:
        for n in names:
            xyz=np.array([r['bricks'][n]['position'] for r in held]);q=np.array([r['bricks'][n]['quaternion'] for r in held]);rot=Rotation.from_quat(q[:,[1,2,3,0]])
            pts=np.concatenate([rr.apply(CORNERS)+p for rr,p in zip(rot,xyz)])
            metrics[n]=dict(final_position=xyz[-1].tolist(),yaw_degrees=float(rot[-1].as_euler('xyz',degrees=True)[2]),
                            drift_mm=float(np.linalg.norm(xyz-xyz[0],axis=1).max()*1000),angle_drift_degrees=float(np.rad2deg((rot[0].inv()*rot).magnitude().max())),
                            table_only=all(r['bricks'][n]['contacts'] and all(set(c['pair'])=={n,'table'} for c in r['bricks'][n]['contacts']) for r in held),
                            minimum_table_support_n=min(sum(c['force_xyz_n'][2] for c in r['bricks'][n]['contacts'] if 'table' in c['pair']) for r in held),
                            inside_analysis_rectangle=bool(np.all(pts[:,:2]>=t.REGION[:,0]) and np.all(pts[:,:2]<=t.REGION[:,1])))
        summary.update(bricks=metrics,held_duration_s=held[-1]['t']-held[0]['t'],target_displacement_mm=((np.array(metrics[TARGET]['final_position'])-initial[TARGET])*1000).tolist())
        # Like the preceding red/blue transfer, the staged obstacle may settle
        # farther aside through brick contact; the placed reds stay protected.
        protected={n:v for n,v in shifts.items() if not (TARGET=='neighbor_4' and n=='neighbor_5')}
        summary['placed_pair_preserved']=max(shifts[n] for n in ['neighbor_9','neighbor_6'])<=.002
        summary['target_staging_verified']=bool(not abort and not bad and max(protected.values())<=.002 and summary['held_duration_s']>=2.98
            and all(v['table_only'] and v['minimum_table_support_n']>=.8*m.body(n).mass*9.81 and v['drift_mm']<=2 and v['angle_drift_degrees']<=2 for n,v in metrics.items())
            and summary['target_displacement_mm'][0]>100 and all(metrics[n]['inside_analysis_rectangle'] for n in ['neighbor_9','neighbor_6'])
            and (TARGET=='neighbor_5' or metrics[TARGET]['inside_analysis_rectangle']))
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    restored=mj.MjData(m);mj.mj_setState(m,restored,state,spec);pos,quat,_=command(float(d.time))
    for data in [d,restored]:
        data.mocap_pos[t.LM]=pos;data.mocap_quat[t.LM]=quat;mj.mj_step(m,data);mj.mj_forward(m,data)
    one=np.empty_like(state);two=np.empty_like(state);mj.mj_getState(m,d,one,spec);mj.mj_getState(m,restored,two,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(one-two)));mj.mj_setState(m,d,state,spec)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));print(json.dumps(summary,indent=2))
    return summary['target_staging_verified'] and summary['checkpoint_next_step_maximum_difference']<=1e-10


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',choices=['neighbor_5','neighbor_4'],default='neighbor_5')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    configure(args.target);args.out.mkdir(parents=True,exist_ok=False);checked=precheck()
    (args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    print(json.dumps(dict(passed=checked['passed'],geometry_failures=len(checked['geometry_failures']),stages=len(checked['stages']))),flush=True)
    passed=checked['passed']
    if passed and args.run:passed=run(args.out)
    raise SystemExit(0 if passed else 1)
