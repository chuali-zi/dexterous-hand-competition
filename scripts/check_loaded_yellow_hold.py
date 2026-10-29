"""Continue an actual cupped pickup checkpoint with three seconds unchanged.

No root motion, reset, force change, platform call, or relaxed hold criterion.
The preceding acquisition's settling-window failure remains in its record.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_yellow_edge_acquisition as a

m,mj,d,p=a.m,a.mj,a.d,a.p


def run(source,out):
    old=json.loads((source/'result.json').read_text());s=old['summary'];last=old['rows'][-1]
    plan=json.loads((source/'plan.json').read_text());state=np.load(source/'integration-state.npz',allow_pickle=False)
    if (s['abort'] or s['unintended_hand_contacts'] or not plan.get('cup_before_lift') or s['duration_s']<28-1e-8
            or s['checkpoint_next_step_maximum_difference']>1e-10 or max(s['other_bricks_max_shift_mm'].values())>2
            or last['lowest_corner_m']<.005 or not last['contacts']
            or not all(any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in last['contacts'])):
        raise ValueError('Source did not end with a safe, raised payload on the hands')
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):
        raise ValueError('Checkpoint runtime mismatch')
    mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    start=float(d.time);body=m.body('neighbor_1').id
    others={m.body(i).name:d.xpos[i].copy() for i in range(m.nbody) if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and i!=body}
    shifts={n:0. for n in others};rows=[];bad=[];abort=None;started=time.monotonic()
    def record():
        contacts=[]
        for ci,c in enumerate(d.contact):
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and 'neighbor_1' not in ns and c.dist<-.0005:
                bad.append(dict(t=float(d.time),pair=ns,depth_mm=-float(c.dist)*1000))
            if 'neighbor_1' in ns:
                f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);v=c.frame.reshape(3,3).T@f[:3]*(1 if ns[1]=='neighbor_1' else -1)
                contacts.append(dict(pair=ns,force_xyz_n=v.tolist()))
        rot=Rotation.from_quat(d.xquat[body][[1,2,3,0]])
        row=dict(elapsed_s=float(d.time)-start,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),
                 lowest_corner_m=float((rot.apply(p.CORNERS)+d.xpos[body])[:,2].min()-a.g.FLOOR),contacts=contacts)
        rows.append(row);return row
    with (out/'trace.jsonl').open('x') as stream:
        stream.write(json.dumps(record(),allow_nan=False)+'\n')
        for step in range(round(3/m.opt.timestep)):
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,position in others.items():
                shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-position)))
                if shifts[n]>.002:abort='other_brick_moved'
            if step%10==9 or abort:
                row=record();stream.write(json.dumps(row,allow_nan=False)+'\n')
                if row['lowest_corner_m']<.005:abort='payload_not_clear'
                if bad:abort='unexpected_hand_contact'
            if abort:break
    xyz=np.array([r['position_m'] for r in rows]);rot=Rotation.from_quat(np.array([r['quaternion_wxyz'] for r in rows])[:,[1,2,3,0]])
    drift=float(np.linalg.norm(xyz-xyz[0],axis=1).max());angle=float(np.degrees((rot[0].inv()*rot).magnitude().max()))
    hand_only=all(r['contacts'] and all(any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in r['contacts']) for r in rows)
    right_only=all(r['contacts'] and all(any(n.startswith('rh_') for n in c['pair']) for c in r['contacts']) for r in rows)
    up=min(sum(c['force_xyz_n'][2] for c in r['contacts'] if any(n.startswith(('lh_','rh_')) for n in c['pair'])) for r in rows)
    summary=dict(source=str(source),start_time_s=start,duration_s=float(d.time)-start,wall_seconds=time.monotonic()-started,
                 abort=abort,unintended_hand_contacts=bad,held_minimum_bottom_mm=min(r['lowest_corner_m'] for r in rows)*1000,
                 held_drift_mm=drift*1000,held_angular_drift_degrees=angle,held_only_hand_contacts=hand_only,
                 held_only_right_hand_contacts=right_only,held_minimum_hand_up_force_n=up,
                 other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},native_verified=False,
                 source_sequence_continuity_verified=False,upper_placement_verified=False,whole_arm_collision_checked=False)
    spec=mj.mjtState.mjSTATE_INTEGRATION;saved=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,saved,spec)
    np.savez(out/'integration-state.npz',state=saved,spec=int(spec),mujoco_version=mj.__version__)
    other=mj.MjData(m);mj.mj_setState(m,other,saved,spec);x=np.empty_like(saved);y=np.empty_like(saved)
    mj.mj_step(m,d);mj.mj_step(m,other);mj.mj_getState(m,d,x,spec);mj.mj_getState(m,other,y,spec);mj.mj_setState(m,d,saved,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(x-y)))
    summary['local_loaded_hold_verified']=bool(not abort and not bad and summary['duration_s']>=3-1e-8
                    and summary['held_minimum_bottom_mm']>=5 and drift<=.002 and angle<=2 and hand_only
                    and up>=.8*m.body('neighbor_1').mass*9.81 and summary['checkpoint_next_step_maximum_difference']<=1e-10)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));print(json.dumps(summary,indent=2))
    return summary['local_loaded_hold_verified']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/upper-layer-93/cupped-receiver'))
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    raise SystemExit(0 if run(args.source,args.out) else 1)
