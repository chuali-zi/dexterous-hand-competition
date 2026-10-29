"""Move a supply yellow brick toward the existing bimanual work area.

Fixed local table push from the retained four-red checkpoint. No platform,
live object-state correction, material changes, or claim of upper placement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import try_taught_upper_tip as trial

g,p,m,mj,d=trial.g,trial.p,trial.m,trial.mj,trial.d
DURATION=26.
SOURCE=g.SOURCE
NORMAL=np.array([0.,1.,0.])
DISTANCE=.180
SEED=None


def configure_west():
    """Continue a safely settled partial result, without declaring staging done."""
    global SOURCE,NORMAL,DISTANCE,DURATION,SEED
    SOURCE=Path('runs/upper-layer-92/yellow-workcell')
    record=json.loads((SOURCE/'result.json').read_text());s=record['summary']
    checked=json.loads((SOURCE/'precheck.json').read_text())
    if (s['abort'] or not checked['passed'] or not s['released_table_only'] or s['released_duration_s']<2.98
            or s['released_drift_mm']>2 or max(s['other_bricks_max_shift_mm'].values())>2
            or s['minimum_table_support_n']<.8*m.body(trial.TARGET).mass*9.81
            or s['checkpoint_next_step_maximum_difference']>1e-10):
        raise ValueError('Partial source is not safely settled')
    state=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):
        raise ValueError('Checkpoint runtime or dimensions differ')
    def restore():
        mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    g.restore=restore
    NORMAL=np.array([1.,0.,0.]);DISTANCE=.080;DURATION=21.
    SEED=checked['stages'][-1]['result']['final_joints_sdk']


def prepare(center_contact=False):
    g.restore(); start=d.mocap_pos[p.t.LM].copy(); startq=d.mocap_quat[p.t.LM].copy()
    target=d.xpos[m.body(trial.TARGET).id].copy(); targetq=d.xquat[m.body(trial.TARGET).id].copy()
    points=Rotation.from_quat(targetq[[1,2,3,0]]).apply(p.CORNERS)+target
    angle=90 if NORMAL[1] else 0
    rotation=Rotation.from_euler('z',angle,degrees=True)*Rotation.from_quat(p.t.QUAT[[1,2,3,0]])
    quat=rotation.as_quat()[[3,0,1,2]]
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=quat
    mj.mj_forward(m,d)
    vertices=np.concatenate([g.mesh_vertices('lh_'+name) for name in ['index_proximal','index_distal','middle_proximal','middle_distal']])
    tangent=np.array([-NORMAL[1],NORMAL[0],0.])
    root=NORMAL*((points@NORMAL).max()-(vertices@NORMAL).min()-.0005)
    root+=tangent*(target@tangent-((vertices@tangent).min()+(vertices@tangent).max())/2)
    root[2]=p.t.ROOT[2]
    correction=0.
    if center_contact:
        d.qpos[p.t.LA:p.t.LA+3]=root;mj.mj_forward(m,d)
        touching=[c.pos.copy() for c in d.contact if set(m.geom(int(i)).name for i in c.geom)=={trial.TARGET,'lh_index_distal'}]
        if not touching:raise ValueError('Expected leading index contact is absent')
        correction=float(target[0]-np.mean(touching,axis=0)[0]);root[0]+=correction
    turn=Slerp([3,7],Rotation.from_quat(np.array([startq,quat])[:,[1,2,3,0]]))
    push_end=11+DISTANCE/.020
    def command(seconds):
        if seconds<3:pos=start+[0,0,.05*seconds/3];phase='rise'
        elif seconds<7:
            f=(seconds-3)/4;pos=(start+[0,0,.05])*(1-f)+(root+.02*NORMAL+[0,0,.07])*f;phase='turn'
        elif seconds<10:pos=root+.02*NORMAL+[0,0,.07*(10-seconds)/3];phase='descend'
        elif seconds<11:pos=root+.02*(11-seconds)*NORMAL;phase='approach'
        elif seconds<push_end:pos=root-.020*(seconds-11)*NORMAL;phase='push_to_workcell'
        elif seconds<push_end+1:pos=root+(-DISTANCE+.020*(seconds-push_end))*NORMAL;phase='retreat'
        elif seconds<push_end+3:pos=root+(-DISTANCE+.020)*NORMAL+[0,0,.050*(seconds-push_end-1)/2];phase='withdraw'
        else:pos=root+(-DISTANCE+.020)*NORMAL+[0,0,.050];phase='released'
        return pos,turn(np.clip(seconds,3,7)).as_quat()[[3,0,1,2]],phase
    g.restore()
    return command,dict(source=str(SOURCE),target=trial.TARGET,initial_center_m=target.tolist(),
                        contact_root_m=root.tolist(),desired_center_m=(target-DISTANCE*NORMAL).tolist(),
                        fixed_push_distance_m=DISTANCE,duration_s=DURATION,center_leading_contact=center_contact,
                        offline_contact_center_correction_x_m=correction)


def run(command,plan,out):
    summary=trial.run(command,out)
    data=json.loads((out/'result.json').read_text());rows=data['rows']
    held=[r for r in rows if r['elapsed_s']>=DURATION-3-1e-8]
    summary['yellow_staged_for_workcell']=False
    if held:
        positions=np.array([r['position_m'] for r in held])
        yaw=Rotation.from_quat(np.array(held[-1]['quaternion_wxyz'])[[1,2,3,0]]).as_euler('xyz',degrees=True)[2]
        table_only=all(r['contacts'] and all(c['pair']==['table',trial.TARGET] for c in r['contacts']) for r in held)
        minimum_support=min(sum(c['force_xyz_n'][2] for c in r['contacts'] if 'table' in c['pair']) for r in held)
        error=np.array(held[-1]['position_m'])-plan['desired_center_m']
        drift=float(np.linalg.norm(positions-positions[0],axis=1).max())
        summary.update(final_center_error_mm=(error*1000).tolist(),final_yaw_degrees=float(yaw),
                       released_duration_s=held[-1]['elapsed_s']-held[0]['elapsed_s'],released_drift_mm=drift*1000,
                       released_table_only=table_only,minimum_table_support_n=minimum_support)
        tangent=np.array([-NORMAL[1],NORMAL[0],0.])
        summary['yellow_staged_for_workcell']=bool(not summary['abort'] and abs(error@tangent)<=.005 and abs(error@NORMAL)<=.010
                    and abs(yaw)<=10 and summary['released_duration_s']>=2.98 and drift<=.002 and table_only
                    and minimum_support>=.8*m.body(trial.TARGET).mass*9.81)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    restored=mj.MjData(m);mj.mj_setState(m,restored,state,spec)
    a=np.empty_like(state);b=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,restored);mj.mj_getState(m,d,a,spec);mj.mj_getState(m,restored,b,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(a-b)))
    mj.mj_setState(m,d,state,spec)
    summary['yellow_staged_for_workcell'] &= summary['checkpoint_next_step_maximum_difference']<=1e-10
    data['summary']=summary;(out/'result.json').write_text(json.dumps(data,allow_nan=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='store_true');parser.add_argument('--center-contact',action='store_true')
    parser.add_argument('--west-from-partial',action='store_true',help='Continue the safely settled, off-target first south push')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.west_from_partial:
        if args.center_contact:raise ValueError('The contact-centering comparison applies only to the south push')
        configure_west()
    trial.DURATION=DURATION
    command,plan=prepare(args.center_contact);checked=trial.precheck(command,SEED)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False))
    (args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief={k:v for k,v in checked.items() if k not in ['geometry_failures','stages']}
    brief.update(geometry_failure_count=len(checked['geometry_failures']),stage_count=len(checked['stages']))
    passed=checked['passed']
    if args.run and passed:
        brief['dynamics']=run(command,plan,args.out);passed=brief['dynamics']['yellow_staged_for_workcell']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
