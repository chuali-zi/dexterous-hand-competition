"""Fixed short alignment of an actually released yellow on the red base.

Two explicit offline-taught stages, preserving all scene state. Protect other
bricks against their positions before the release, not freshly reset anchors.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import try_guided_yellow_placement as placement

m,mj,d,p=placement.m,placement.mj,placement.d,placement.p
TARGET=placement.TARGET;SUPPORTS=placement.SUPPORTS;DURATION=14.


def configure(axis):
    source=Path('runs/upper-layer-94/guided-release' if axis=='y' else 'runs/upper-layer-94/align-y')
    record=json.loads((source/'result.json').read_text());s=record['summary']
    check=json.loads((source/'precheck.json').read_text());state=np.load(source/'integration-state.npz',allow_pickle=False)
    if (s['abort'] or s['unintended_contacts'] or not s['held_only_base_contacts'] or s['held_duration_s']<2.98
            or s['held_drift_mm']>2 or s['held_angular_drift_degrees']>2 or s['final_cube_face_tilt_degrees']>2
            or s['final_bottom_mm']<45 or max(s['other_bricks_max_shift_mm'].values())>2
            or s['checkpoint_next_step_maximum_difference']>1e-10 or not check['passed']):
        raise ValueError('Source is not a safely released, near-flat placement')
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):
        raise ValueError('Checkpoint runtime mismatch')
    def restore():
        mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    seed=check['stages']['left'][-1]['result']['final_joints_sdk']
    original=np.load('runs/upper-layer-93/loaded-transfer/integration-state.npz',allow_pickle=False)
    temp=mj.MjData(m);mj.mj_setState(m,temp,original['state'],int(original['spec']));mj.mj_forward(m,temp)
    anchors={m.body(i).name:temp.xpos[i].copy() for i in range(m.nbody) if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and m.body(i).name!=TARGET}
    return source,restore,seed,anchors


def prepare(axis,source,restore):
    restore();start=d.mocap_pos[p.t.LM].copy();startq=d.mocap_quat[p.t.LM].copy()
    target=d.xpos[m.body(TARGET).id].copy();q=d.xquat[m.body(TARGET).id].copy();points=Rotation.from_quat(q[[1,2,3,0]]).apply(p.CORNERS)+target
    pair=np.mean([d.xpos[m.body(n).id] for n in SUPPORTS],axis=0);component=0 if axis=='x' else 1
    error=target[component]-pair[component];distance=abs(float(error))
    if not .001<=distance<=.040:raise ValueError('This short correction requires an axis error between 1 and 40 mm')
    normal=np.zeros(3);normal[component]=np.sign(error);tangent=np.array([-normal[1],normal[0],0.])
    rotation=Rotation.from_euler('z',90 if axis=='y' else 0,degrees=True)*Rotation.from_quat(p.t.QUAT[[1,2,3,0]])
    quat=rotation.as_quat()[[3,0,1,2]]
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]+.050];d.qpos[p.t.LA+3:p.t.LA+7]=quat;mj.mj_forward(m,d)
    vertices=np.concatenate([placement.a.g.mesh_vertices('lh_'+n) for n in ['index_distal','index_proximal','middle_distal','middle_proximal']])
    root=normal*((points@normal).max()-(vertices@normal).min()-.0005)
    root+=tangent*(target@tangent-((vertices@tangent).min()+(vertices@tangent).max())/2);root[2]=p.t.ROOT[2]+.050
    turn=Slerp([0.,2.],Rotation.from_quat(np.array([startq,quat])[:,[1,2,3,0]]))
    def command(seconds):
        if seconds<2:
            f=seconds/2;pos=start*(1-f)+(root+.020*normal+[0,0,.050])*f;phase='above_target'
        elif seconds<4:pos=root+.020*normal+[0,0,.050*(4-seconds)/2];phase='descend'
        elif seconds<5:pos=root+.020*(5-seconds)*normal;phase='approach'
        elif seconds<8:pos=root-distance*(seconds-5)/3*normal;phase='align_upper'
        elif seconds<9:pos=root+(-distance+.020*(seconds-8))*normal;phase='retreat'
        elif seconds<11:pos=root+(-distance+.020)*normal+[0,0,.050*(seconds-9)/2];phase='withdraw'
        else:pos=root+(-distance+.020)*normal+[0,0,.050];phase='released_hold'
        return pos,turn(np.clip(seconds,0,2)).as_quat()[[3,0,1,2]],phase
    restore()
    return command,dict(source=str(source),axis=axis,contact_root_m=root.tolist(),normal=normal.tolist(),distance_m=distance,
                        pair_center_m=pair.tolist(),initial_target_center_m=target.tolist(),
                        protected_reference='runs/upper-layer-93/loaded-transfer',native_verified=False,whole_arm_collision_checked=False)


def precheck(command,restore,seed):
    started=time.monotonic();bad=[]
    for seconds in np.linspace(0,DURATION,281):
        restore();pos,q,phase=command(seconds);d.qpos[p.t.LA:p.t.LA+3]=pos;d.qpos[p.t.LA+3:p.t.LA+7]=q;mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and TARGET not in ns:bad.append(dict(t=float(seconds),pair=ns,depth_mm=-float(c.dist)*1000))
    restore();ctx=p.t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020
    actual=np.asarray(p.t.routes.sdk.forward_kinematics('left',seed))@ctx['mount']
    initial_error=float(np.linalg.norm(actual[:3,3]-d.mocap_pos[p.t.LM]));initial_angle=float((Rotation.from_matrix(actual[:3,:3]).inv()*Rotation.from_quat(d.mocap_quat[p.t.LM][[1,2,3,0]])).magnitude())
    stages=[];q=seed
    if not bad and initial_error<1e-5 and initial_angle<1e-4:
        for seconds in np.linspace(0,DURATION,57)[1:]:
            pos,quat,phase=command(seconds);ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            result=p.t.routes.stage('left',ctx,q,phase,(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=float(seconds),result=result))
            if not result['continuous']:break
            q=result['final_joints_sdk']
    return dict(passed=not bad and len(stages)==56 and all(x['result']['continuous'] for x in stages),geometry_failures=bad,
                stages=dict(left=stages),initial_seed_error_m=initial_error,initial_seed_angle_rad=initial_angle,wall_seconds=time.monotonic()-started,whole_arm_collision_checked=False)


def run(command,plan,restore,anchors,out):
    restore();start=float(d.time);body=m.body(TARGET).id;shifts={n:0. for n in anchors};bad=[];rows=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            pos,q,phase=command(float(d.time)-start);d.mocap_pos[p.t.LM]=pos;d.mocap_quat[p.t.LM]=q
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,position in anchors.items():
                shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-position)))
                if shifts[n]>.002:abort='cumulative_base_or_neighbor_movement'
            contacts=[]
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(i)).name for i in c.geom];hand=any(n.startswith(('lh_','rh_')) for n in ns)
                intended=TARGET in ns and (hand or any(n in SUPPORTS for n in ns))
                if (hand or TARGET in ns) and not intended and c.dist<-.0005:
                    bad.append(dict(t=float(d.time)-start,pair=ns,depth_mm=-float(c.dist)*1000));abort='unexpected_contact'
                if TARGET in ns:
                    f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);v=c.frame.reshape(3,3).T@f[:3]*(1 if ns[1]==TARGET else -1);contacts.append(dict(pair=ns,force_xyz_n=v.tolist()))
            rot=Rotation.from_quat(d.xquat[body][[1,2,3,0]]);bottom=float((rot.apply(p.CORNERS)+d.xpos[body])[:,2].min()-placement.a.g.FLOOR)
            if bottom<.040:abort='payload_below_base_top'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row=dict(elapsed_s=float(d.time)-start,phase=phase,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),lowest_corner_m=bottom,
                         contacts=contacts,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),support_positions_m={n:d.xpos[m.body(n).id].tolist() for n in SUPPORTS})
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort:break
    summary=dict(axis=plan['axis'],abort=abort,duration_s=float(d.time)-start,wall_seconds=time.monotonic()-started,unintended_contacts=bad,
                 other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},native_verified=False,whole_arm_collision_checked=False,
                 full_pyramid_verified=False,official_region_verified=False,source_sequence_continuity_verified=False)
    measured_plan=dict(plan,pair_center_m=np.mean([rows[-1]['support_positions_m'][n] for n in SUPPORTS],axis=0).tolist())
    placement.evaluate(summary,rows,measured_plan,hold_start=11.)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    other=mj.MjData(m);mj.mj_setState(m,other,state,spec);x=np.empty_like(state);y=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,other);mj.mj_getState(m,d,x,spec);mj.mj_getState(m,other,y,spec);mj.mj_setState(m,d,state,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(x-y)))
    if summary['checkpoint_next_step_maximum_difference']>1e-10:summary['local_released_bridge_verified']=summary['local_first_yellow_aligned']=False
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--axis',choices=['x','y'],required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source,restore,seed,anchors=configure(args.axis);command,plan=prepare(args.axis,source,restore);checked=precheck(command,restore,seed)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),stage_count=len(checked['stages']['left']),precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if args.run and passed:brief['dynamics']=run(command,plan,restore,anchors,args.out);passed=brief['dynamics']['local_first_yellow_aligned']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
