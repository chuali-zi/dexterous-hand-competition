"""Move the first red from the retained blue-clearance state, locally.

Fixed offline teaching, continuous full-state continuation, no platform calls.
The south move clears the supply column before the west move and a final short
south correction. This candidate workcell is not a verified competition area.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
import try_blue_lane_from_zero as b

m,d,mj,p=b.m,b.d,b.mj,b.p
TARGET='neighbor_9';DURATION=23.;HOLD=20.


def configure(axis,source):
    record=json.loads((source/'result.json').read_text());check=json.loads((source/'precheck.json').read_text())
    state=np.load(source/'integration-state.npz',allow_pickle=False);summary=record['summary']
    safe_source=summary.get('local_blue_lane_clear',False) if axis=='south' else (
        summary.get('axis')==('south' if axis=='west' else 'west') and summary.get('held_duration_s',0)>=2.98 and summary.get('held_table_only',False)
        and summary.get('held_drift_mm',float('inf'))<=2 and summary.get('held_angular_drift_degrees',float('inf'))<=2
        and summary.get('final_face_tilt_degrees',float('inf'))<=2 and summary.get('minimum_table_support_n',0)>=.8*m.body(TARGET).mass*9.81
        and abs(summary.get('axis_error_mm',float('inf')))<=5 and not summary.get('unintended_contacts')
        and max(summary['other_bricks_max_shift_mm'].values())<=2)
    if not safe_source or summary['abort'] or not check['passed'] or summary['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source did not pass required physical and route checks')
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):
        raise ValueError('Checkpoint runtime mismatch')
    def restore():
        mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    restore()
    if not np.allclose(d.qpos,record['rows'][-1]['qpos'],atol=1e-12):raise ValueError('State and trace differ')
    seed=check['stages'][-1]['result']['final_joints_sdk']
    return restore,seed


def prepare(axis,source,restore,approach_height=.070,adapt_lateral_face=False,target_name=TARGET,goal_xy_m=None):
    restore();start=d.mocap_pos[p.t.LM].copy();startq=d.mocap_quat[p.t.LM].copy()
    target=d.xpos[m.body(target_name).id].copy();corners=Rotation.from_quat(d.xquat[m.body(target_name).id][[1,2,3,0]]).apply(p.CORNERS)+target
    normal=np.array([1.,0.,0.]) if axis=='west' else np.array([0.,1.,0.]);tangent=np.array([-normal[1],normal[0],0.])
    component=0 if axis=='west' else 1;destination=(goal_xy_m or [.300,-.040])[component];distance=float(target[component]-destination)
    if not .001<distance<.150:raise ValueError('Requested move outside this local trial')
    # Keep the south-facing wrist for the lateral push: either quarter-turn
    # route reaches joint 2's lower bound before the high approach endpoint.
    quat=(Rotation.from_euler('z',90,degrees=True)*Rotation.from_quat(p.t.QUAT[[1,2,3,0]])).as_quat()[[3,0,1,2]]
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=quat;mj.mj_forward(m,d)
    vertices=np.concatenate([b.a.placement.a.g.mesh_vertices('lh_'+n) for n in ['index_distal','index_proximal','middle_distal','middle_proximal']])
    root=normal*((corners@normal).max()-(vertices@normal).min()-.0005)
    root+=tangent*(target@tangent-((vertices@tangent).min()+(vertices@tangent).max())/2);root[2]=p.t.ROOT[2]
    adaptation=None;approach_offset=.020
    if adapt_lateral_face:
        if axis!='west':raise ValueError('Contact-face adaptation only applies to the lateral move')
        restore();reference=Rotation.from_quat(startq[[1,2,3,0]])
        d.qpos[p.t.LA:p.t.LA+3]=0;d.qpos[p.t.LA+3:p.t.LA+7]=[1,0,0,0];mj.mj_forward(m,d)
        local=np.concatenate([b.a.placement.a.g.mesh_vertices('lh_'+n) for n in ['index_distal','index_proximal','middle_distal','middle_proximal']])
        seed=np.array(json.loads((source/'precheck.json').read_text())['stages'][-1]['result']['final_joints_sdk'])
        bounds=p.t.routes.context()['left']['intersection']+[.02,-.02]
        goal=np.array([corners[:,0].max()-.0005,target[1],b.a.placement.a.g.FLOOR+.005])
        def metrics(q):
            h=b.forward('left',q);v=local@h[:3,:3].T+h[:3,3]
            return np.array([v[:,0].min(),(v[:,1].min()+v[:,1].max())/2,v[:,2].min()]),h
        def residual(q):
            values,h=metrics(q)
            return np.r_[(values-goal)/.0001,(reference.inv()*Rotation.from_matrix(h[:3,:3])).as_rotvec()*.3,(q-seed)*.01]
        fit=least_squares(residual,np.clip(seed,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0],bounds[:,1]),max_nfev=600)
        values,h=metrics(fit.x);rotation=Rotation.from_matrix(h[:3,:3]);change=float((reference.inv()*rotation).magnitude())
        if np.max(np.abs(values-goal))>1e-5 or change>np.deg2rad(25):raise ValueError('Adapted side face failed geometric constraints')
        root=h[:3,3];quat=rotation.as_quat()[[3,0,1,2]];approach_offset=0.
        adaptation=dict(optimizer_converged=bool(fit.success),geometric_error_mm=((values-goal)*1000).tolist(),
                        orientation_change_degrees=float(np.degrees(change)),contact_joints_sdk=fit.x.tolist(),
                        scope='Contact geometry candidate, not force or stability verification')
    if approach_height not in [.030,.050,.070]:raise ValueError('Unsupported local approach height')
    above=root+approach_offset*normal+[0,0,approach_height]
    turn=Slerp([0.,1.],Rotation.from_quat(np.array([startq,quat])[:,[1,2,3,0]]))
    def command(seconds):
        if seconds<2:pos=start+[0,0,.030*seconds/2];q=startq;phase='rise'
        elif seconds<6:
            f=(seconds-2)/4;pos=(start+[0,0,.030])*(1-f)+above*f;q=turn(f).as_quat()[[3,0,1,2]];phase='transfer_above'
        elif seconds<9:pos=root+approach_offset*normal+[0,0,approach_height*(9-seconds)/3];q=quat;phase='descend'
        elif seconds<10:pos=root+approach_offset*(10-seconds)*normal;q=quat;phase='approach'
        elif seconds<16:pos=root-distance*(seconds-10)/6*normal;q=quat;phase='push_red'
        elif seconds<18:pos=root-distance*normal+[0,0,.050*(seconds-16)/2];q=quat;phase='lift_clear'
        elif seconds<20:pos=root+(-distance+.020*(seconds-18)/2)*normal+[0,0,.050];q=quat;phase='retreat'
        else:pos=root+(-distance+.020)*normal+[0,0,.050];q=quat;phase='released_hold'
        return pos,q,phase
    restore()
    plan=dict(source=str(source),axis=axis,target=target_name,initial_target_center_m=target.tolist(),destination_axis_m=destination,
                        component=component,distance_m=distance,contact_root_m=root.tolist(),duration_s=DURATION,hold_start_s=HOLD,
                        approach_height_m=approach_height,lateral_face_adaptation=adaptation,local_tolerance_mm=5,whole_arm_collision_checked=False,official_region_verified=False)
    if goal_xy_m is not None:plan['goal_xy_m']=list(goal_xy_m)
    return command,plan


def precheck(command,restore,seed,target=TARGET,finger_command=None):
    started=time.monotonic();bad=[]
    for seconds in np.linspace(0,DURATION,461):
        restore();pos,q,phase=command(seconds);d.qpos[p.t.LA:p.t.LA+3]=pos;d.qpos[p.t.LA+3:p.t.LA+7]=q
        if finger_command is not None:
            for name,value in finger_command(seconds).items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and not (target in ns and any(n.startswith('lh_') for n in ns)):
                bad.append(dict(t=float(seconds),phase=phase,pair=ns,depth_mm=-float(c.dist)*1000))
    restore();ctx=p.t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.02,-.02];ctx['inset']=.02
    h=np.asarray(p.t.routes.sdk.forward_kinematics('left',seed))@ctx['mount'];pe=float(np.linalg.norm(h[:3,3]-d.mocap_pos[p.t.LM]))
    ae=float((Rotation.from_matrix(h[:3,:3]).inv()*Rotation.from_quat(d.mocap_quat[p.t.LM][[1,2,3,0]])).magnitude())
    stages=[];q=seed
    if not bad and pe<1e-5 and ae<1e-4:
        for seconds in np.linspace(0,DURATION,93)[1:]:
            pos,quat,phase=command(seconds);ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            result=p.t.routes.stage('left',ctx,q,phase,(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=float(seconds),result=result))
            if not result['continuous']:break
            q=result['final_joints_sdk']
    return dict(passed=not bad and len(stages)==92 and all(s['result']['continuous'] for s in stages),geometry_failures=bad,stages=stages,
                initial_seed_error_m=pe,initial_seed_angle_rad=ae,wall_seconds=time.monotonic()-started,whole_arm_collision_checked=False,native_verified=False)


def run(command,plan,restore,out,finger_command=None,protected_payload=None):
    target=plan.get('target',TARGET)
    restore();start=float(d.time);names=[m.body(i).name for i in range(m.nbody) if m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')]
    anchors={n:d.xpos[m.body(n).id].copy() for n in names};shifts={n:0. for n in names if n!=target};body=m.body(target).id
    rows=[];bad=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            pos,q,phase=command(float(d.time)-start);d.mocap_pos[p.t.LM]=pos;d.mocap_quat[p.t.LM]=q
            if finger_command is not None:
                for name,value in finger_command(float(d.time)-start).items():d.ctrl[m.actuator(name).id]=value
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n in shifts:
                shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-anchors[n])))
                if shifts[n]>.002:abort='other_brick_movement'
            contacts=[];protected_contacts=[]
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(i)).name for i in c.geom];hand=any(n.startswith(('lh_','rh_')) for n in ns)
                intended=target in ns and (any(n.startswith('lh_') for n in ns) or 'table' in ns)
                if (hand or target in ns) and not intended and c.dist<-.0005:
                    bad.append(dict(t=float(d.time)-start,pair=ns,depth_mm=-float(c.dist)*1000));abort='unexpected_contact'
                if target in ns:
                    force=np.zeros(6);mj.mj_contactForce(m,d,ci,force);vector=c.frame.reshape(3,3).T@force[:3]*(1 if ns[1]==target else -1)
                    contacts.append(dict(pair=ns,force_xyz_n=vector.tolist()))
                if protected_payload is not None and protected_payload in ns:
                    force=np.zeros(6);mj.mj_contactForce(m,d,ci,force);vector=c.frame.reshape(3,3).T@force[:3]*(1 if ns[1]==protected_payload else -1)
                    protected_contacts.append(dict(pair=ns,force_xyz_n=vector.tolist()))
            corners=Rotation.from_quat(d.xquat[body][[1,2,3,0]]).apply(p.CORNERS)+d.xpos[body]
            if corners[:,2].min()<b.a.placement.a.g.FLOOR-.010:abort='payload_below_table'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row=dict(elapsed_s=float(d.time)-start,phase=phase,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),contacts=contacts,
                         qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                if protected_payload is not None:row['protected_payload_contacts']=protected_contacts
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort:break
    summary=dict(target=target,axis=plan['axis'],source=plan['source'],abort=abort,duration_s=float(d.time)-start,cumulative_simulation_s=float(d.time),wall_seconds=time.monotonic()-started,
                 unintended_contacts=bad,other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},local_red_position_verified=False,
                 native_verified=False,official_region_verified=False,whole_arm_collision_checked=False,full_pyramid_verified=False)
    held=[r for r in rows if r['elapsed_s']>=HOLD-1e-8]
    if held:
        positions=np.array([r['position_m'] for r in held]);rot=Rotation.from_quat(np.array([r['quaternion_wxyz'] for r in held])[:,[1,2,3,0]])
        drift=float(np.linalg.norm(positions-positions[0],axis=1).max());angle=float(np.degrees((rot[0].inv()*rot).magnitude().max()))
        table=all(r['contacts'] and all('table' in c['pair'] for c in r['contacts']) for r in held)
        force=min(sum(c['force_xyz_n'][2] for c in r['contacts'] if 'table' in c['pair']) for r in held)
        face=float(np.degrees(np.arccos(np.clip(np.max(np.abs(rot[-1].as_matrix()[2,:])),0,1))))
        yaw=float((rot[-1].as_euler('xyz',degrees=True)[2]+45)%90-45);component=plan['component'];other=1-component
        error=float(positions[-1,component]-plan['destination_axis_m']);cross=float(positions[-1,other]-plan['initial_target_center_m'][other])
        summary.update(held_duration_s=held[-1]['elapsed_s']-held[0]['elapsed_s'],held_drift_mm=drift*1000,held_angular_drift_degrees=angle,
                       held_table_only=table,minimum_table_support_n=force,final_face_tilt_degrees=face,final_yaw_mod90_degrees=yaw,
                       final_position_m=positions[-1].tolist(),axis_error_mm=error*1000,cross_axis_shift_mm=cross*1000)
        goal_error=float(np.linalg.norm(positions[-1,:2]-plan.get('goal_xy_m',[.300,-.040])))
        summary['candidate_slot_xy_error_mm']=goal_error*1000
        position_ok=abs(error)<=.005 and abs(cross)<=.005 if plan['axis']=='south' and 'goal_xy_m' not in plan else goal_error<=.005
        summary['local_red_position_verified']=bool(not abort and not bad and summary['held_duration_s']>=2.98 and table and force>=.8*m.body(target).mass*9.81
                                                   and drift<=.002 and angle<=2 and face<=2 and abs(yaw)<=10 and position_ok)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    other=mj.MjData(m);mj.mj_setState(m,other,state,spec);x=np.empty_like(state);y=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,other);mj.mj_getState(m,d,x,spec);mj.mj_getState(m,other,y,spec);mj.mj_setState(m,d,state,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(x-y)))
    if summary['checkpoint_next_step_maximum_difference']>1e-10:summary['local_red_position_verified']=False
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--axis',choices=['south','west','finish-south'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--approach-height-mm',type=int,choices=[30,50,70],default=70)
    parser.add_argument('--adapt-lateral-face',action='store_true')
    args=parser.parse_args();restore,seed=configure(args.axis,args.source);command,plan=prepare(args.axis,args.source,restore,args.approach_height_mm/1000,args.adapt_lateral_face);args.out.mkdir(parents=True,exist_ok=False)
    checked=precheck(command,restore,seed);(args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),stages=len(checked['stages']),precheck_wall_seconds=checked['wall_seconds']);passed=checked['passed']
    if args.run and passed:brief['dynamics']=run(command,plan,restore,args.out);passed=brief['dynamics']['local_red_position_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
