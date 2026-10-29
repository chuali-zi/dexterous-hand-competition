"""Move the blocking blue brick using a fresh local SDK-zero start.

Original ten brick poses are retained. Both hand roots start at SDK joint zero;
this is not a verified native initial state or a complete arm collision model.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import align_released_yellow as a

m,d,mj,p=a.m,a.d,a.mj,a.p
TARGET='neighbor_7';DURATION=27.;CTX=p.t.routes.context()


def forward(side,q):return np.asarray(p.t.routes.sdk.forward_kinematics(side,list(q)))@CTX[side]['mount']


def initialize():
    mj.mj_resetData(m,d)
    for side,address,index in [('left',p.t.LA,p.t.LM),('right',p.t.RA,p.t.RM)]:
        h=forward(side,np.zeros(7));quat=Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]]
        d.qpos[address:address+3]=h[:3,3];d.qpos[address+3:address+7]=quat
        d.mocap_pos[index]=h[:3,3];d.mocap_quat[index]=quat
    mj.mj_forward(m,d)


def prepare():
    initialize();blue=d.xpos[m.body(TARGET).id].copy()
    corners=Rotation.from_quat(d.xquat[m.body(TARGET).id][[1,2,3,0]]).apply(p.CORNERS)+blue
    goals=p.t.CANDIDATE['hand_targets'].copy()
    for n,v in goals.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=p.t.QUAT;mj.mj_forward(m,d)
    vertices=np.concatenate([a.placement.a.g.mesh_vertices('lh_'+n) for n in ['index_proximal','index_distal','middle_proximal','middle_distal']])
    root=np.array([corners[:,0].min()-vertices[:,0].max()+.0005,blue[1]-(vertices[:,1].min()+vertices[:,1].max())/2,p.t.ROOT[2]])
    above=root+[-.020,0,.080];rotation=Rotation.from_quat(p.t.QUAT[[1,2,3,0]]).as_matrix();bounds=CTX['left']['intersection']+[.02,-.02]
    def residual(q):
        h=forward('left',q)
        return np.r_[(h[:3,3]-above)/1e-5,Rotation.from_matrix(rotation.T@h[:3,:3]).as_rotvec()/1e-4,q*.01]
    solved=least_squares(residual,np.clip(CTX['left']['seed'],bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=(bounds[:,0],bounds[:,1]),max_nfev=500)
    h=forward('left',solved.x);pe=float(np.linalg.norm(h[:3,3]-above));ae=float(Rotation.from_matrix(rotation.T@h[:3,:3]).magnitude())
    if pe>=1e-5 or ae>=1e-4:raise ValueError('No accepted high approach endpoint')
    outside=np.zeros(7);outside[0]=.5;bent=solved.x.copy();bent[0]=.5
    waypoints=np.array([np.zeros(7),outside,bent,solved.x])
    if np.any(waypoints<bounds[:,0]) or np.any(waypoints>bounds[:,1]):raise ValueError('Entry waypoint violates limits')
    right=forward('right',np.zeros(7));right_pose=(right[:3,3],Rotation.from_matrix(right[:3,:3]).as_quat()[[3,0,1,2]])
    def entry_q(seconds):
        if seconds<2:return waypoints[0]
        if seconds<5:return waypoints[0]+(waypoints[1]-waypoints[0])*(seconds-2)/3
        if seconds<9:return waypoints[1]+(waypoints[2]-waypoints[1])*(seconds-5)/4
        if seconds<13:return waypoints[2]+(waypoints[3]-waypoints[2])*(seconds-9)/4
        return waypoints[3]
    def command(seconds):
        if seconds<13:
            h=forward('left',entry_q(seconds));left=h[:3,3];quat=Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]];phase='entry' if seconds>=2 else 'settle_and_fold'
        else:
            quat=p.t.QUAT
            if seconds<16:left=root+[-.020,0,.080*(16-seconds)/3];phase='descend'
            elif seconds<17:left=root+[-.020*(17-seconds),0,0];phase='approach'
            elif seconds<21:left=root+[.020*(seconds-17),0,0];phase='push_blue'
            elif seconds<23:left=root+[.080,0,.050*(seconds-21)/2];phase='lift_clear'
            elif seconds<24:left=root+[.080-.020*(seconds-23),0,.050];phase='retreat'
            else:left=root+[.060,0,.050];phase='released_hold'
        return dict(left=(left,quat),right=right_pose),{n:v*np.clip(seconds/2,0,1) for n,v in goals.items()},phase
    plan=dict(target=TARGET,initial_target_center_m=blue.tolist(),contact_root_m=root.tolist(),above_root_m=above.tolist(),
              entry_waypoints_sdk=waypoints.tolist(),endpoint_position_error_m=pe,endpoint_angle_error_rad=ae,
              push_distance_m=.080,clearance_minimum_x_m=.350,hold_start_s=24.,duration_s=DURATION,
              initial_state='Retained original bricks and SDK-zero hand roots; zero initial controls',native_initial_pose_verified=False)
    initialize();return command,plan,entry_q


def precheck(command,plan,entry_q):
    started=time.monotonic();bad=[]
    for seconds in np.linspace(0,DURATION,541):
        initialize();poses,ctrl,phase=command(seconds)
        for side,address in [('left',p.t.LA),('right',p.t.RA)]:
            pos,q=poses[side];d.qpos[address:address+3]=pos;d.qpos[address+3:address+7]=q
        for n,v in ctrl.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
        mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and not (TARGET in ns and any(n.startswith('lh_') for n in ns)):
                bad.append(dict(t=float(seconds),phase=phase,pair=ns,depth_mm=-float(c.dist)*1000))
    ctx=p.t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.02,-.02];ctx['inset']=.02
    stages=[];q=plan['entry_waypoints_sdk'][-1]
    if not bad:
        for seconds in np.linspace(13,27,57)[1:]:
            pos,quat=command(seconds)[0]['left'];ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            result=p.t.routes.stage('left',ctx,q,'blue_lane',(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=float(seconds),result=result))
            if not result['continuous']:break
            q=result['final_joints_sdk']
    entry=np.array([entry_q(t) for t in np.linspace(0,13,261)]);bounds=ctx['intersection']
    return dict(passed=not bad and len(stages)==56 and all(s['result']['continuous'] for s in stages),geometry_failures=bad,stages=stages,
                entry_minimum_margin_rad=float(np.minimum(entry-bounds[:,0],bounds[:,1]-entry).min()),
                entry_maximum_joint_speed_rad_s=float(np.max(np.abs(np.diff(entry,axis=0)))/.05),
                wall_seconds=time.monotonic()-started,whole_arm_collision_checked=False,native_verified=False)


def run(command,plan,out):
    initialize();body=m.body(TARGET).id;names=[m.body(i).name for i in range(m.nbody) if m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')]
    initial={n:d.xpos[m.body(n).id].copy() for n in names};settled=None;shifts={n:0. for n in names if n!=TARGET};horizontal=shifts.copy()
    bad=[];rows=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            poses,ctrl,phase=command(float(d.time))
            for side,index in [('left',p.t.LM),('right',p.t.RM)]:d.mocap_pos[index],d.mocap_quat[index]=poses[side]
            for n,v in ctrl.items():d.ctrl[m.actuator(n).id]=v
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n in shifts:
                pos=d.xpos[m.body(n).id];horizontal[n]=max(horizontal[n],float(np.linalg.norm(pos[:2]-initial[n][:2])))
                if horizontal[n]>.002:abort='other_brick_horizontal_movement'
                if settled is None:
                    if not -.003<=pos[2]-initial[n][2]<=.001:abort='unexpected_initial_settling'
                else:
                    shifts[n]=max(shifts[n],float(np.linalg.norm(pos-settled[n])))
                    if shifts[n]>.002:abort='other_brick_movement_after_settling'
            if settled is None and d.time>=2-1e-8:settled={n:d.xpos[m.body(n).id].copy() for n in names}
            contacts=[]
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(i)).name for i in c.geom];hand=any(n.startswith(('lh_','rh_')) for n in ns)
                intended=TARGET in ns and any(n.startswith('lh_') for n in ns)
                if hand and not intended and c.dist<-.0005:
                    bad.append(dict(t=float(d.time),pair=ns,depth_mm=-float(c.dist)*1000));abort='unexpected_hand_contact'
                if TARGET in ns:
                    if not intended and 'table' not in ns and c.dist<-.0005:abort='unexpected_payload_contact'
                    f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);v=c.frame.reshape(3,3).T@f[:3]*(1 if ns[1]==TARGET else -1)
                    contacts.append(dict(pair=ns,force_xyz_n=v.tolist()))
            corners=Rotation.from_quat(d.xquat[body][[1,2,3,0]]).apply(p.CORNERS)+d.xpos[body]
            if corners[:,2].min()<a.placement.a.g.FLOOR-.010:abort='payload_below_table'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row=dict(elapsed_s=float(d.time),phase=phase,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),
                         contacts=contacts,lowest_corner_m=float(corners[:,2].min()-a.placement.a.g.FLOOR),minimum_x_m=float(corners[:,0].min()),
                         qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort:break
    summary=dict(abort=abort,duration_s=float(d.time),wall_seconds=time.monotonic()-started,unintended_contacts=bad,
                 initial_positions_m={n:v.tolist() for n,v in initial.items()},settled_positions_m={n:v.tolist() for n,v in (settled or {}).items()},
                 other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},other_bricks_max_horizontal_shift_mm={n:v*1000 for n,v in horizontal.items()},
                 local_blue_lane_clear=False,native_initial_pose_verified=False,native_verified=False,whole_arm_collision_checked=False,
                 official_region_verified=False,full_pyramid_verified=False)
    held=[r for r in rows if r['elapsed_s']>=24-1e-8]
    if held:
        positions=np.array([r['position_m'] for r in held]);rot=Rotation.from_quat(np.array([r['quaternion_wxyz'] for r in held])[:,[1,2,3,0]])
        drift=float(np.linalg.norm(positions-positions[0],axis=1).max());angle=float(np.degrees((rot[0].inv()*rot).magnitude().max()))
        only_table=all(r['contacts'] and all('table' in c['pair'] for c in r['contacts']) for r in held)
        support=min(sum(c['force_xyz_n'][2] for c in r['contacts'] if 'table' in c['pair']) for r in held)
        face=float(np.degrees(np.arccos(np.clip(np.max(np.abs(rot[-1].as_matrix()[2,:])),0,1))))
        summary.update(held_duration_s=held[-1]['elapsed_s']-held[0]['elapsed_s'],held_drift_mm=drift*1000,held_angular_drift_degrees=angle,
                       held_table_only=only_table,minimum_table_support_n=support,final_minimum_x_m=held[-1]['minimum_x_m'],final_face_tilt_degrees=face)
        summary['local_blue_lane_clear']=bool(not abort and not bad and summary['held_duration_s']>=2.98 and only_table and support>=.8*m.body(TARGET).mass*9.81
                                              and drift<=.002 and angle<=2 and face<=2 and held[-1]['minimum_x_m']>=plan['clearance_minimum_x_m'])
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    other=mj.MjData(m);mj.mj_setState(m,other,state,spec);x=np.empty_like(state);y=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,other);mj.mj_getState(m,d,x,spec);mj.mj_getState(m,other,y,spec);mj.mj_setState(m,d,state,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(x-y)))
    if summary['checkpoint_next_step_maximum_difference']>1e-10:summary['local_blue_lane_clear']=False
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);command,plan,entry_q=prepare();checked=precheck(command,plan,entry_q)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),stages=len(checked['stages']),precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if args.run and passed:brief['dynamics']=run(command,plan,args.out);passed=brief['dynamics']['local_blue_lane_clear']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
