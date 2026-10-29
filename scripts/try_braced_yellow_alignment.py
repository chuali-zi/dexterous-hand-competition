"""Local fixed bimanual alignment: brace red 9 before pushing yellow south.

Continue the released checkpoint, retaining cumulative base movement limits.
Offline geometry teaching is not online object-state feedback or native proof.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
import align_released_yellow as a

m,d,mj,p=a.m,a.d,a.mj,a.p
TARGET=a.TARGET;SUPPORTS=a.SUPPORTS;DURATION=30.
# A geometry-constrained numerical endpoint, independently checked below. The
# exploratory optimizer exhausted its evaluations; this is not an optimum.
BRACE_SEED=np.array([.45820395748931375,-.1192452160326622,-.7737702159633929,
                     -.6323416810622998,1.56818965354406,.4429748232148613,.14832284943224056])


def prepare(preload_mm=0.):
    if preload_mm not in (0.,.4):raise ValueError('Only the measured zero-gap and 0.4 mm preload comparisons are supported')
    source,restore,left_seed,anchors=a.configure('y');restore()
    check=json.loads((source/'precheck.json').read_text())
    right_seed=np.array(check['stages']['right'][-1]['result']['final_joints_sdk'])
    initial_ctrl={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)}
    goals={n.replace('lh_','rh_'):v for n,v in p.t.CANDIDATE['hand_targets'].items()}
    ctx=p.t.routes.context()['right'];ctx['limits']=ctx['intersection']+[.02,-.02];ctx['inset']=.02
    def forward(q):return np.asarray(p.t.routes.sdk.forward_kinematics('right',list(q)))@ctx['mount']
    start=forward(right_seed)
    if np.linalg.norm(start[:3,3]-d.mocap_pos[p.t.RM])>1e-5 or (Rotation.from_matrix(start[:3,:3]).inv()*Rotation.from_quat(d.mocap_quat[p.t.RM][[1,2,3,0]])).magnitude()>1e-4:
        raise ValueError('Right source seed does not match checkpoint')
    pose=forward(BRACE_SEED);quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
    for n,v in goals.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
    d.qpos[p.t.RA:p.t.RA+3]=pose[:3,3];d.qpos[p.t.RA+3:p.t.RA+7]=quat;mj.mj_forward(m,d)
    points=np.zeros(6);mj.mj_geomDistance(m,d,m.geom('rh_middle_distal').id,m.geom('neighbor_9').id,1.,points)
    center=d.xpos[m.body('neighbor_9').id].copy();contact=(points[:3]+points[3:])/2
    root=pose[:3,3].copy();root[[0,2]]+=center[[0,2]]-contact[[0,2]]
    def clearance(delta):
        d.qpos[p.t.RA:p.t.RA+3]=root+[0,delta,0];mj.mj_forward(m,d)
        return mj.mj_geomDistance(m,d,m.geom('rh_middle_distal').id,m.geom('neighbor_9').id,1.,None)
    root[1]+=brentq(clearance,-.005,.005,xtol=1e-10)
    root[1]+=preload_mm/1000
    ctx['root'][:3,3]=pose[:3,3];ctx['root'][:3,:3]=pose[:3,:3]
    centered=p.t.routes.stage('right',ctx,BRACE_SEED.tolist(),'center_brace',(root-pose[:3,3]).tolist())
    if not centered['continuous']:raise ValueError('Centered brace endpoint is unreachable')
    ctx['root'][:3,3]=root
    backoff=p.t.routes.stage('right',ctx,centered['final_joints_sdk'],'south_backoff',[0,-.04,0])
    if not backoff['continuous']:raise ValueError('Brace backoff is unreachable')
    waypoints=np.array([right_seed,backoff['final_joints_sdk'],centered['final_joints_sdk']])
    if np.any(waypoints<ctx['limits'][:,0]) or np.any(waypoints>ctx['limits'][:,1]):raise ValueError('Right waypoint violates limits')
    left_command,plan=a.prepare('y',source,restore)
    def right_q(seconds):
        if seconds<2:return waypoints[0]
        if seconds<7:return waypoints[0]+(waypoints[1]-waypoints[0])*(seconds-2)/5
        if seconds<10:return waypoints[1]+(waypoints[2]-waypoints[1])*(seconds-7)/3
        if seconds<21:return waypoints[2]
        if seconds<24:return waypoints[2]+(waypoints[1]-waypoints[2])*(seconds-21)/3
        if seconds<27:return waypoints[1]+(waypoints[0]-waypoints[1])*(seconds-24)/3
        return waypoints[0]
    def command(seconds):
        left,lq,phase=left_command(np.clip(seconds-10,0,14));h=forward(right_q(seconds))
        poses=dict(left=(left,lq),right=(h[:3,3],Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]]))
        ctrl={n:initial_ctrl[n]+(v-initial_ctrl[n])*np.clip(seconds/2,0,1) for n,v in goals.items()}
        if seconds<10:phase='brace_entry'
        elif seconds>=27:phase='released_hold'
        elif seconds>=21:phase='brace_withdraw'
        return poses,ctrl,phase
    plan.update(braced_body='neighbor_9',brace_preload_mm=preload_mm,brace_root_m=root.tolist(),right_waypoints_sdk=waypoints.tolist(),
                brace_endpoint_checks=dict(centered=centered,backoff=backoff),hold_start_s=27.,duration_s=DURATION)
    restore();return command,plan,restore,left_seed,anchors,right_q


def intended(pair):
    if TARGET in pair:return any(n.startswith('lh_') or n in SUPPORTS for n in pair if n!=TARGET)
    return 'neighbor_9' in pair and any(n.startswith('rh_') for n in pair)


def precheck(command,restore,left_seed,right_q):
    started=time.monotonic();bad=[]
    for seconds in np.linspace(0,DURATION,601):
        restore();poses,ctrl,phase=command(seconds)
        for side,address in [('left',p.t.LA),('right',p.t.RA)]:
            pos,q=poses[side];d.qpos[address:address+3]=pos;d.qpos[address+3:address+7]=q
        for n,v in ctrl.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
        mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            # The frozen yellow occupies its pre-push position: penetration by
            # its intended pusher here is not a prediction of dynamic overlap.
            if any(n.startswith(('lh_','rh_')) for n in ns) and (not intended(ns) or (TARGET not in ns and c.dist<-.0005)):
                bad.append(dict(t=float(seconds),phase=phase,pair=ns,depth_mm=-float(c.dist)*1000))
    restore();ctx=p.t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.02,-.02];ctx['inset']=.02
    stages=[];q=left_seed
    if not bad:
        for seconds in np.linspace(10,21,45)[1:]:
            pos,quat=command(seconds)[0]['left'];ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            checked=p.t.routes.stage('left',ctx,q,'align_upper',(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=float(seconds),result=checked))
            if not checked['continuous']:break
            q=checked['final_joints_sdk']
    qs=np.array([right_q(t) for t in np.linspace(0,DURATION,601)]);bounds=p.t.routes.context()['right']['intersection']
    return dict(passed=not bad and len(stages)==44 and all(x['result']['continuous'] for x in stages),geometry_failures=bad,
                stages=dict(left=stages),right_minimum_margin_rad=float(np.minimum(qs-bounds[:,0],bounds[:,1]-qs).min()),
                right_maximum_joint_speed_rad_s=float(np.max(np.abs(np.diff(qs,axis=0)))/.05),
                wall_seconds=time.monotonic()-started,whole_arm_collision_checked=False,scope='Hand geometry with frozen bricks; fixed right joint routes and continuous left numerical route')


def run(command,plan,restore,anchors,out):
    restore();start=float(d.time);body=m.body(TARGET).id;shifts={n:0. for n in anchors};rows=[];bad=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            poses,ctrl,phase=command(float(d.time)-start)
            for side,index in [('left',p.t.LM),('right',p.t.RM)]:d.mocap_pos[index],d.mocap_quat[index]=poses[side]
            for n,v in ctrl.items():d.ctrl[m.actuator(n).id]=v
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,position in anchors.items():
                shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-position)))
                if shifts[n]>.002:abort='cumulative_base_or_neighbor_movement'
            contacts=[];brace_contacts=[]
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(i)).name for i in c.geom];hand=any(n.startswith(('lh_','rh_')) for n in ns)
                if (hand or TARGET in ns) and not intended(ns) and c.dist<-.0005:
                    bad.append(dict(t=float(d.time)-start,pair=ns,depth_mm=-float(c.dist)*1000));abort='unexpected_contact'
                observed=TARGET if TARGET in ns else ('neighbor_9' if 'neighbor_9' in ns and hand else None)
                if observed:
                    f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);v=c.frame.reshape(3,3).T@f[:3]*(1 if ns[1]==observed else -1)
                    (contacts if observed==TARGET else brace_contacts).append(dict(pair=ns,force_xyz_n=v.tolist()))
            bottom=float((Rotation.from_quat(d.xquat[body][[1,2,3,0]]).apply(p.CORNERS)+d.xpos[body])[:,2].min()-a.placement.a.g.FLOOR)
            if bottom<.040:abort='payload_below_base_top'
            if np.linalg.norm(d.xpos[body][:2]-plan['pair_center_m'][:2])>.060:abort='payload_outside_pair'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row=dict(elapsed_s=float(d.time)-start,phase=phase,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),
                         lowest_corner_m=bottom,contacts=contacts,brace_contacts=brace_contacts,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),
                         support_positions_m={n:d.xpos[m.body(n).id].tolist() for n in SUPPORTS})
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort:break
    summary=dict(abort=abort,duration_s=float(d.time)-start,wall_seconds=time.monotonic()-started,unintended_contacts=bad,
                 other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},native_verified=False,whole_arm_collision_checked=False,
                 full_pyramid_verified=False,official_region_verified=False,source_sequence_continuity_verified=False)
    measured_plan=dict(plan,pair_center_m=np.mean([rows[-1]['support_positions_m'][n] for n in SUPPORTS],axis=0).tolist())
    a.placement.evaluate(summary,rows,measured_plan,hold_start=27.)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    other=mj.MjData(m);mj.mj_setState(m,other,state,spec);x=np.empty_like(state);y=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,other);mj.mj_getState(m,d,x,spec);mj.mj_getState(m,other,y,spec);mj.mj_setState(m,d,state,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(x-y)))
    if summary['checkpoint_next_step_maximum_difference']>1e-10:summary['local_released_bridge_verified']=summary['local_first_yellow_aligned']=False
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--brace-preload-mm',type=float,choices=[0.,.4],default=0.)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    command,plan,restore,seed,anchors,right_q=prepare(args.brace_preload_mm);checked=precheck(command,restore,seed,right_q)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),left_stages=len(checked['stages']['left']),precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if args.run and passed:brief['dynamics']=run(command,plan,restore,anchors,args.out);passed=brief['dynamics']['local_first_yellow_aligned']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
