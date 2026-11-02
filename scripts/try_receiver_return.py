"""Return the empty receiver continuously after the first upper placement.

Preserve the four-red/one-yellow state. Move outside the table before folding
the right arm to SDK zero; no initialization reset or native execution.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_fourth_red_with_upper as fourth
first,m,d,mj,p=fourth.first,fourth.m,fourth.d,fourth.mj,fourth.p
SOURCE=Path('runs/fourth-red-105/final-south')
RIGHT_SEED=Path('runs/upper-place-102/centered-tip/precheck.json')
TARGET='neighbor_3';DURATION=16.;HOLD=13.


def configure():
    record=json.loads((SOURCE/'result.json').read_text());check=json.loads((SOURCE/'precheck.json').read_text());s=record['summary']
    if not s['local_four_red_with_upper_verified'] or not s['protected_structure_verified'] or s['abort'] or s['unintended_contacts'] or not check['passed'] or s['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source is not a verified four-red/one-yellow state')
    state=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=mj.MjData(m);mj.mj_setState(m,candidate,state['state'],int(state['spec']));mj.mj_forward(m,candidate)
    np.testing.assert_allclose(candidate.qpos,record['rows'][-1]['qpos'],atol=1e-12,rtol=0);np.testing.assert_allclose(candidate.qvel,record['rows'][-1]['qvel'],atol=1e-12,rtol=0)
    def restore():mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    return restore,check['stages'][-1],np.array(json.loads(RIGHT_SEED.read_text())['stages']['right'][-1]['result']['final_joints_sdk'])


def prepare(restore,seed):
    restore();left=(d.mocap_pos[p.t.LM].copy(),d.mocap_quat[p.t.LM].copy());initial={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu) if m.actuator(i).name.startswith('rh_')}
    ctx=p.t.routes.context()['right'];outside=seed.copy();outside[0]=-.5;points=[seed,outside,np.array([-.5,0,0,0,0,0,0]),np.zeros(7)]
    times=[0.,4.,10.,13.]
    def joints(t):
        i=min(int(np.searchsorted(times,t,side='right'))-1,2);i=max(i,0);fraction=np.clip((t-times[i])/(times[i+1]-times[i]),0,1)
        return points[i]*(1-fraction)+points[i+1]*fraction
    def right(t):
        h=np.asarray(p.t.routes.sdk.forward_kinematics('right',joints(t).tolist()))@ctx['mount']
        return h[:3,3],Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]]
    def command(t):return *left,'right_swing_outside' if t<4 else 'right_fold_outside' if t<10 else 'right_zero_entry' if t<13 else 'released_hold'
    def fingers(t):return {n:v*(1-np.clip(t/4,0,1)) for n,v in initial.items()}
    center=d.xpos[m.body(TARGET).id].copy()
    plan=dict(source=str(SOURCE),target=TARGET,axis='stationary',component=0,destination_axis_m=float(center[0]),initial_target_center_m=center.tolist(),goal_xy_m=center[:2].tolist(),
              duration_s=DURATION,hold_start_s=HOLD,right_joint_waypoints=[q.tolist() for q in points],right_waypoint_times_s=times,right_seed_source=str(RIGHT_SEED),
              initial_right_controls=initial,source_sequence_continuity_verified=True,native_verified=False,whole_arm_collision_checked=False)
    return command,right,fingers,joints,plan


def precheck(command,right,fingers,joints,restore,left_stage):
    started=time.monotonic();restore();initial={};bad=[];ctxs=p.t.routes.context()
    for side,index,q in [('left',p.t.LM,left_stage['result']['final_joints_sdk']),('right',p.t.RM,joints(0).tolist())]:
        h=np.asarray(p.t.routes.sdk.forward_kinematics(side,q))@ctxs[side]['mount']
        initial[side]=dict(position_error_m=float(np.linalg.norm(h[:3,3]-d.mocap_pos[index])),angle_error_rad=float((Rotation.from_matrix(h[:3,:3]).inv()*Rotation.from_quat(d.mocap_quat[index][[1,2,3,0]])).magnitude()))
    samples=np.linspace(0,DURATION,round(DURATION*20)+1);joint_path=np.array([joints(t) for t in samples]);limits=ctxs['right']['intersection']+[.02,-.02]
    limit_ok=bool(np.all(joint_path>=limits[:,0]) and np.all(joint_path<=limits[:,1]))
    for t in samples:
        restore();pos,q=right(t);d.qpos[p.t.RA:p.t.RA+3]=pos;d.qpos[p.t.RA+3:p.t.RA+7]=q
        for n,v in fingers(t).items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
        mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns):bad.append(dict(t=float(t),pair=ns,depth_mm=-float(c.dist)*1000))
    return dict(passed=not bad and limit_ok and all(v['position_error_m']<1e-5 and v['angle_error_rad']<1e-4 for v in initial.values()),
                geometry_failures=bad,stages=[left_stage],stationary_left=True,initial_seed_errors=initial,right_joint_path_within_limits=limit_ok,
                right_final_joints_sdk=joints(DURATION).tolist(),right_geometry_samples=len(samples),wall_seconds=time.monotonic()-started,whole_arm_collision_checked=False)


def evaluate(out):
    record=json.loads((out/'result.json').read_text());s=record['summary'];rows=record['rows']
    metrics,only,forces=fourth.third.structure_status(rows,SOURCE,extra_bases=('neighbor_4','neighbor_2'))
    safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1 and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in metrics.values()))
    state=np.load(out/'integration-state.npz',allow_pickle=False);mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    h=np.asarray(p.t.routes.sdk.forward_kinematics('right',[0.]*7))@p.t.routes.context()['right']['mount']
    error=float(np.linalg.norm(h[:3,3]-d.mocap_pos[p.t.RM]));angle=float((Rotation.from_matrix(h[:3,:3]).inv()*Rotation.from_quat(d.mocap_quat[p.t.RM][[1,2,3,0]])).magnitude())
    no_contact=all(all(not any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in row['contacts']) for row in rows)
    fingers_open=all(abs(d.ctrl[i])<1e-12 for i in range(m.nu) if m.actuator(i).name.startswith('rh_'))
    held=[r for r in rows if r['elapsed_s']>=HOLD-1e-8]
    actual_position_error=float(np.linalg.norm(np.array([r['qpos'][p.t.RA:p.t.RA+3] for r in held])-h[:3,3],axis=1).max()) if held else None
    actual_angle_error=float((Rotation.from_matrix(h[:3,:3]).inv()*Rotation.from_quat(np.array([r['qpos'][p.t.RA+3:p.t.RA+7] for r in held])[:,[1,2,3,0]])).magnitude().max()) if held else None
    joints=[m.joint(i) for i in range(m.njnt) if m.joint(i).name.startswith('rh_')]
    actual_finger_error=max((abs(r['qpos'][m.jnt_qposadr[j.id]]) for r in held for j in joints),default=None)
    actual_finger_speed=max((abs(r['qvel'][m.jnt_dofadr[j.id]]) for r in held for j in joints),default=None)
    actual_settled=bool(held and actual_position_error<.001 and actual_angle_error<.01 and actual_finger_error<.01 and actual_finger_speed<.01)
    s.pop('local_red_position_verified',None)
    s.update(source_sequence_continuity_verified=True,protected_structure_metrics=metrics,protected_structure_verified=safe,upper_only_base_contacts=only,upper_minimum_support_force_n=forces,
             right_zero_command_position_error_m=error,right_zero_command_angle_error_rad=angle,right_finger_commands_open=fingers_open,
             held_actual_right_root_error_mm=actual_position_error*1000 if held else None,held_actual_right_root_angle_error_rad=actual_angle_error,
             held_actual_right_finger_error_rad=actual_finger_error,held_actual_right_finger_speed_rad_s=actual_finger_speed,
             local_receiver_return_verified=bool(safe and no_contact and fingers_open and actual_settled and error<1e-5 and angle<1e-4 and s.get('held_table_only',False) and s.get('held_duration_s',0)>=2.98
               and s.get('held_drift_mm',1e6)<=2 and s.get('held_angular_drift_degrees',1e6)<=2 and s['checkpoint_next_step_maximum_difference']<=1e-10))
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    restore,left_stage,seed=configure();command,right,fingers,joints,plan=prepare(restore,seed);checked=precheck(command,right,fingers,joints,restore,left_stage)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failures=len(checked['geometry_failures']),precheck_wall_seconds=checked['wall_seconds']);passed=checked['passed']
    if passed and args.run:
        first.run(command,plan,restore,args.out,finger_command=fingers,protected_payload=fourth.UPPER,right_command=right);brief['dynamics']=evaluate(args.out);passed=brief['dynamics']['local_receiver_return_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
