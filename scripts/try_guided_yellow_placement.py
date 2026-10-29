"""Local guided release from the actual loaded-transfer checkpoint.

Left fingers form a north-side guide; the right hand lowers 10 mm, opens and
withdraws southwest by default. Optional local comparisons change the guide,
withdrawal direction or opening sequence. No object resets or online truth.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import try_yellow_edge_acquisition as a

m,mj,d,p=a.m,a.mj,a.d,a.p
SOURCE=Path('runs/upper-layer-93/loaded-transfer')
RECORD=json.loads((SOURCE/'result.json').read_text())
STATE=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
CHECK=json.loads((SOURCE/'precheck.json').read_text())
TARGET='neighbor_1';SUPPORTS=['neighbor_9','neighbor_6'];DURATION=22.
CONTINUOUS_SOURCE=False


def configure_continuation(source,target):
    global SOURCE,RECORD,STATE,CHECK,TARGET,CONTINUOUS_SOURCE
    new_source=Path(source);record=json.loads((new_source/'result.json').read_text());check=json.loads((new_source/'precheck.json').read_text())
    state=np.load(new_source/'integration-state.npz',allow_pickle=False);s=record['summary']
    if (s.get('target')!=target or not s.get('local_loaded_transfer_verified',False) or s['abort'] or s['unintended_contacts'] or not check['passed']
            or not s['held_only_right_hand_contacts'] or s['held_duration_s']<2.98 or s['held_drift_mm']>2 or s['held_angular_drift_degrees']>2
            or s['held_minimum_up_force_n']<.8*m.body(target).mass*9.81 or max(s['other_bricks_max_shift_mm'].values())>2
            or s['checkpoint_next_step_maximum_difference']>1e-10):raise ValueError('Source is not a verified loaded transfer')
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=mj.MjData(m);mj.mj_setState(m,candidate,state['state'],int(state['spec']));mj.mj_forward(m,candidate)
    if not np.allclose(candidate.qpos,record['rows'][-1]['qpos'],rtol=0,atol=1e-12) or not np.allclose(candidate.qvel,record['rows'][-1]['qvel'],rtol=0,atol=1e-12):raise ValueError('State and trace differ')
    SOURCE=new_source;RECORD=record;STATE=state;CHECK=check;TARGET=target;CONTINUOUS_SOURCE=bool(s.get('source_sequence_continuity_verified',False))


def restore():
    s=RECORD['summary']
    if not s['local_loaded_transfer_verified'] or s['abort'] or s['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source did not pass its loaded transfer')
    if str(STATE['mujoco_version'])!=mj.__version__ or len(STATE['state'])!=mj.mj_stateSize(m,int(STATE['spec'])):
        raise ValueError('Checkpoint runtime mismatch')
    mj.mj_setState(m,d,STATE['state'],int(STATE['spec']));mj.mj_forward(m,d)


def prepare(compensate_landing=False,footprint_guide=False,south_only_withdrawal=False,open_before_withdrawal=False):
    global DURATION
    if compensate_landing and footprint_guide:raise ValueError('Footprint guide does not use prior landing compensation')
    if open_before_withdrawal and footprint_guide:raise ValueError('Compare release sequencing separately from guide motion')
    DURATION=25. if footprint_guide else 24. if open_before_withdrawal else 22.
    restore();start={s:(d.mocap_pos[i].copy(),d.mocap_quat[i].copy()) for s,i in [('left',p.t.LM),('right',p.t.RM)]}
    initial_ctrl={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)}
    target=d.xpos[m.body(TARGET).id].copy();quat=d.xquat[m.body(TARGET).id].copy()
    corners=Rotation.from_quat(quat[[1,2,3,0]]).apply(p.CORNERS)+target
    pair=np.mean([d.xpos[m.body(n).id] for n in SUPPORTS],axis=0)
    correction=np.zeros(3)
    if compensate_landing:
        prior=json.loads(Path('runs/upper-layer-94/guided-release/result.json').read_text())['summary']
        if prior['abort'] or not prior['held_only_base_contacts'] or prior['held_duration_s']<2.98:
            raise ValueError('Landing calibration must be an actually released stable trial')
        correction[:2]=-np.asarray(prior['final_center_offset_mm'])/1000
        if np.linalg.norm(correction)>.035:raise ValueError('Landing correction exceeds the local trial range')
    goals=p.t.CANDIDATE['hand_targets'].copy()
    for n,v in goals.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
    guide_rotation=Rotation.from_euler('z',90,degrees=True)*Rotation.from_quat(p.t.QUAT[[1,2,3,0]])
    guide_quat=guide_rotation.as_quat()[[3,0,1,2]]
    d.qpos[p.t.LA:p.t.LA+3]=0;d.qpos[p.t.LA+3:p.t.LA+7]=guide_quat;mj.mj_forward(m,d)
    vs=np.concatenate([a.g.mesh_vertices('lh_'+n) for n in ['index_proximal','index_distal','middle_proximal','middle_distal']])
    root=np.array([pair[0]-(vs[:,0].min()+vs[:,0].max())/2,corners[:,1].max()+.003-vs[:,1].min(),a.g.FLOOR+.075-vs[:,2].min()])
    root+=correction
    intermediate=root.copy();intermediate[1]=start['left'][0][1]
    turn=Slerp([2.,6.],Rotation.from_quat(np.array([start['left'][1],guide_quat])[:,[1,2,3,0]]))
    guide_delta=float(pair[1]+.027-(corners[:,1].max()+.003)) if footprint_guide else 0.
    if footprint_guide and not -.030<=guide_delta<=0:raise ValueError('Required inward guide motion is outside this local comparison')
    def command(seconds):
        elapsed=seconds
        if footprint_guide:seconds=seconds if seconds<9 else 9. if seconds<12 else seconds-3
        if open_before_withdrawal:seconds=seconds if seconds<11 else 11. if seconds<13 else seconds-2
        if seconds<2:left=start['left'][0];phase='fold_unused_left_fingers'
        elif seconds<6:
            f=(seconds-2)/4;left=start['left'][0]*(1-f)+intermediate*f;phase='guide_transit'
        elif seconds<9:
            f=(seconds-6)/3;left=intermediate*(1-f)+root*f;phase='guide_approach'
        elif seconds<17:left=root;phase='guide_hold'
        elif seconds<19:left=root+[0,.05*(seconds-17)/2,.05*(seconds-17)/2];phase='guide_withdraw'
        else:left=root+[0,.05,.05];phase='released_hold'
        if seconds<9:offset=np.zeros(3)
        elif seconds<11:offset=np.array([0,0,-.010*(seconds-9)/2]);phase='lower_receiver'
        elif seconds<13:
            f=(seconds-11)/2;offset=np.array([-.040*f,-.050*f,-.010]);phase='open_and_withdraw_receiver'
        elif seconds<15:
            f=(seconds-13)/2;offset=np.array([-.040-.040*f,-.050-.030*f,-.010]);phase='clear_receiver'
        else:offset=np.array([-.080,-.080,-.010])
        if south_only_withdrawal:offset[0]=0.
        if footprint_guide:
            left=left.copy();left[1]+=guide_delta*np.clip((elapsed-9)/3,0,1)
            if 9<=elapsed<12:phase='move_guide_to_final_footprint'
        poses=dict(left=(left,turn(np.clip(seconds,2,6)).as_quat()[[3,0,1,2]]),right=(start['right'][0]+offset+correction*np.clip(seconds/3,0,1),start['right'][1]))
        ctrl={n:initial_ctrl[n]+(v-initial_ctrl[n])*np.clip(seconds/2,0,1) for n,v in goals.items()}
        for finger in ['index','middle','ring','pinky']:
            for suffix in ['_mcp_pitch','_dip']:
                finger_time=elapsed if open_before_withdrawal else seconds
                name='rh_'+finger+suffix;ctrl[name]=initial_ctrl[name]*(1-np.clip((finger_time-11)/2,0,1))
        if open_before_withdrawal and 11<=elapsed<13:phase='open_stationary_receiver'
        return poses,ctrl,phase
    return command,dict(source=str(SOURCE),landing_correction_m=correction.tolist(),calibration_source='runs/upper-layer-94/guided-release' if compensate_landing else None,
                        guide_root_m=root.tolist(),guide_plane_y_m=float(corners[:,1].max()+.003+correction[1]),
                        guide_minimum_finger_height_m=.075,pair_center_m=pair.tolist(),initial_target_center_m=target.tolist(),
                        target=TARGET,supports=SUPPORTS,native_verified=False,whole_arm_collision_checked=False,
                        footprint_guide=footprint_guide,guide_inward_y_m=guide_delta,final_guide_plane_y_m=float(pair[1]+.027) if footprint_guide else None,
                        south_only_withdrawal=south_only_withdrawal,open_before_withdrawal=open_before_withdrawal,duration_s=DURATION,hold_start_s=DURATION-3,
                        source_sequence_continuity_verified=CONTINUOUS_SOURCE)


def precheck(command):
    started=time.monotonic();bad=[]
    for seconds in np.linspace(0,DURATION,round(DURATION*20)+1):
        restore();poses,ctrl,phase=command(seconds)
        for side,address in [('left',p.t.LA),('right',p.t.RA)]:
            pos,quat=poses[side];d.qpos[address:address+3]=pos;d.qpos[address+3:address+7]=quat
        for name,value in ctrl.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and TARGET not in ns:
                bad.append(dict(t=float(seconds),phase=phase,pair=ns,depth_mm=-float(c.dist)*1000))
    restore();stages={};initial={}
    for side,index in [('left',p.t.LM),('right',p.t.RM)]:
        ctx=p.t.routes.context()[side];ctx['root'][:3,3]=d.mocap_pos[index];ctx['root'][:3,:3]=Rotation.from_quat(d.mocap_quat[index][[1,2,3,0]]).as_matrix()
        ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020;q=CHECK['stages'][side][-1]['final_joints_sdk']
        actual=np.asarray(p.t.routes.sdk.forward_kinematics(side,q))@ctx['mount']
        initial[side]=dict(position_error_m=float(np.linalg.norm(actual[:3,3]-ctx['root'][:3,3])),angle_error_rad=float(Rotation.from_matrix(ctx['root'][:3,:3].T@actual[:3,:3]).magnitude()))
        stages[side]=[]
        if not bad and initial[side]['position_error_m']<1e-5 and initial[side]['angle_error_rad']<1e-4:
            for seconds in np.linspace(0,DURATION,round(DURATION*4)+1)[1:]:
                pos,quat=command(seconds)[0][side];ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                result=p.t.routes.stage(side,ctx,q,'guided_release',(pos-ctx['root'][:3,3]).tolist());stages[side].append(dict(t=float(seconds),result=result))
                if not result['continuous']:break
                q=result['final_joints_sdk']
    return dict(passed=not bad and all(len(v)==round(DURATION*4) and all(x['result']['continuous'] for x in v) for v in stages.values()),
                geometry_failures=bad,stages=stages,initial_seed_errors=initial,wall_seconds=time.monotonic()-started,
                scope='Hand paths with frozen source payload; actual release is unpredicted',whole_arm_collision_checked=False)


def evaluate(summary,rows,plan,hold_start=19.):
    held=[r for r in rows if r['elapsed_s']>=hold_start-1e-8]
    summary.update(local_released_bridge_verified=False,local_first_yellow_aligned=False)
    if not held:return
    xyz=np.array([r['position_m'] for r in held]);rot=Rotation.from_quat(np.array([r['quaternion_wxyz'] for r in held])[:,[1,2,3,0]])
    drift=float(np.linalg.norm(xyz-xyz[0],axis=1).max());angle=float(np.degrees((rot[0].inv()*rot).magnitude().max()))
    only_supports=all(r['contacts'] and all(all(n in [TARGET]+SUPPORTS for n in c['pair']) for c in r['contacts']) for r in held)
    support_force={n:min(sum(c['force_xyz_n'][2] for c in r['contacts'] if n in c['pair']) for r in held) for n in SUPPORTS}
    matrix=rot[-1].as_matrix();flatness=float(np.degrees(np.arccos(np.clip(np.max(np.abs(matrix[2,:])),0,1))))
    axis=matrix[:,int(np.argmax(np.abs(matrix[0,:])))];yaw=float((np.degrees(np.arctan2(axis[1],axis[0]))+45)%90-45)
    offset=xyz[-1,:2]-np.array(plan['pair_center_m'][:2])
    summary.update(held_duration_s=held[-1]['elapsed_s']-held[0]['elapsed_s'],held_drift_mm=drift*1000,held_angular_drift_degrees=angle,
                   support_minimum_up_force_n=support_force,held_only_base_contacts=only_supports,final_center_offset_mm=(offset*1000).tolist(),
                   final_cube_face_tilt_degrees=flatness,final_cube_yaw_mod90_degrees=yaw,final_bottom_mm=held[-1]['lowest_corner_m']*1000)
    summary['local_released_bridge_verified']=bool(not summary['abort'] and not summary['unintended_contacts'] and summary['held_duration_s']>=2.98
                          and drift<=.002 and angle<=2 and only_supports and min(support_force.values())>=.1
                          and max(summary['other_bricks_max_shift_mm'].values())<=2)
    summary['local_first_yellow_aligned']=bool(summary['local_released_bridge_verified'] and np.linalg.norm(offset)<=.005 and flatness<=2 and abs(yaw)<=10)


def run(command,plan,out):
    restore();start=float(d.time);body=m.body(TARGET).id
    initial={m.body(i).name:d.xpos[i].copy() for i in range(m.nbody) if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and i!=body}
    shifts={n:0. for n in initial};rows=[];bad=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            poses,ctrl,phase=command(float(d.time)-start)
            for side,index in [('left',p.t.LM),('right',p.t.RM)]:d.mocap_pos[index],d.mocap_quat[index]=poses[side]
            for name,value in ctrl.items():d.ctrl[m.actuator(name).id]=value
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,position in initial.items():
                shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-position)))
                if shifts[n]>.002:abort='other_brick_moved'
            contacts=[]
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(i)).name for i in c.geom]
                hand=any(n.startswith(('lh_','rh_')) for n in ns)
                intended=TARGET in ns and (hand or any(n in SUPPORTS for n in ns))
                if (hand or TARGET in ns) and not intended and c.dist<-.0005:
                    bad.append(dict(t=float(d.time)-start,pair=ns,depth_mm=-float(c.dist)*1000));abort='unexpected_contact'
                if TARGET in ns:
                    f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);v=c.frame.reshape(3,3).T@f[:3]*(1 if ns[1]==TARGET else -1)
                    contacts.append(dict(pair=ns,force_xyz_n=v.tolist()))
            rot=Rotation.from_quat(d.xquat[body][[1,2,3,0]]);bottom=float((rot.apply(p.CORNERS)+d.xpos[body])[:,2].min()-a.g.FLOOR)
            if bottom<.040:abort='payload_below_base_top'
            if np.linalg.norm(d.xpos[body][:2]-plan['pair_center_m'][:2])>.060:abort='payload_outside_pair'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row=dict(elapsed_s=float(d.time)-start,phase=phase,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),
                         lowest_corner_m=bottom,contacts=contacts,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort:break
    summary=dict(target=TARGET,abort=abort,duration_s=float(d.time)-start,cumulative_simulation_s=float(d.time),wall_seconds=time.monotonic()-started,unintended_contacts=bad,
                 other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},native_verified=False,whole_arm_collision_checked=False,
                 full_pyramid_verified=False,official_region_verified=False,source_sequence_continuity_verified=CONTINUOUS_SOURCE)
    evaluate(summary,rows,plan,hold_start=plan.get('hold_start_s',19.))
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    other=mj.MjData(m);mj.mj_setState(m,other,state,spec);x=np.empty_like(state);y=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,other);mj.mj_getState(m,d,x,spec);mj.mj_getState(m,other,y,spec);mj.mj_setState(m,d,state,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(x-y)))
    if summary['checkpoint_next_step_maximum_difference']>1e-10:summary['local_released_bridge_verified']=summary['local_first_yellow_aligned']=False
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--compensate-landing',action='store_true',help='Apply the prior measured landing offset through a continuous loaded translation')
    parser.add_argument('--source',type=Path);parser.add_argument('--target',choices=['neighbor_1','neighbor_5'],default='neighbor_1')
    parser.add_argument('--footprint-guide',action='store_true');parser.add_argument('--south-only-withdrawal',action='store_true')
    parser.add_argument('--open-before-withdrawal',action='store_true',help='Open at a stationary wrist for two seconds before withdrawing')
    args=parser.parse_args()
    if args.source:configure_continuation(args.source,args.target)
    elif args.target!='neighbor_1':raise ValueError('A different target requires its actual source')
    args.out.mkdir(parents=True,exist_ok=False);command,plan=prepare(args.compensate_landing,args.footprint_guide,args.south_only_withdrawal,args.open_before_withdrawal);checked=precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),stage_counts={s:len(v) for s,v in checked['stages'].items()},precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if args.run and passed:brief['dynamics']=run(command,plan,args.out);passed=brief['dynamics']['local_first_yellow_aligned']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
