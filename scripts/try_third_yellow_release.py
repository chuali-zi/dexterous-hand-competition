"""Release the actually transported third yellow over the middle base pair."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_third_yellow_feed as feed

r=feed.release;g=r.g


def prepare_yaw(degrees):
    if not np.isfinite(degrees) or abs(degrees)>90:raise ValueError('Local common yaw is limited to 90 degrees')
    g.restore();center=g.d.xpos[g.m.body(g.TARGET).id].copy()
    starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0)
    shift=np.r_[pair[:2]-center[:2],0.];g.DURATION=10.
    def command(t):
        turn=Rotation.from_euler('z',degrees*np.clip((t-3)/4,0,1),degrees=True);poses={}
        for side,(pos,quat) in starts.items():
            poses[side]=(turn.apply(pos-center)+center+shift*np.clip(t/3,0,1),(turn*Rotation.from_quat(quat[[1,2,3,0]])).as_quat()[[3,0,1,2]])
        return poses,controls,'align_shared_hold' if t<3 else 'common_yaw' if t<7 else 'reoriented_hold'
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='common-yaw',
        yaw_degrees=degrees,common_translation_m=shift.tolist(),pivot_m=center.tolist(),duration_s=10.,hold_start_s=7.,
        finger_controls_unchanged=True,source_sequence_continuity_verified=True,online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


def prepare(aim_xy,open_receiver=False):
    base,plan=r.prepare_withdrawal(aim_xy)
    if not open_receiver:return base,plan
    g.DURATION=15.
    def command(t):
        clock=t if t<7 else 7. if t<9 else t-2.
        poses,ctrl,phase=base(clock);ctrl=ctrl.copy()
        for finger in ('index','middle','ring','pinky'):
            for suffix in ('_mcp_pitch','_dip'):
                name='rh_'+finger+suffix;ctrl[name]*=1-np.clip((t-7)/2,0,1)
        if 7<=t<9:phase='open_stationary_receiver'
        return poses,ctrl,phase
    plan.update(mode='open-then-withdraw',duration_s=15.,hold_start_s=12.,finger_controls_unchanged=False,open_receiver_interval_s=[7.,9.])
    return command,plan


def prepare_flatten_right():
    g.restore();center=g.d.xpos[g.m.body(g.TARGET).id].copy()
    starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    cube=Rotation.from_quat(g.d.xquat[g.m.body(g.TARGET).id][[1,2,3,0]])
    candidates=Rotation.create_group('O');delta=candidates[int(np.argmin((candidates*cube.inv()).magnitude()))]*cube.inv()
    pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0);g.DURATION=10.
    def command(t):
        turn=Rotation.from_rotvec(delta.as_rotvec()*np.clip((t-3)/4,0,1))
        left=(starts['left'][0]+np.array([.08,.08,.08])*np.clip(t/3,0,1),starts['left'][1])
        pos,quat=starts['right'];right=(turn.apply(pos-center)+center,(turn*Rotation.from_quat(quat[[1,2,3,0]])).as_quat()[[3,0,1,2]])
        return dict(left=left,right=right),controls,'withdraw_left_support' if t<3 else 'flatten_right_hold' if t<7 else 'flat_hold'
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='flatten-right',
        rotation_vector_rad=delta.as_rotvec().tolist(),rotation_degrees=float(np.degrees(delta.magnitude())),pivot_m=center.tolist(),
        left_withdrawal_m=[.08,.08,.08],duration_s=10.,hold_start_s=7.,finger_controls_unchanged=True,
        source_sequence_continuity_verified=True,online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


def prepare_narrow_lower(lowering=.040):
    if not np.isfinite(lowering) or not 0<=lowering<=.060:raise ValueError('Lowering must be between zero and 60 mm')
    g.restore();starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0);g.DURATION=16.
    folded=('lh_index_mcp_pitch','rh_ring_mcp_pitch','rh_pinky_mcp_pitch')
    def command(t):
        ctrl=controls.copy()
        for name in folded:ctrl[name]+=(1.45-ctrl[name])*np.clip(t if name.startswith('lh_') else t-1,0,1)
        for finger in ('index','middle'):
            for suffix in ('_mcp_pitch','_dip'):
                name='rh_'+finger+suffix;ctrl[name]*=1-np.clip((t-6)/2,0,1)
        lower=np.array([0.,0.,-lowering])*np.clip((t-2)/4,0,1)
        poses={side:(pos+lower+np.array([-.080,0,.040] if side=='right' else [.040,0,.040])*np.clip((t-(8 if side=='right' else 11))/(3 if side=='right' else 2),0,1),quat) for side,(pos,quat) in starts.items()}
        phase='fold_unused_fingers' if t<2 else 'lower_between_upper_bricks' if t<6 else 'open_under_left_guide' if t<8 else 'withdraw_right_west' if t<11 else 'withdraw_left_guide' if t<13 else 'released_hold'
        return poses,ctrl,phase
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='narrow-lower',
        folded_mcp_joints=list(folded),folded_mcp_pitch_rad=1.45,fold_schedule_s={'left':[0.,1.],'right':[1.,2.]},lowering_m=lowering,duration_s=16.,hold_start_s=13.,
        source_sequence_continuity_verified=True,online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


def evaluate(out):
    summary=feed.evaluate(out,allow_third_yellow_contact=True)
    record=json.loads((out/'result.json').read_text());rows=record['rows']
    centers=[]
    for name in g.SUPPORTS:
        adr=g.m.jnt_qposadr[g.m.joint(name+'_free').id];centers.append(rows[-1]['qpos'][adr:adr+3])
    offset=(np.array(rows[-1]['position_m'])-np.mean(centers,axis=0))[:2]
    summary['final_actual_support_center_error_mm']=float(np.linalg.norm(offset)*1000)
    summary['local_released_bridge_verified'] &= summary['protected_structure_verified']
    summary['local_third_yellow_aligned']=bool(summary.pop('local_first_yellow_aligned',summary.get('local_third_yellow_aligned',False)) and summary['protected_structure_verified'] and np.linalg.norm(offset)<=.005)
    plan=json.loads((out/'plan.json').read_text())
    if plan['mode'] in ('common-yaw','flatten-right'):
        held=[row for row in rows if row['elapsed_s']>=plan['hold_start_s']-1e-8]
        only=bool(held and all(row['contacts'] and all(any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in row['contacts']) for row in held))
        up=min((sum(c['force_xyz_n'][2] for c in row['contacts']) for row in held),default=0.)
        summary.update(held_only_hand_contacts=only,held_minimum_up_force_n=up,
            local_shared_reorientation_verified=bool(summary['protected_structure_verified'] and only and up>=.8*g.m.body(g.TARGET).mass*9.81
                and summary.get('held_duration_s',0)>=2.98 and summary.get('held_drift_mm',float('inf'))<=2 and summary.get('held_angular_drift_degrees',float('inf'))<=2))
        summary['held_only_right_hand_contacts']=bool(held and all(row['contacts'] and all(any(n.startswith('rh_') for n in c['pair']) for c in row['contacts']) for row in held))
        summary['local_third_yellow_flat_hold_verified']=bool(summary['local_shared_reorientation_verified'] and summary.get('final_cube_face_tilt_degrees',float('inf'))<=2
            and abs(summary.get('final_cube_yaw_mod90_degrees',float('inf')))<=10)
    record['summary']=summary;(out/'result.json').write_text(json.dumps(record,allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/third-placement-112/transfer'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--lower-mm',type=float,default=40.)
    parser.add_argument('--aim-mm',type=float,nargs=2,default=[0.,0.]);modes=parser.add_mutually_exclusive_group();modes.add_argument('--open-receiver',action='store_true');modes.add_argument('--yaw-degrees',type=float);modes.add_argument('--narrow-lower',action='store_true');modes.add_argument('--flatten-right',action='store_true');args=parser.parse_args()
    r.configure(args.source,target=feed.TARGET)
    command,plan=prepare_flatten_right() if args.flatten_right else prepare_yaw(args.yaw_degrees) if args.yaw_degrees is not None else prepare_narrow_lower(args.lower_mm/1000) if args.narrow_lower else prepare(np.array(args.aim_mm)/1000,args.open_receiver)
    plan.update(second_upper_reference=str(feed.SECOND_REFERENCE))
    args.out.mkdir(parents=True,exist_ok=False);check=g.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),stage_counts={s:len(v) for s,v in check['stages'].items()},precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        g.run(command,plan,args.out,protected_payload='neighbor_5',additional_protected_payloads=(feed.SECOND,*feed.BASES))
        brief['dynamics']=evaluate(args.out);passed=brief['dynamics']['local_third_yellow_flat_hold_verified' if args.flatten_right else 'local_shared_reorientation_verified' if args.yaw_degrees is not None else 'local_third_yellow_aligned']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
