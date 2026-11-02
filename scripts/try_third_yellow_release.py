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


def prepare_flatten_right(keep_left_guide=False):
    g.restore();center=g.d.xpos[g.m.body(g.TARGET).id].copy()
    starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    cube=Rotation.from_quat(g.d.xquat[g.m.body(g.TARGET).id][[1,2,3,0]])
    candidates=Rotation.create_group('O');delta=candidates[int(np.argmin((candidates*cube.inv()).magnitude()))]*cube.inv()
    pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0);g.DURATION=10.
    def command(t):
        turn=Rotation.from_rotvec(delta.as_rotvec()*np.clip((t-3)/4,0,1))
        left=(starts['left'][0]+np.array([0.,0.,0.] if keep_left_guide else [.08,.08,.08])*np.clip(t/3,0,1),starts['left'][1])
        pos,quat=starts['right'];right=(turn.apply(pos-center)+center,(turn*Rotation.from_quat(quat[[1,2,3,0]])).as_quat()[[3,0,1,2]])
        return dict(left=left,right=right),controls,('hold_left_guide' if keep_left_guide else 'withdraw_left_support') if t<3 else 'flatten_right_hold' if t<7 else 'flat_hold'
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='flatten-guided' if keep_left_guide else 'flatten-right',
        rotation_vector_rad=delta.as_rotvec().tolist(),rotation_degrees=float(np.degrees(delta.magnitude())),pivot_m=center.tolist(),
        left_withdrawal_m=[0.,0.,0.] if keep_left_guide else [.08,.08,.08],duration_s=10.,hold_start_s=7.,finger_controls_unchanged=True,
        source_sequence_continuity_verified=True,online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


def prepare_unfold_receiver(distal_only=True,settle_seconds=0.):
    """Keep the actual left guide while opening the right support at fixed wrists."""
    if not np.isfinite(settle_seconds) or not 0<=settle_seconds<=6:raise ValueError('Unfold settling time must be between zero and six seconds')
    g.restore();starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    suffixes=('_dip',) if distal_only else ('_mcp_pitch','_dip')
    opened=['rh_'+finger+suffix for finger in ('index','middle','ring','pinky') for suffix in suffixes]
    pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0);g.DURATION=6.+settle_seconds
    def command(t):
        ctrl=controls.copy()
        for name in opened:ctrl[name]*=1-np.clip((t-1)/2,0,1)
        return starts,ctrl,'hold_shared_support' if t<1 else 'unfold_receiver_under_guide' if t<3 else 'hold_unfolded_support'
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='unfold-receiver',
        distal_only=distal_only,opened_joints=opened,duration_s=g.DURATION,hold_start_s=3.+settle_seconds,settle_seconds=settle_seconds,finger_controls_unchanged=False,
        wrists_unchanged=True,left_controls_unchanged=True,source_sequence_continuity_verified=True,
        online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


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


def prepare_narrow_open_support():
    """Fold the right outer fingers while preserving the existing left guide."""
    g.restore();starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    if not g.RECORD['summary'].get('local_third_yellow_level_hold_verified',False):raise ValueError('Narrowing requires a level loaded hold')
    pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0);g.DURATION=8.
    def command(t):
        ctrl=controls.copy()
        for name,start,duration in [('rh_pinky_mcp_pitch',0.,1.),('rh_ring_mcp_pitch',1.,2.)]:
            ctrl[name]+=(1.45-ctrl[name])*np.clip((t-start)/duration,0,1)
        return starts,ctrl,'fold_pinky' if t<1 else 'fold_ring_with_left_guide' if t<3 else 'settle_narrow_support' if t<5 else 'hold_narrow_support'
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='narrow-open-support',
        folded_joints=['rh_pinky_mcp_pitch','rh_ring_mcp_pitch'],fold_angle_rad=1.45,duration_s=8.,hold_start_s=5.,
        wrists_unchanged=True,left_controls_unchanged=True,source_sequence_continuity_verified=True,
        online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


def prepare_top_stop(joint_entry=False):
    """Move the left guide around the cube's upper edge into a top stop."""
    g.restore();center=g.d.xpos[g.m.body(g.TARGET).id].copy()
    starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    if not g.RECORD['summary'].get('local_third_yellow_flat_hold_verified',False):raise ValueError('Top-stop transfer requires an aligned level hold')
    pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0);g.DURATION=10.
    entry=None;seed=None;ctx=None
    if joint_entry:
        turn=Rotation.from_euler('y',-90,degrees=True);pos,quat=starts['left']
        endpoint=(turn.apply(pos-center)+center,(turn*Rotation.from_quat(quat[[1,2,3,0]])).as_quat()[[3,0,1,2]])
        entry=g.a.g.endpoint('left',*endpoint)
        if not entry['accepted']:raise ValueError('Top-stop endpoint is not reachable')
        last=g.CHECK['stages']['left'][-1];seed=np.array(last.get('result',last)['final_joints_sdk']);ctx=g.p.t.routes.context()['left']
    def command(t):
        angle=np.pi/2*np.clip((t-1)/6,0,1);turn=Rotation.from_euler('y',-angle)
        radial=np.array([np.cos(angle),0,np.sin(angle)])
        corner_offset=(.025/max(np.cos(angle),np.sin(angle))-.025)*radial
        pos,quat=starts['left'];left=(turn.apply(pos-center)+center+corner_offset,(turn*Rotation.from_quat(quat[[1,2,3,0]])).as_quat()[[3,0,1,2]])
        if joint_entry:
            q=seed+(np.array(entry['joints'])-seed)*np.clip((t-1)/6,0,1)
            fk=np.asarray(g.p.t.routes.sdk.forward_kinematics('left',q.tolist()))@ctx['mount'];left=(fk[:3,3],Rotation.from_matrix(fk[:3,:3]).as_quat()[[3,0,1,2]])
        return dict(left=left,right=starts['right']),controls,'hold_aligned_support' if t<1 else 'move_left_over_upper_edge' if t<7 else 'hold_top_stop'
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='top-stop',
        pivot_m=center.tolist(),rotation_axis_world=[0,-1,0],rotation_degrees=90.,cube_half_extent_m=.025,
        duration_s=10.,hold_start_s=7.,finger_controls_unchanged=True,right_wrist_unchanged=True,
        joint_entry=joint_entry,left_joint_seed=seed.tolist() if joint_entry else None,left_joint_endpoint=entry,
        source_sequence_continuity_verified=True,online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


def prepare_strip_support(stop_lift=0.,away_from_stop=False):
    """Compare withdrawal relative to a left stop, then clear the stop."""
    if not np.isfinite(stop_lift) or not 0<=stop_lift<=.020:raise ValueError('Stop lift must be between zero and 20 mm')
    g.restore();starts={side:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    if not g.RECORD['summary'].get('local_third_yellow_flat_hold_verified',False):raise ValueError('Stripping requires an aligned flat loaded hold')
    if any(abs(v)>1e-8 for n,v in controls.items() if n.startswith('rh_') and not n.startswith('rh_thumb')):raise ValueError('Right support fingers must already be open')
    normals=[]
    for c in g.d.contact:
        names=[g.m.geom(int(i)).name for i in c.geom]
        if g.TARGET in names and any(n.startswith('lh_') for n in names):normals.append(c.frame[:3]*(1 if names[0]==g.TARGET else -1))
    if not normals:raise ValueError('No actual left stop contact')
    normal=np.mean(normals,axis=0);normal[2]=0
    if np.linalg.norm(normal)<.5:raise ValueError('Left stop has no clear horizontal normal')
    normal/=np.linalg.norm(normal);withdrawal=normal*.070*(-1 if away_from_stop else 1);pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0);delay=2. if stop_lift else 0.;g.DURATION=13.+delay
    def command(t):
        clock=t-delay
        right=(starts['right'][0]+withdrawal*np.clip(clock/4,0,1),starts['right'][1])
        left=(starts['left'][0]+np.array([0,0,stop_lift])*np.clip(t,0,1)+(normal*.030+[0,0,.080])*np.clip((clock-7)/3,0,1),starts['left'][1])
        return dict(left=left,right=right),controls,'raise_left_stop' if t<delay else 'strip_open_support' if clock<4 else 'settle_against_left_stop' if clock<7 else 'clear_left_stop' if clock<10 else 'released_hold'
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),mode='strip-support',
        stop_outward_normal=normal.tolist(),right_withdrawal_m=withdrawal.tolist(),away_from_stop=away_from_stop,stop_lift_m=stop_lift,duration_s=g.DURATION,hold_start_s=10.+delay,
        finger_controls_unchanged=True,source_sequence_continuity_verified=True,online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


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
    if plan['mode'] in ('common-yaw','flatten-right','flatten-guided','flatten-shared','unfold-receiver','narrow-open-support','top-stop'):
        held=[row for row in rows if row['elapsed_s']>=plan['hold_start_s']-1e-8]
        only=bool(held and all(row['contacts'] and all(any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in row['contacts']) for row in held))
        up=min((sum(c['force_xyz_n'][2] for c in row['contacts']) for row in held),default=0.)
        summary.update(held_only_hand_contacts=only,held_minimum_up_force_n=up,
            local_shared_reorientation_verified=bool(summary['protected_structure_verified'] and only and up>=.8*g.m.body(g.TARGET).mass*9.81
                and summary.get('held_duration_s',0)>=2.98 and summary.get('held_drift_mm',float('inf'))<=2 and summary.get('held_angular_drift_degrees',float('inf'))<=2))
        summary['held_only_right_hand_contacts']=bool(held and all(row['contacts'] and all(any(n.startswith('rh_') for n in c['pair']) for c in row['contacts']) for row in held))
        summary['local_third_yellow_level_hold_verified']=bool(summary['local_shared_reorientation_verified'] and summary.get('final_cube_face_tilt_degrees',float('inf'))<=2)
        summary['local_third_yellow_flat_hold_verified']=bool(summary['local_third_yellow_level_hold_verified'] and abs(summary.get('final_cube_yaw_mod90_degrees',float('inf')))<=10)
        if plan['mode']=='top-stop':
            left_up=[sum(c['force_xyz_n'][2] for c in row['contacts'] if any(n.startswith('lh_') for n in c['pair'])) for row in held]
            summary['held_maximum_left_up_force_n']=max(left_up) if left_up else None
            summary['local_top_stop_verified']=bool(summary['local_third_yellow_flat_hold_verified'] and left_up and max(left_up)<=-.1)
    record['summary']=summary;(out/'result.json').write_text(json.dumps(record,allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/third-placement-112/transfer'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--lower-mm',type=float,default=40.)
    parser.add_argument('--unfold-settle-seconds',type=float,default=0.)
    parser.add_argument('--stop-lift-mm',type=float,default=0.)
    parser.add_argument('--strip-away-from-stop',action='store_true')
    parser.add_argument('--top-stop-joint-entry',action='store_true')
    parser.add_argument('--level-fraction',type=float,default=1.)
    parser.add_argument('--aim-mm',type=float,nargs=2,default=[0.,0.]);modes=parser.add_mutually_exclusive_group();modes.add_argument('--open-receiver',action='store_true');modes.add_argument('--yaw-degrees',type=float);modes.add_argument('--narrow-lower',action='store_true');modes.add_argument('--flatten-right',action='store_true');modes.add_argument('--guided-flatten',action='store_true');modes.add_argument('--flatten-shared',action='store_true');modes.add_argument('--unfold-distal',action='store_true');modes.add_argument('--unfold-full',action='store_true');modes.add_argument('--strip-support',action='store_true');modes.add_argument('--narrow-open-support',action='store_true');modes.add_argument('--top-stop',action='store_true');args=parser.parse_args()
    if args.unfold_settle_seconds and not (args.unfold_distal or args.unfold_full):parser.error('--unfold-settle-seconds requires an unfold mode')
    if args.stop_lift_mm and not args.strip_support:parser.error('--stop-lift-mm requires --strip-support')
    if args.strip_away_from_stop and not args.strip_support:parser.error('--strip-away-from-stop requires --strip-support')
    if args.top_stop_joint_entry and not args.top_stop:parser.error('--top-stop-joint-entry requires --top-stop')
    if args.level_fraction!=1. and not args.flatten_shared:parser.error('--level-fraction requires --flatten-shared')
    r.configure(args.source,target=feed.TARGET)
    command,plan=prepare_top_stop(args.top_stop_joint_entry) if args.top_stop else prepare_narrow_open_support() if args.narrow_open_support else prepare_strip_support(args.stop_lift_mm/1000,args.strip_away_from_stop) if args.strip_support else r.prepare(args.level_fraction) if args.flatten_shared else prepare_flatten_right(args.guided_flatten) if args.flatten_right or args.guided_flatten else prepare_unfold_receiver(args.unfold_distal,args.unfold_settle_seconds) if args.unfold_distal or args.unfold_full else prepare_yaw(args.yaw_degrees) if args.yaw_degrees is not None else prepare_narrow_lower(args.lower_mm/1000) if args.narrow_lower else prepare(np.array(args.aim_mm)/1000,args.open_receiver)
    if args.flatten_shared:plan['mode']='flatten-shared'
    plan.update(second_upper_reference=str(feed.SECOND_REFERENCE))
    args.out.mkdir(parents=True,exist_ok=False);check=g.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),stage_counts={s:len(v) for s,v in check['stages'].items()},precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        g.run(command,plan,args.out,protected_payload='neighbor_5',additional_protected_payloads=(feed.SECOND,*feed.BASES))
        brief['dynamics']=evaluate(args.out);passed=brief['dynamics']['local_top_stop_verified' if args.top_stop else 'local_third_yellow_level_hold_verified' if args.flatten_shared or args.narrow_open_support else 'local_third_yellow_flat_hold_verified' if args.flatten_right or args.guided_flatten else 'local_shared_reorientation_verified' if args.yaw_degrees is not None or args.unfold_distal or args.unfold_full else 'local_third_yellow_aligned']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
