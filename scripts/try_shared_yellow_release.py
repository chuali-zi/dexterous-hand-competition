"""Continue the shared second-yellow hold without resetting hands or objects.

Compare fixed wrist rotation and support withdrawal, then short post-release
alignment. Actual dynamics decide success; no binding or online truth feedback.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_guided_yellow_placement as g
import try_fourth_red_with_upper as fourth

REFERENCE=Path('runs/fourth-red-105/final-south')


def configure(source,end_pair=False,released=False):
    record=json.loads((source/'result.json').read_text());s=record['summary'];check=json.loads((source/'precheck.json').read_text())
    source_ok=s.get('local_released_bridge_verified',False) if released else (s.get('local_loaded_transfer_verified',False) or s.get('local_shared_reorientation_verified',False))
    contact_ok=s.get('held_only_base_contacts',False) if released else s.get('held_only_hand_contacts',False)
    if released and (not end_pair or json.loads((source/'plan.json').read_text())['supports']!=['neighbor_4','neighbor_2']):raise ValueError('Alignment requires the verified end-pair release')
    if released and (s['final_cube_face_tilt_degrees']>2 or s['final_bottom_mm']<45):raise ValueError('Alignment requires a near-flat released brick')
    if (s.get('target')!='neighbor_3' or not source_ok
            or not s.get('protected_structure_verified',False) or not s.get('source_sequence_continuity_verified',False)
            or s['abort'] or s['unintended_contacts'] or not check['passed'] or not contact_ok
            or s['held_duration_s']<2.98 or s['held_drift_mm']>2 or s['held_angular_drift_degrees']>2
            or s['held_minimum_up_force_n']<.8*g.m.body('neighbor_3').mass*9.81 or max(s['other_bricks_max_shift_mm'].values())>2
            or s['checkpoint_next_step_maximum_difference']>1e-10):raise ValueError('Source is not a verified protected shared hold')
    state=np.load(source/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=g.mj.__version__ or len(state['state'])!=g.mj.mj_stateSize(g.m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=g.mj.MjData(g.m);g.mj.mj_setState(g.m,candidate,state['state'],int(state['spec']));g.mj.mj_forward(g.m,candidate)
    np.testing.assert_allclose(candidate.qpos,record['rows'][-1]['qpos'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(candidate.qvel,record['rows'][-1]['qvel'],rtol=0,atol=1e-12)
    def restore():g.mj.mj_setState(g.m,g.d,state['state'],int(state['spec']));g.mj.mj_forward(g.m,g.d)
    g.SOURCE=source;g.RECORD=record;g.STATE=state;g.CHECK=check;g.TARGET='neighbor_3';g.SUPPORTS=['neighbor_4','neighbor_2'] if end_pair else ['neighbor_6','neighbor_4'];g.CONTINUOUS_SOURCE=True
    g.restore=restore


def prepare_alignment(fraction=1.):
    if not 0<fraction<=1:raise ValueError('Alignment fraction must be in (0, 1]')
    import align_released_yellow as align
    align.TARGET=g.TARGET;align.SUPPORTS=g.SUPPORTS
    left,plan=align.prepare('y',g.SOURCE,g.restore);g.restore();g.DURATION=align.DURATION
    right=(g.d.mocap_pos[g.p.t.RM].copy(),g.d.mocap_quat[g.p.t.RM].copy())
    controls={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    def command(t):
        pos,quat,phase=left(t)
        pos=pos+np.array(plan['normal'])*plan['distance_m']*(1-fraction)*np.clip((t-5)/3,0,1)
        return dict(left=(pos,quat),right=right),controls,phase
    plan.update(target=g.TARGET,supports=g.SUPPORTS,mode='align-y',duration_s=g.DURATION,hold_start_s=11.,
                protected_reference=str(REFERENCE),structure_reference=str(REFERENCE),source_sequence_continuity_verified=True,
                alignment_fraction=fraction,executed_push_distance_m=plan['distance_m']*fraction)
    return command,plan


def prepare(turn_fraction=1.):
    if not 0<turn_fraction<=1:raise ValueError('Turn fraction must be in (0, 1]')
    g.restore();m,d=g.m,g.d
    starts={side:(d.mocap_pos[i].copy(),d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)}
    center=d.xpos[m.body(g.TARGET).id].copy();cube=Rotation.from_quat(d.xquat[m.body(g.TARGET).id][[1,2,3,0]])
    normal=cube.as_matrix()[:,int(np.argmax(np.abs(cube.as_matrix()[2,:])))].copy()
    if normal[2]<0:normal=-normal
    axis=np.cross(normal,[0,0,1]);angle=float(np.arccos(np.clip(normal[2],-1,1)))
    delta=Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle) if angle>1e-10 else Rotation.identity()
    delta=Rotation.from_rotvec(delta.as_rotvec()*turn_fraction);turn_seconds=4.*turn_fraction;g.DURATION=turn_seconds+3
    def command(t):
        turn=Rotation.from_rotvec(delta.as_rotvec()*np.clip(t/turn_seconds,0,1));poses={}
        for side,(pos,quat) in starts.items():
            poses[side]=(turn.apply(pos-center)+center,(turn*Rotation.from_quat(quat[[1,2,3,0]])).as_quat()[[3,0,1,2]])
        return poses,controls,'shared_reorient' if t<turn_seconds else 'shared_reoriented_hold'
    pair=np.mean([d.xpos[m.body(n).id] for n in g.SUPPORTS],axis=0)
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),
                        initial_target_center_m=center.tolist(),rotation_vector_rad=delta.as_rotvec().tolist(),rotation_degrees=float(np.degrees(delta.magnitude())),turn_fraction=turn_fraction,
                        duration_s=g.DURATION,hold_start_s=turn_seconds,structure_reference=str(REFERENCE),finger_controls_unchanged=True,
                        source_sequence_continuity_verified=True,native_verified=False,whole_arm_collision_checked=False,online_object_feedback=False)


def prepare_withdrawal(aim_xy):
    aim=np.r_[np.asarray(aim_xy,dtype=float),0.]
    if not np.isfinite(aim).all() or np.linalg.norm(aim)>.050:raise ValueError('Release aim must be finite and within 50 mm')
    g.restore();m,d=g.m,g.d
    starts={side:(d.mocap_pos[i].copy(),d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)};g.DURATION=13.
    pair=np.mean([d.xpos[m.body(n).id] for n in g.SUPPORTS],axis=0)
    translation=aim+np.r_[pair[:2]-d.xpos[m.body(g.TARGET).id][:2],0.]
    def command(t):
        shift=translation*np.clip(t/3,0,1)
        left=shift+np.array([.03,0,.03])*np.clip((t-3)/4,0,1)
        right=shift+np.array([-.08,-.08,.05])*np.clip((t-7)/3,0,1)
        poses={side:(starts[side][0]+offset,starts[side][1]) for side,offset in [('left',left),('right',right)]}
        phase='align_release' if t<3 else 'withdraw_left_support' if t<7 else 'withdraw_right_receiver' if t<10 else 'released_hold'
        return poses,controls,phase
    return command,dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),
                        initial_target_center_m=d.xpos[m.body(g.TARGET).id].tolist(),mode='withdraw-release',aim_xy_m=aim[:2].tolist(),
                        common_translation_m=translation.tolist(),
                        left_withdrawal_m=[.03,0,.03],right_withdrawal_m=[-.08,-.08,.05],duration_s=g.DURATION,hold_start_s=10.,
                        structure_reference=str(REFERENCE),finger_controls_unchanged=True,source_sequence_continuity_verified=True,
                        native_verified=False,whole_arm_collision_checked=False,online_object_feedback=False)


def evaluate(out,release=False):
    record=json.loads((out/'result.json').read_text());s=record['summary'];rows=record['rows']
    metrics,only,forces=fourth.third.structure_status(rows,REFERENCE,extra_bases=('neighbor_4','neighbor_2'))
    safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1
              and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in metrics.values()))
    s.update(protected_structure_metrics=metrics,protected_structure_verified=safe,upper_only_base_contacts=only,upper_minimum_support_force_n=forces,local_shared_flatten_verified=False,local_shared_reorientation_verified=False)
    s['local_released_bridge_verified']=bool(s['local_released_bridge_verified'] and safe)
    s['local_second_yellow_aligned']=bool(s.pop('local_first_yellow_aligned',s.get('local_second_yellow_aligned',False)) and safe)
    supports=json.loads((out/'plan.json').read_text())['supports']
    centers=[np.array(rows[-1]['qpos'])[g.m.jnt_qposadr[g.m.joint(n+'_free').id]:g.m.jnt_qposadr[g.m.joint(n+'_free').id]+3] for n in supports]
    actual_offset=(np.array(rows[-1]['position_m'])-np.mean(centers,axis=0))[:2]
    s['final_actual_support_center_offset_mm']=(actual_offset*1000).tolist()
    s['local_second_yellow_aligned'] &= bool(np.linalg.norm(actual_offset)<=.005)
    held=[r for r in rows if r['elapsed_s']>=g.DURATION-3-1e-8]
    if held:
        hands=all(r['contacts'] and all(any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in r['contacts']) for r in held)
        up=min(sum(c['force_xyz_n'][2] for c in r['contacts']) for r in held)
        s.update(held_only_hand_contacts=hands,held_minimum_up_force_n=up,held_minimum_bottom_mm=1000*min(r['lowest_corner_m'] for r in held))
        s['held_only_right_hand_contacts']=bool(all(r['contacts'] and all(any(n.startswith('rh_') for n in c['pair']) for c in r['contacts']) for r in held))
        s['local_shared_reorientation_verified']=bool(not release and safe and hands and up>=.8*g.m.body(g.TARGET).mass*9.81 and s['held_duration_s']>=2.98
                         and s['held_drift_mm']<=2 and s['held_angular_drift_degrees']<=2
                         and s['checkpoint_next_step_maximum_difference']<=1e-10)
        s['local_shared_flatten_verified']=bool(s['local_shared_reorientation_verified'] and s['final_cube_face_tilt_degrees']<=2)
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=Path('runs/shared-transfer-107/raised'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');parser.add_argument('--turn-fraction',type=float,default=1.)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--withdraw-release',action='store_true');modes.add_argument('--align-y',action='store_true')
    parser.add_argument('--aim-mm',type=float,nargs=2,default=[0.,0.])
    parser.add_argument('--align-fraction',type=float,default=1.)
    parser.add_argument('--end-pair',action='store_true',help='Place this yellow on red 4/2 before filling the middle gap');args=parser.parse_args()
    configure(args.source,args.end_pair,args.align_y)
    command,plan=prepare_alignment(args.align_fraction) if args.align_y else prepare_withdrawal(np.array(args.aim_mm)/1000) if args.withdraw_release else prepare(args.turn_fraction)
    args.out.mkdir(parents=True,exist_ok=False);check=g.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),stage_counts={s:len(v) for s,v in check['stages'].items()},precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        g.run(command,plan,args.out,protected_payload=fourth.UPPER);brief['dynamics']=evaluate(args.out,args.withdraw_release or args.align_y)
        passed=brief['dynamics']['local_second_yellow_aligned' if args.withdraw_release or args.align_y else 'local_shared_reorientation_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
