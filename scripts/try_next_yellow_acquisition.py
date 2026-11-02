"""Acquire the next yellow while preserving the actual four-red/one-yellow state.

Reuse the table-edge receiver only from a continuous, released supply source.
The new source must independently pass full geometry, SDK and dynamics checks.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_yellow_edge_acquisition as a
import try_fourth_red_with_upper as fourth
REFERENCE=Path('runs/fourth-red-105/final-south')


def prepare_handoff(source):
    record=json.loads((source/'result.json').read_text());s=record['summary'];check=json.loads((source/'precheck.json').read_text())
    if (s.get('target')!='neighbor_3' or not s.get('local_acquisition_verified',False) or not s.get('protected_structure_verified',False)
            or not s.get('source_sequence_continuity_verified',False) or s['abort'] or s['unintended_hand_contacts'] or not check['passed']
            or s['held_duration_s']<2.98 or s['held_drift_mm']>2 or s['held_angular_drift_degrees']>2 or s['held_minimum_bottom_mm']<5
            or not s['held_only_hand_contacts'] or s['checkpoint_next_step_maximum_difference']>1e-10):raise ValueError('Handoff source is not a stable protected hand-supported hold')
    state=np.load(source/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=a.mj.__version__ or len(state['state'])!=a.mj.mj_stateSize(a.m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=a.mj.MjData(a.m);a.mj.mj_setState(a.m,candidate,state['state'],int(state['spec']));a.mj.mj_forward(a.m,candidate)
    np.testing.assert_allclose(candidate.qpos,record['rows'][-1]['qpos'],rtol=0,atol=1e-12);np.testing.assert_allclose(candidate.qvel,record['rows'][-1]['qvel'],rtol=0,atol=1e-12)
    a.SOURCE=source;a.STATE=state;a.RECORD=record;a.CHECK=check;a.TARGET='neighbor_3';a.CONTINUOUS_SOURCE=True;a.DURATION=8.
    a.CUP_BEFORE_LIFT=False;a.DELAYED_PINKY=False;a.RECENTER_WIDE_FACE=False;a.FACE_TARGETS={}
    a.restore();start={side:(a.d.mocap_pos[index].copy(),a.d.mocap_quat[index].copy()) for side,index in [('left',a.p.t.LM),('right',a.p.t.RM)]}
    def command(t):
        return dict(left=(start['left'][0]+np.array([.03,0,.03])*np.clip(t/4,0,1),start['left'][1]),right=start['right']), 'left_withdraw' if t<4 else 'right_hold'
    plan=dict(source=str(source),target=a.TARGET,duration_s=a.DURATION,hold_start_s=5.,left_clear_m=[.03,0,.03],structure_reference=str(REFERENCE),source_sequence_continuity_verified=True,native_verified=False,whole_arm_collision_checked=False)
    return command,plan


def precheck_handoff(command):
    started=time.monotonic();bad=[];initial={};stages=[]
    for t in np.linspace(0,a.DURATION,161):
        a.restore();poses,phase=command(t)
        for side,address in [('left',a.p.t.LA),('right',a.p.t.RA)]:a.d.qpos[address:address+3],a.d.qpos[address+3:address+7]=poses[side]
        a.mj.mj_forward(a.m,a.d)
        for c in a.d.contact:
            ns=[a.m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and a.TARGET not in ns:bad.append(dict(t=float(t),phase=phase,pair=ns,depth_mm=-float(c.dist)*1000))
    a.restore();contexts=a.p.t.routes.context();seed=a.CHECK['left_stages'][-1]['result']['final_joints_sdk']
    for side,index,q in [('left',a.p.t.LM,seed),('right',a.p.t.RM,a.CHECK['right_lift']['final_joints_sdk'])]:
        h=np.asarray(a.p.t.routes.sdk.forward_kinematics(side,q))@contexts[side]['mount']
        initial[side]=dict(position_error_m=float(np.linalg.norm(h[:3,3]-a.d.mocap_pos[index])),angle_error_rad=float((Rotation.from_matrix(h[:3,:3]).inv()*Rotation.from_quat(a.d.mocap_quat[index][[1,2,3,0]])).magnitude()))
    seed_ok=all(v['position_error_m']<1e-5 and v['angle_error_rad']<1e-4 for v in initial.values())
    if not bad and seed_ok:
        ctx=contexts['left'];ctx['limits']=ctx['intersection']+[.02,-.02];ctx['inset']=.02;q=seed
        for t in np.linspace(0,a.DURATION,33)[1:]:
            pos,quat=command(t)[0]['left'];ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            result=a.p.t.routes.stage('left',ctx,q,'left_handoff_withdraw',(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=float(t),result=result))
            if not result['continuous']:break
            q=result['final_joints_sdk']
    return dict(passed=not bad and seed_ok and len(stages)==32 and all(s['result']['continuous'] for s in stages),geometry_failures=bad,left_stages=stages,
                right_lift=a.CHECK['right_lift'],stationary_right=True,initial_seed_errors=initial,wall_seconds=time.monotonic()-started,whole_arm_collision_checked=False)


def prepare(source,four_fingers=False,cup_start=22.):
    if cup_start not in [20.,22.] or (cup_start!=22. and not four_fingers):raise ValueError('Earlier cupping compares the unchanged wide pusher only')
    a.configure_continuation(source,'neighbor_3');a.DURATION=31.;a.CUP_BEFORE_LIFT=True
    a.CUP_START=cup_start
    a.DELAYED_PINKY=not four_fingers;a.RECENTER_WIDE_FACE=not four_fingers
    a.four_finger_face() if four_fingers else a.four_finger_face(['index','middle','ring'])
    command,plan=a.prepare();plan.update(structure_reference=str(REFERENCE),protected_structure=[*fourth.BASES,fourth.TARGET,fourth.UPPER])
    return command,plan


def evaluate(out):
    record=json.loads((out/'result.json').read_text());s=record['summary']
    metrics,only,forces=fourth.third.structure_status(record['rows'],REFERENCE,extra_bases=('neighbor_4','neighbor_2'))
    safe=bool(not s['abort'] and not s['unintended_hand_contacts'] and only and min(forces.values())>=.1
              and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in metrics.values()))
    s.update(protected_structure_metrics=metrics,protected_structure_verified=safe,upper_only_base_contacts=only,upper_minimum_support_force_n=forces)
    s['local_acquisition_verified']=bool(s['local_acquisition_verified'] and safe)
    held=[r for r in record['rows'] if r['elapsed_s']>=a.DURATION-3-1e-8]
    s['held_only_right_hand_contacts']=bool(held and all(r['contacts'] and all(any(n.startswith('rh_') for n in c['pair']) for c in r['contacts']) for r in held))
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--four-finger-face',action='store_true');parser.add_argument('--cup-start-seconds',type=int,choices=[20,22],default=22)
    parser.add_argument('--handoff',action='store_true',help='Withdraw the left hand from a verified shared hold')
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.handoff:
        if args.four_finger_face or args.cup_start_seconds!=22:raise ValueError('Handoff preserves existing finger controls')
        command,plan=prepare_handoff(args.source);check=precheck_handoff(command)
    else:command,plan=prepare(args.source,args.four_finger_face,args.cup_start_seconds);check=a.precheck(command)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    brief=dict(precheck_passed=check['passed'],geometry_failures=len(check['geometry_failures']),left_stages=len(check['left_stages']),precheck_wall_seconds=check['wall_seconds']);passed=check['passed']
    if passed and args.run:
        a.run(command,args.out,protected_payload=fourth.UPPER);brief['dynamics']=evaluate(args.out);passed=brief['dynamics']['local_acquisition_verified'] and brief['dynamics']['held_only_right_hand_contacts']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
