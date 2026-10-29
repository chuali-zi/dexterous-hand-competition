"""Continue the released two-red/one-yellow structure with the third red.

Local fixed actions with complete checkpoint continuation. Preserve the upper
brick's actual contact loads while moving another brick; no online truth control.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_replanned_second_red as second
first=second.first;m,d,mj,p=first.m,first.d,first.mj,first.p
TARGET='neighbor_4';UPPER='neighbor_5';BASES=['neighbor_9','neighbor_6']
REFERENCE=Path('runs/upper-place-102/centered-tip')


def configure(source):
    record=json.loads((source/'result.json').read_text());s=record['summary'];check=json.loads((source/'precheck.json').read_text())
    initial=source.resolve()==REFERENCE.resolve() and s.get('local_first_yellow_aligned',False) and s.get('local_released_bridge_verified',False)
    continued=(s.get('target')==TARGET and s.get('protected_structure_verified',False) and s.get('held_table_only',False)
               and s.get('held_duration_s',0)>=2.98 and s.get('held_drift_mm',1e6)<=2 and s.get('held_angular_drift_degrees',1e6)<=2
               and s.get('final_face_tilt_degrees',1e6)<=2 and s.get('minimum_table_support_n',0)>=.8*m.body(TARGET).mass*9.81)
    if not (initial or continued) or not s.get('source_sequence_continuity_verified',False) or s['abort'] or s['unintended_contacts'] or not check['passed'] or s['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source is not a stable continuation with the released bridge preserved')
    state=np.load(source/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=mj.MjData(m);mj.mj_setState(m,candidate,state['state'],int(state['spec']));mj.mj_forward(m,candidate)
    np.testing.assert_allclose(candidate.qpos,record['rows'][-1]['qpos'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(candidate.qvel,record['rows'][-1]['qvel'],atol=1e-12,rtol=0)
    def restore():mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    stages=check['stages']['left'] if isinstance(check['stages'],dict) else check['stages']
    return restore,stages[-1]['result']['final_joints_sdk']


def prepare(axis,source,restore,vertical_approach=False,tangent_shift_mm=0,approach_height=.070):
    reference=np.load(REFERENCE/'integration-state.npz',allow_pickle=False);ref=mj.MjData(m)
    mj.mj_setState(m,ref,reference['state'],int(reference['spec']));mj.mj_forward(m,ref)
    goal=[float(np.mean([ref.xpos[m.body(n).id,0] for n in BASES])),float(ref.xpos[m.body(BASES[-1]).id,1]+.055)]
    restore();initial={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)};targets=p.t.CANDIDATE['hand_targets'].copy()
    def fingers(t):return {n:initial[n]+(v-initial[n])*np.clip(t/2,0,1) for n,v in targets.items()}
    command,plan=second.prepare(axis,source,restore,tangent_shift_mm=tangent_shift_mm,target_name=TARGET,goal_xy_m=goal,
                                hand_targets=targets,vertical_approach=vertical_approach,approach_height=approach_height)
    plan.update(protected_structure=[*BASES,UPPER],structure_reference=str(REFERENCE),third_red_goal_xy_m=goal,
                initial_left_controls={n:initial[n] for n in targets},left_finger_targets=targets,source_sequence_continuity_verified=True)
    return command,fingers,plan


def structure_status(rows,reference_path=REFERENCE,extra_bases=()):
    reference=np.load(reference_path/'integration-state.npz',allow_pickle=False);ref=mj.MjData(m)
    mj.mj_setState(m,ref,reference['state'],int(reference['spec']));mj.mj_forward(m,ref)
    protected={}
    for name in [*BASES,UPPER,*extra_bases]:
        adr=m.jnt_qposadr[m.joint(name+'_free').id];xyz=np.array([row['qpos'][adr:adr+3] for row in rows])
        rotations=Rotation.from_quat(np.array([row['qpos'][adr+3:adr+7] for row in rows])[:,[1,2,3,0]])
        initial=Rotation.from_quat(ref.xquat[m.body(name).id][[1,2,3,0]])
        protected[name]=dict(maximum_shift_mm=float(np.linalg.norm(xyz-ref.xpos[m.body(name).id],axis=1).max()*1000),
                             maximum_angular_shift_degrees=float(np.degrees((initial.inv()*rotations).magnitude().max())))
    only=all(row['protected_payload_contacts'] and all(all(n in [*BASES,UPPER] for n in c['pair']) for c in row['protected_payload_contacts']) for row in rows)
    forces={n:min(sum(c['force_xyz_n'][2] for c in row['protected_payload_contacts'] if n in c['pair']) for row in rows) for n in BASES}
    return protected,only,forces


def evaluate(out,plan):
    record=json.loads((out/'result.json').read_text());s=record['summary'];rows=record['rows']
    protected,only,forces=structure_status(rows)
    s.update(protected_structure_metrics=protected,upper_only_base_contacts=only,upper_minimum_support_force_n=forces,
             protected_structure_verified=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1
                                               and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in protected.values())),
             source_sequence_continuity_verified=True)
    state=np.load(out/'integration-state.npz',allow_pickle=False);mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    names=[*BASES,TARGET];centers={n:d.xpos[m.body(n).id].copy() for n in names};corners={};yaw={};face={}
    for n in names:
        rotation=Rotation.from_quat(d.xquat[m.body(n).id][[1,2,3,0]]);corners[n]=rotation.apply(p.CORNERS)+centers[n]
        yaw[n]=float((rotation.as_euler('xyz',degrees=True)[2]+45)%90-45);face[n]=float(np.degrees(np.arccos(np.clip(np.max(np.abs(rotation.as_matrix()[2])),0,1))))
    spacing=[float(centers[b][1]-centers[a][1]) for a,b in zip(names,names[1:])]
    gaps=[float(corners[b][:,1].min()-corners[a][:,1].max()) for a,b in zip(names,names[1:])]
    span=float(np.ptp([centers[n][0] for n in names]))
    s.update(three_red_x_span_mm=span*1000,three_red_y_spacing_mm=(np.array(spacing)*1000).tolist(),three_red_gaps_mm=(np.array(gaps)*1000).tolist(),three_red_yaw_degrees=yaw,
             local_three_red_with_upper_verified=bool(s['local_red_position_verified'] and s['protected_structure_verified'] and span<=.005
                                                       and all(.050<=v<=.060 for v in spacing) and all(0<=v<=.010 for v in gaps)
                                                       and all(abs(v)<=10 for v in yaw.values()) and all(v<=2 for v in face.values())))
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--axis',choices=['east','south','align-west'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--vertical-approach',action='store_true');parser.add_argument('--tangent-shift-mm',type=int,choices=[0,10,15,20],default=0)
    parser.add_argument('--approach-height-mm',type=int,choices=[30,50,70],default=70)
    args=parser.parse_args();restore,seed=configure(args.source);command,fingers,plan=prepare(args.axis,args.source,restore,args.vertical_approach,args.tangent_shift_mm,args.approach_height_mm/1000)
    args.out.mkdir(parents=True,exist_ok=False);checked=first.precheck(command,restore,seed,target=TARGET,finger_command=fingers)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failures=len(checked['geometry_failures']),stages=len(checked['stages']),precheck_wall_seconds=checked['wall_seconds']);passed=checked['passed']
    if passed and args.run:
        first.run(command,plan,restore,args.out,finger_command=fingers,protected_payload=UPPER);brief['dynamics']=evaluate(args.out,plan)
        passed=brief['dynamics']['local_three_red_with_upper_verified'] if args.axis!='east' else brief['dynamics']['local_red_position_verified'] and brief['dynamics']['protected_structure_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
