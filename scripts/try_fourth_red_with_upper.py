"""Fill the fourth base slot from the retained three-red/one-yellow chain.

Local fixed actions and complete-state continuation. Preserve the existing
structure; no online object truth control or platform execution.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_third_red_with_upper as third
first,second=third.first,third.second
m,d,mj,p=third.m,third.d,third.mj,third.p
TARGET='neighbor_2';BASES=[*third.BASES,'neighbor_4'];UPPER=third.UPPER
REFERENCE=Path('runs/next-feed-104/west-finish')


def configure(source):
    record=json.loads((source/'result.json').read_text());s=record['summary'];checked=json.loads((source/'precheck.json').read_text())
    initial=source.resolve()==REFERENCE.resolve() and s.get('target')=='neighbor_3'
    continued=s.get('target') in [TARGET,'neighbor_7']
    stable=(s.get('held_table_only',False) and s.get('held_duration_s',0)>=2.98 and s.get('held_drift_mm',1e6)<=2
            and s.get('held_angular_drift_degrees',1e6)<=2 and s.get('final_face_tilt_degrees',1e6)<=2
            and s.get('minimum_table_support_n',0)>=.8*m.body(s['target']).mass*9.81)
    if not (initial or continued) or not stable or not s.get('protected_structure_verified',False) or not s.get('source_sequence_continuity_verified',False) or s['abort'] or s['unintended_contacts'] or not checked['passed'] or s['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source is not a stable protected continuation for the fourth base')
    state=np.load(source/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=mj.MjData(m);mj.mj_setState(m,candidate,state['state'],int(state['spec']));mj.mj_forward(m,candidate)
    np.testing.assert_allclose(candidate.qpos,record['rows'][-1]['qpos'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(candidate.qvel,record['rows'][-1]['qvel'],atol=1e-12,rtol=0)
    def restore():mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    return restore,checked['stages'][-1]['result']['final_joints_sdk']


def prepare(axis,source,restore,approach_height=.030,tangent_shift_mm=0,south_wrist_yaw=90,compact_lateral_face=False):
    if compact_lateral_face and axis!='align-west':raise ValueError('Compact face is a lateral alignment candidate')
    reference=np.load(REFERENCE/'integration-state.npz',allow_pickle=False);ref=mj.MjData(m)
    mj.mj_setState(m,ref,reference['state'],int(reference['spec']));mj.mj_forward(m,ref)
    goal=[float(np.mean([ref.xpos[m.body(n).id,0] for n in BASES])),float(ref.xpos[m.body(BASES[-1]).id,1]+.055)]
    restore();initial={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)};targets=p.t.CANDIDATE['hand_targets'].copy()
    moving='neighbor_7' if axis=='north-clear' else TARGET
    move_goal=[float(d.xpos[m.body(moving).id,0]),.240] if axis=='north-clear' else goal
    if compact_lateral_face:
        for finger in ['index','middle']:
            for suffix in ['_mcp_pitch','_dip']:targets['lh_'+finger+suffix]=1.0
    def fingers(t):return {n:initial[n]+(v-initial[n])*np.clip(t/2,0,1) for n,v in targets.items()}
    if compact_lateral_face:
        command,plan=first.prepare('west',source,restore,target_name=TARGET,goal_xy_m=goal,hand_targets=targets,
                                   approach_height=approach_height,adapt_lateral_face=True)
    else:
        command,plan=second.prepare('north' if axis=='north-clear' else axis,source,restore,target_name=moving,goal_xy_m=move_goal,hand_targets=targets,
                                vertical_approach=axis in ['south','north-clear'],approach_height=approach_height,tangent_shift_mm=tangent_shift_mm,south_wrist_yaw=south_wrist_yaw)
    plan.update(fourth_red_goal_xy_m=goal,structure_reference=str(REFERENCE),protected_structure=[*BASES,UPPER],
                initial_left_controls={n:initial[n] for n in targets},left_finger_targets=targets,compact_lateral_face=compact_lateral_face,source_sequence_continuity_verified=True)
    return command,fingers,plan


def evaluate(out,plan):
    record=json.loads((out/'result.json').read_text());s=record['summary']
    protected,only,forces=third.structure_status(record['rows'],REFERENCE,extra_bases=('neighbor_4',))
    safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1
              and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in protected.values()))
    s.update(protected_structure_metrics=protected,protected_structure_verified=safe,upper_only_base_contacts=only,
             upper_minimum_support_force_n=forces,source_sequence_continuity_verified=True)
    state=np.load(out/'integration-state.npz',allow_pickle=False);mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    names=[*BASES,TARGET];centers={n:d.xpos[m.body(n).id].copy() for n in names};corners={};yaw={};face={}
    for n in names:
        rotation=Rotation.from_quat(d.xquat[m.body(n).id][[1,2,3,0]]);corners[n]=rotation.apply(p.CORNERS)+centers[n]
        yaw[n]=float((rotation.as_euler('xyz',degrees=True)[2]+45)%90-45)
        face[n]=float(np.degrees(np.arccos(np.clip(np.max(np.abs(rotation.as_matrix()[2])),0,1))))
    spacing=[float(centers[b][1]-centers[a][1]) for a,b in zip(names,names[1:])]
    gaps=[float(corners[b][:,1].min()-corners[a][:,1].max()) for a,b in zip(names,names[1:])]
    span=float(np.ptp([centers[n][0] for n in names]))
    s.update(four_red_x_span_mm=span*1000,four_red_y_spacing_mm=(np.array(spacing)*1000).tolist(),four_red_gaps_mm=(np.array(gaps)*1000).tolist(),
             four_red_yaw_degrees=yaw,local_four_red_with_upper_verified=bool(s['target']==TARGET and s['local_red_position_verified'] and safe and span<=.005
                 and all(.050<=v<=.060 for v in spacing) and all(0<=v<=.010 for v in gaps)
                 and all(abs(v)<=10 for v in yaw.values()) and all(v<=2 for v in face.values())))
    if s['target']=='neighbor_7':
        s['local_blue_clearance_verified']=bool(s.pop('local_red_position_verified') and safe)
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--axis',choices=['east','south','align-west','north-clear'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--approach-height-mm',type=int,choices=[30,50,70],default=30);parser.add_argument('--tangent-shift-mm',type=int,choices=[0,10,15,20],default=0)
    parser.add_argument('--south-wrist-yaw',type=int,choices=[0,90,180,270],default=90)
    parser.add_argument('--compact-lateral-face',action='store_true')
    args=parser.parse_args();restore,seed=configure(args.source);command,fingers,plan=prepare(args.axis,args.source,restore,args.approach_height_mm/1000,args.tangent_shift_mm,args.south_wrist_yaw,args.compact_lateral_face)
    args.out.mkdir(parents=True,exist_ok=False);checked=first.precheck(command,restore,seed,target=plan['target'],finger_command=fingers)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failures=len(checked['geometry_failures']),stages=len(checked['stages']),precheck_wall_seconds=checked['wall_seconds']);passed=checked['passed']
    if passed and args.run:
        first.run(command,plan,restore,args.out,finger_command=fingers,protected_payload=UPPER);brief['dynamics']=evaluate(args.out,plan);passed=brief['dynamics']['local_four_red_with_upper_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
