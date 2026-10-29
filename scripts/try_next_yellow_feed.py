"""Feed the next yellow from a retained three-red/one-yellow structure.

First clear the northern obstacle laterally, then use a narrow centered pushing
face. Fixed offline commands, full-state continuation, existing structure kept.
"""
import argparse,json
from pathlib import Path
import numpy as np
import try_third_red_with_upper as structure
first=structure.first;m,d,mj,p=structure.m,structure.d,structure.mj,structure.p
TARGET='neighbor_3';REFERENCE=Path('runs/third-red-103/final-south')


def configure(source):
    record=json.loads((source/'result.json').read_text());s=record['summary'];checked=json.loads((source/'precheck.json').read_text())
    initial=source.resolve()==REFERENCE.resolve() and s.get('local_three_red_with_upper_verified',False)
    continuing=(s.get('target')==TARGET and s.get('protected_structure_verified',False) and s.get('held_duration_s',0)>=2.98
                and s.get('held_table_only',False) and s.get('held_drift_mm',1e6)<=2 and s.get('held_angular_drift_degrees',1e6)<=2
                and s.get('final_face_tilt_degrees',1e6)<=2 and s.get('minimum_table_support_n',0)>=.8*m.body(TARGET).mass*9.81)
    if not (initial or continuing) or not s.get('source_sequence_continuity_verified',False) or s['abort'] or s['unintended_contacts'] or not checked['passed'] or s['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source is not a stable protected continuation')
    state=np.load(source/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=mj.MjData(m);mj.mj_setState(m,candidate,state['state'],int(state['spec']));mj.mj_forward(m,candidate)
    np.testing.assert_allclose(candidate.qpos,record['rows'][-1]['qpos'],atol=1e-12,rtol=0);np.testing.assert_allclose(candidate.qvel,record['rows'][-1]['qvel'],atol=1e-12,rtol=0)
    def restore():mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    return restore,checked['stages'][-1]['result']['final_joints_sdk']


def prepare(mode,source,restore,double_finger=False,approach_height=.070,deploy_middle=False):
    if deploy_middle and (mode!='south' or double_finger):raise ValueError('Middle deployment compares narrow-entry south feeding only')
    restore();position=d.xpos[m.body(TARGET).id].copy();initial={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)}
    goal=[.289,float(position[1])] if mode=='west-clear' else [float(position[0]),.140] if mode=='south' else [.227,float(position[1])]
    targets=p.t.CANDIDATE['hand_targets'].copy()
    if not double_finger:
        for suffix in ['_mcp_pitch','_dip']:targets['lh_middle'+suffix]=1.45
    contact_fingers=('index','middle') if double_finger else ('index',)
    axis='south' if mode=='south' else 'west'
    command,plan=first.prepare(axis,source,restore,approach_height=approach_height,adapt_lateral_face=axis=='west',
                               target_name=TARGET,goal_xy_m=goal,contact_fingers=contact_fingers,hand_targets=targets,vertical_approach=axis=='south')
    def fingers(t):return {n:initial[n]+(v-initial[n])*np.clip(t/2,0,1) for n,v in targets.items()}
    if deploy_middle:
        double_targets=p.t.CANDIDATE['hand_targets'].copy()
        _,double_plan=first.prepare(axis,source,restore,approach_height=approach_height,target_name=TARGET,goal_xy_m=goal,
                                    hand_targets=double_targets,vertical_approach=True)
        shift=np.array(double_plan['contact_root_m'])-plan['contact_root_m'];original_command=command;original_fingers=fingers
        def command(t):
            pos,q,phase=original_command(t)
            return pos+shift*np.clip((t-11)/1,0,1),q,'deploy_centered_middle' if 11<=t<12 else phase
        def fingers(t):
            ctrl=original_fingers(t)
            for suffix in ['_mcp_pitch','_dip']:
                name='lh_middle'+suffix;ctrl[name]=ctrl[name]+(double_targets[name]-ctrl[name])*np.clip((t-11)/1,0,1)
            return ctrl
        plan.update(middle_deployment_s=[11,12],two_finger_recenter_m=shift.tolist(),final_left_finger_targets=double_targets)
    plan.update(mode=mode,structure_reference=str(REFERENCE),protected_structure=[*structure.BASES,structure.UPPER,'neighbor_4'],
                initial_left_controls={n:initial[n] for n in targets},left_finger_targets=targets,source_sequence_continuity_verified=True)
    return command,fingers,plan


def evaluate(out,plan):
    record=json.loads((out/'result.json').read_text());s=record['summary']
    protected,only,forces=structure.structure_status(record['rows'],REFERENCE,extra_bases=('neighbor_4',))
    safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1
              and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in protected.values()))
    s.update(protected_structure_metrics=protected,protected_structure_verified=safe,upper_only_base_contacts=only,upper_minimum_support_force_n=forces,
             source_sequence_continuity_verified=True,local_feed_position_verified=bool(s.pop('local_red_position_verified') and safe),
             local_acquisition_verified=False)
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['west-clear','south','west-pickup'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--double-finger',action='store_true');parser.add_argument('--approach-height-mm',type=int,choices=[30,50,70],default=70)
    parser.add_argument('--deploy-middle-after-clearance',action='store_true')
    args=parser.parse_args();restore,seed=configure(args.source);command,fingers,plan=prepare(args.mode,args.source,restore,args.double_finger,args.approach_height_mm/1000,args.deploy_middle_after_clearance)
    args.out.mkdir(parents=True,exist_ok=False);checked=first.precheck(command,restore,seed,target=TARGET,finger_command=fingers)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failures=len(checked['geometry_failures']),stages=len(checked['stages']),precheck_wall_seconds=checked['wall_seconds']);passed=checked['passed']
    if passed and args.run:
        first.run(command,plan,restore,args.out,finger_command=fingers,protected_payload=structure.UPPER);brief['dynamics']=evaluate(args.out,plan);passed=brief['dynamics']['local_feed_position_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
