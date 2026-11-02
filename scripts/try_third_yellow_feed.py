"""Continue third-yellow supply while preserving the four-red/two-yellow state.

Use fixed north, west and south feed segments. This does not
claim third-yellow acquisition or placement, nor native platform verification.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_shared_yellow_release as release

g=release.g;first=release.fourth.first
SOURCE=Path('runs/settled-release-109/short-push')
TARGET='neighbor_1';SECOND='neighbor_3';SUPPORTS=('neighbor_4','neighbor_2')
SECOND_REFERENCE=SOURCE
BASES=('neighbor_9','neighbor_6','neighbor_4','neighbor_2')


def configure(source):
    global SOURCE
    SOURCE=Path(source)
    if SOURCE.resolve()==SECOND_REFERENCE.resolve():
        release.configure(SOURCE,end_pair=True,released=True)
        if not g.RECORD['summary'].get('local_second_yellow_aligned',False):raise ValueError('Second yellow is not aligned')
        return
    record=json.loads((SOURCE/'result.json').read_text());s=record['summary'];check=json.loads((SOURCE/'precheck.json').read_text())
    if (s.get('target')!=TARGET or not s.get('protected_structure_verified',False) or not s.get('source_sequence_continuity_verified',False)
            or s['abort'] or s['unintended_contacts'] or not check['passed'] or not s.get('held_table_only',False) or s.get('held_duration_s',0)<2.98
            or s['held_drift_mm']>2 or s['held_angular_drift_degrees']>2 or s['final_face_tilt_degrees']>2
            or s['minimum_table_support_n']<.8*g.m.body(TARGET).mass*9.81 or s['second_upper_actual_center_error_mm']>5
            or s['checkpoint_next_step_maximum_difference']>1e-10):raise ValueError('Source is not a stable table feed preserving both upper bricks')
    state=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=g.mj.__version__ or len(state['state'])!=g.mj.mj_stateSize(g.m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    candidate=g.mj.MjData(g.m);g.mj.mj_setState(g.m,candidate,state['state'],int(state['spec']));g.mj.mj_forward(g.m,candidate)
    np.testing.assert_allclose(candidate.qpos,record['rows'][-1]['qpos'],rtol=0,atol=1e-12);np.testing.assert_allclose(candidate.qvel,record['rows'][-1]['qvel'],rtol=0,atol=1e-12)
    def restore():g.mj.mj_setState(g.m,g.d,state['state'],int(state['spec']));g.mj.mj_forward(g.m,g.d)
    g.SOURCE=SOURCE;g.RECORD=record;g.STATE=state;g.CHECK=check;g.restore=restore


def seeds():
    check=g.CHECK
    return (check['stages']['left'][-1]['result']['final_joints_sdk'],check['stages']['right'][-1]['result']['final_joints_sdk']) if isinstance(check['stages'],dict) else (check['stages'][-1]['result']['final_joints_sdk'],check['right_final_joints_sdk'])


def prepare(mode='south-stage',north_yaw=90,north_goal=.240,single_finger=False):
    if north_goal not in [.240,.270]:raise ValueError('Unsupported northern staging position')
    if single_finger and not mode.startswith('west'):raise ValueError('Narrow contact compares west feeding only')
    g.restore();initial={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)};targets=g.p.t.CANDIDATE['hand_targets'].copy()
    if single_finger:
        for suffix in ['_mcp_pitch','_dip']:targets['lh_middle'+suffix]=1.45
    position=g.d.xpos[g.m.body(TARGET).id].copy()
    goals={'south-stage':[float(position[0]),.220],'north-clear':[float(position[0]),north_goal],
           'west-stage':[.180,float(position[1])],'west-edge':[.165,float(position[1])],
           'south-mid':[float(position[0]),.140],'south-pickup':[float(position[0]),.080]}
    if mode not in goals:raise ValueError('Unknown feed mode')
    axis='west' if mode.startswith('west') else 'north' if mode=='north-clear' else 'south'
    if axis=='north':
        command,plan=release.fourth.third.second.prepare(axis,SOURCE,g.restore,approach_height=.070,target_name=TARGET,goal_xy_m=goals[mode],hand_targets=targets,vertical_approach=True,south_wrist_yaw=north_yaw)
    else:
        command,plan=first.prepare(axis,SOURCE,g.restore,approach_height=.070,target_name=TARGET,goal_xy_m=goals[mode],
                                   contact_fingers=('index',) if single_finger else ('index','middle'),hand_targets=targets,vertical_approach=axis=='south',adapt_lateral_face=axis=='west')
    def fingers(t):return {n:initial[n]+(v-initial[n])*np.clip(t/2,0,1) for n,v in targets.items()}
    plan.update(mode=mode,single_finger=single_finger,source_sequence_continuity_verified=True,structure_reference=str(release.REFERENCE),second_upper_reference=str(SECOND_REFERENCE),
                protected_upper_bricks=['neighbor_5',SECOND],right_seed_source=str(SOURCE/'precheck.json'))
    return command,fingers,plan


def current_structure(rows,reference,allow_third_yellow_contact=False):
    # Fixed local construction target, fitted once to the retained six-brick
    # checkpoint. These are engineering bounds, not an official score checker.
    names=[*BASES,'neighbor_5',SECOND]
    ideal=np.array([[0,0,.025],[0,.05,.025],[0,.10,.025],[0,.15,.025],[0,.025,.075],[0,.125,.075]])
    translation=np.mean(np.array([reference.xpos[g.m.body(n).id,:2] for n in names])-ideal[:,:2],axis=0)
    ideal[:,:2]+=translation;ideal[:,2]+=g.a.g.FLOOR
    metrics={};base_forces={};allowed=set([*names,'table']);base_contacts_ok=True
    for i,n in enumerate(names):
        adr=g.m.jnt_qposadr[g.m.joint(n+'_free').id];xyz=np.array([r['qpos'][adr:adr+3] for r in rows])
        rotations=Rotation.from_quat(np.array([r['qpos'][adr+3:adr+7] for r in rows])[:,[1,2,3,0]])
        matrices=rotations.as_matrix();face=np.degrees(np.arccos(np.clip(np.max(np.abs(matrices[:,2,:]),axis=1),0,1)))
        # A cube may rest on any face; Euler-Z is not its top-face heading.
        axes=np.take_along_axis(matrices,np.argmax(np.abs(matrices[:,0,:]),axis=1)[:,None,None],axis=2)[:,:,0]
        yaw=(np.degrees(np.arctan2(axes[:,1],axes[:,0]))+45)%90-45
        metrics[n]=dict(maximum_horizontal_error_mm=float(np.linalg.norm(xyz[:,:2]-ideal[i,:2],axis=1).max()*1000),
                        maximum_vertical_error_mm=float(np.abs(xyz[:,2]-ideal[i,2]).max()*1000),maximum_face_tilt_degrees=float(face.max()),
                        maximum_abs_yaw_degrees=float(np.abs(yaw).max()),segment_angular_drift_degrees=float(np.degrees((rotations[0].inv()*rotations).magnitude().max())))
        if n in BASES:
            cs=[r.get('additional_protected_payload_contacts',{}).get(n,[]) for r in rows]
            permitted=allowed|({TARGET} if allow_third_yellow_contact and n in ('neighbor_6','neighbor_4') else set())
            base_contacts_ok &= all(c and all(all(x in permitted for x in entry['pair']) for entry in c) for c in cs)
            base_forces[n]=min(sum(c['force_xyz_n'][2] for c in contacts if 'table' in c['pair']) for contacts in cs)
    valid=bool(base_contacts_ok and all(base_forces[n]>=.8*g.m.body(n).mass*9.81 for n in BASES)
               and all(v['maximum_horizontal_error_mm']<=15 and v['maximum_vertical_error_mm']<=5
                       and v['maximum_face_tilt_degrees']<=2 and v['maximum_abs_yaw_degrees']<=10 and v['segment_angular_drift_degrees']<=2 for v in metrics.values()))
    return dict(passed=valid,metrics=metrics,base_minimum_table_support_n=base_forces,base_contacts_valid=bool(base_contacts_ok),
                fixed_xy_translation_m=translation.tolist(),target_reference=str(SECOND_REFERENCE),horizontal_limit_mm=15,vertical_limit_mm=5,
                third_yellow_contact_allowed_on_middle_pair=allow_third_yellow_contact,
                scope='Current six-brick local geometry and recorded table support; not official scoring or full pyramid verification')


def evaluate(out,allow_third_yellow_contact=False):
    record=json.loads((out/'result.json').read_text());s=record['summary'];rows=record['rows']
    metrics,only,forces=release.fourth.third.structure_status(rows,release.REFERENCE,extra_bases=SUPPORTS)
    reference=np.load(SECOND_REFERENCE/'integration-state.npz',allow_pickle=False);ref=g.mj.MjData(g.m)
    g.mj.mj_setState(g.m,ref,reference['state'],int(reference['spec']));g.mj.mj_forward(g.m,ref);adr=g.m.jnt_qposadr[g.m.joint(SECOND+'_free').id]
    xyz=np.array([r['qpos'][adr:adr+3] for r in rows]);rot=Rotation.from_quat(np.array([r['qpos'][adr+3:adr+7] for r in rows])[:,[1,2,3,0]])
    initial=Rotation.from_quat(ref.xquat[g.m.body(SECOND).id][[1,2,3,0]])
    second_shift=float(np.linalg.norm(xyz-ref.xpos[g.m.body(SECOND).id],axis=1).max())
    second_angle=float(np.degrees((initial.inv()*rot).magnitude().max()))
    contacts=[r['additional_protected_payload_contacts'][SECOND] for r in rows]
    second_only=all(cs and all(all(n in [SECOND,*SUPPORTS] for n in c['pair']) for c in cs) for cs in contacts)
    second_forces={n:min(sum(c['force_xyz_n'][2] for c in cs if n in c['pair']) for cs in contacts) for n in SUPPORTS}
    centers=[np.array(rows[-1]['qpos'])[g.m.jnt_qposadr[g.m.joint(n+'_free').id]:g.m.jnt_qposadr[g.m.joint(n+'_free').id]+3] for n in SUPPORTS]
    offset=xyz[-1,:2]-np.mean(centers,axis=0)[:2]
    legacy_safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1
              and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in metrics.values())
              and second_shift<=.002 and second_angle<=2 and second_only and min(second_forces.values())>=.1 and np.linalg.norm(offset)<=.005)
    structure=current_structure(rows,ref,allow_third_yellow_contact)
    first_centers=[]
    for n in ('neighbor_9','neighbor_6'):
        address=g.m.jnt_qposadr[g.m.joint(n+'_free').id];first_centers.append(np.array(rows[-1]['qpos'][address:address+3]))
    address=g.m.jnt_qposadr[g.m.joint('neighbor_5_free').id]
    first_offset=np.array(rows[-1]['qpos'][address:address+3])[:2]-np.mean(first_centers,axis=0)[:2]
    safe=bool(not s['abort'] and not s['unintended_contacts'] and structure['passed'] and only and min(forces.values())>=.1
              and second_only and min(second_forces.values())>=.1 and np.linalg.norm(offset)<=.005 and np.linalg.norm(first_offset)<=.005
              and max(s['other_bricks_max_shift_mm'].values())<=2 and s['checkpoint_next_step_maximum_difference']<=1e-10)
    position_pass=bool(s.pop('local_red_position_verified',s.get('candidate_feed_position_verified',False)))
    s.update(source_sequence_continuity_verified=True,protected_structure_verified=safe,protected_structure_metrics=metrics,
             upper_only_base_contacts=only,upper_minimum_support_force_n=forces,second_upper_maximum_shift_mm=second_shift*1000,
             second_upper_maximum_angular_shift_degrees=second_angle,second_upper_only_base_contacts=second_only,
             second_upper_minimum_support_force_n=second_forces,second_upper_actual_center_error_mm=float(np.linalg.norm(offset)*1000),
             current_structure=structure,legacy_cumulative_protection_passed=legacy_safe,
             cumulative_displacement_warning=bool(second_shift>.002 or any(v['maximum_shift_mm']>2 for v in metrics.values())),
             first_upper_actual_center_error_mm=float(np.linalg.norm(first_offset)*1000),
             candidate_feed_position_verified=position_pass,local_third_yellow_feed_verified=bool(position_pass and safe))
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


def payload_precheck(plan):
    g.restore();adr=g.m.jnt_qposadr[g.m.joint(TARGET+'_free').id];initial=g.d.qpos[adr:adr+3].copy();delta=np.zeros(3)
    delta[plan['component']]=plan['destination_axis_m']-initial[plan['component']];bad=[]
    for fraction in np.linspace(0,1,151):
        g.restore();g.d.qpos[adr:adr+3]=initial+delta*fraction;g.mj.mj_forward(g.m,g.d)
        for c in g.d.contact:
            ns=[g.m.geom(int(i)).name for i in c.geom]
            if TARGET in ns and any(n=='target' or (n.startswith('neighbor_') and n!=TARGET) for n in ns):
                bad.append(dict(fraction=float(fraction),pair=ns,depth_mm=-float(c.dist)*1000))
    g.restore();return bad


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--mode',choices=['south-stage','north-clear','west-stage','west-edge','south-mid','south-pickup'],default='south-stage')
    parser.add_argument('--north-yaw',type=int,choices=[0,90,180,270],default=90);parser.add_argument('--north-goal-mm',type=int,choices=[240,270],default=240)
    parser.add_argument('--single-finger',action='store_true');args=parser.parse_args()
    configure(args.source);command,fingers,plan=prepare(args.mode,args.north_yaw,args.north_goal_mm/1000,args.single_finger);args.out.mkdir(parents=True,exist_ok=False);left_seed,right_seed=seeds()
    check=first.precheck(command,g.restore,left_seed,target=TARGET,finger_command=fingers)
    check['predicted_payload_geometry_failures']=payload_precheck(plan);check['passed'] &= not check['predicted_payload_geometry_failures']
    check['payload_prediction_scope']='151 constant-orientation translated-payload geometry samples only; dynamics are not bound to this path'
    check['right_final_joints_sdk']=right_seed;check['stationary_right']=True
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),payload_geometry_failure_count=len(check['predicted_payload_geometry_failures']),stage_count=len(check['stages']),precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        first.run(command,plan,g.restore,args.out,finger_command=fingers,protected_payload='neighbor_5',additional_protected_payloads=(SECOND,*BASES))
        brief['dynamics']=evaluate(args.out);passed=brief['dynamics']['local_third_yellow_feed_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
