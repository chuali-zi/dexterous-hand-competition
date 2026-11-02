"""Start third-yellow supply from the actual aligned four-red/two-yellow state.

Use a fixed south feed and preserve both released upper bricks. This does not
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


def prepare():
    release.configure(SOURCE,end_pair=True,released=True)
    if not g.RECORD['summary'].get('local_second_yellow_aligned',False):raise ValueError('Second yellow is not aligned')
    g.restore();initial={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)};targets=g.p.t.CANDIDATE['hand_targets'].copy()
    position=g.d.xpos[g.m.body(TARGET).id].copy()
    command,plan=first.prepare('south',SOURCE,g.restore,approach_height=.070,target_name=TARGET,goal_xy_m=[float(position[0]),.220],
                               contact_fingers=('index','middle'),hand_targets=targets,vertical_approach=True)
    def fingers(t):return {n:initial[n]+(v-initial[n])*np.clip(t/2,0,1) for n,v in targets.items()}
    plan.update(source_sequence_continuity_verified=True,structure_reference=str(release.REFERENCE),second_upper_reference=str(SOURCE),
                protected_upper_bricks=['neighbor_5',SECOND],right_seed_source=str(SOURCE/'precheck.json'))
    return command,fingers,plan


def evaluate(out):
    record=json.loads((out/'result.json').read_text());s=record['summary'];rows=record['rows']
    metrics,only,forces=release.fourth.third.structure_status(rows,release.REFERENCE,extra_bases=SUPPORTS)
    g.restore();adr=g.m.jnt_qposadr[g.m.joint(SECOND+'_free').id]
    xyz=np.array([r['qpos'][adr:adr+3] for r in rows]);rot=Rotation.from_quat(np.array([r['qpos'][adr+3:adr+7] for r in rows])[:,[1,2,3,0]])
    initial=Rotation.from_quat(g.d.xquat[g.m.body(SECOND).id][[1,2,3,0]])
    second_shift=float(np.linalg.norm(xyz-g.d.xpos[g.m.body(SECOND).id],axis=1).max())
    second_angle=float(np.degrees((initial.inv()*rot).magnitude().max()))
    contacts=[r['additional_protected_payload_contacts'][SECOND] for r in rows]
    second_only=all(cs and all(all(n in [SECOND,*SUPPORTS] for n in c['pair']) for c in cs) for cs in contacts)
    second_forces={n:min(sum(c['force_xyz_n'][2] for c in cs if n in c['pair']) for cs in contacts) for n in SUPPORTS}
    centers=[np.array(rows[-1]['qpos'])[g.m.jnt_qposadr[g.m.joint(n+'_free').id]:g.m.jnt_qposadr[g.m.joint(n+'_free').id]+3] for n in SUPPORTS]
    offset=xyz[-1,:2]-np.mean(centers,axis=0)[:2]
    safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1
              and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in metrics.values())
              and second_shift<=.002 and second_angle<=2 and second_only and min(second_forces.values())>=.1 and np.linalg.norm(offset)<=.005)
    s.update(source_sequence_continuity_verified=True,protected_structure_verified=safe,protected_structure_metrics=metrics,
             upper_only_base_contacts=only,upper_minimum_support_force_n=forces,second_upper_maximum_shift_mm=second_shift*1000,
             second_upper_maximum_angular_shift_degrees=second_angle,second_upper_only_base_contacts=second_only,
             second_upper_minimum_support_force_n=second_forces,second_upper_actual_center_error_mm=float(np.linalg.norm(offset)*1000),
             local_third_yellow_feed_verified=bool(s.pop('local_red_position_verified') and safe))
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    command,fingers,plan=prepare();args.out.mkdir(parents=True,exist_ok=False)
    check=first.precheck(command,g.restore,g.CHECK['stages']['left'][-1]['result']['final_joints_sdk'],target=TARGET,finger_command=fingers)
    check['right_final_joints_sdk']=g.CHECK['stages']['right'][-1]['result']['final_joints_sdk'];check['stationary_right']=True
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),stage_count=len(check['stages']),precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        first.run(command,plan,g.restore,args.out,finger_command=fingers,protected_payload='neighbor_5',additional_protected_payloads=(SECOND,))
        brief['dynamics']=evaluate(args.out);passed=brief['dynamics']['local_third_yellow_feed_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
