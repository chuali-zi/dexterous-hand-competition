"""Push the green and next yellow north together to open the pickup corridor.

Only these two supply bricks may move. Existing four bases and the released
upper yellow remain protected. Fixed local commands, no online truth control.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_receiver_return as receiver
import try_next_yellow_feed as feed
m,d,mj,p=receiver.m,receiver.d,receiver.mj,receiver.p
first=receiver.first
SOURCE=Path('runs/next-pickup-106/right-return');SECONDARY='neighbor_3'


def prepare():
    s=json.loads((SOURCE/'result.json').read_text())['summary']
    if not s.get('local_receiver_return_verified',False):raise ValueError('The continuous receiver return has not passed')
    restore,seed=feed.configure(SOURCE);restore();initial={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)}
    goals=p.t.CANDIDATE['hand_targets'].copy();position=d.xpos[m.body('target').id].copy()
    command,plan=feed.structure.second.prepare('north',SOURCE,restore,target_name='target',goal_xy_m=[float(position[0]),.175],hand_targets=goals,vertical_approach=True,approach_height=.070)
    def fingers(t):return {n:initial[n]+(v-initial[n])*np.clip(t/2,0,1) for n,v in goals.items()}
    plan.update(planned_secondary_targets=[SECONDARY],structure_reference=str(receiver.SOURCE),source_sequence_continuity_verified=True)
    return restore,seed,command,fingers,plan


def evaluate(out):
    record=json.loads((out/'result.json').read_text());s=record['summary'];rows=record['rows'];held=[r for r in rows if r['elapsed_s']>=20-1e-8]
    metrics,only,forces=receiver.fourth.third.structure_status(rows,receiver.SOURCE,extra_bases=('neighbor_4','neighbor_2'))
    safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1 and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in metrics.values()))
    released={}
    for name in ['target',SECONDARY]:
        if not held:continue
        adr=m.jnt_qposadr[m.joint(name+'_free').id];xyz=np.array([r['qpos'][adr:adr+3] for r in held]);rot=Rotation.from_quat(np.array([r['qpos'][adr+3:adr+7] for r in held])[:,[1,2,3,0]])
        contacts=[r['contacts'] if name=='target' else r['secondary_contacts'][name] for r in held]
        # Final continuation requires separation: each supply brick must
        # independently receive its weight from the table with no side contact.
        table_only=all(cs and all('table' in c['pair'] for c in cs) for cs in contacts)
        up=min(sum(c['force_xyz_n'][2] for c in cs if 'table' in c['pair']) for cs in contacts)
        drift=float(np.linalg.norm(xyz-xyz[0],axis=1).max()*1000);angle=float(np.degrees((rot[0].inv()*rot).magnitude().max()))
        face=float(np.degrees(np.arccos(np.clip(np.max(np.abs(rot[-1].as_matrix()[2])),0,1))))
        released[name]=dict(held_duration_s=held[-1]['elapsed_s']-held[0]['elapsed_s'],held_table_only=table_only,minimum_table_support_n=up,
                           held_drift_mm=drift,held_angular_drift_degrees=angle,final_face_tilt_degrees=face,final_position_m=xyz[-1].tolist(),
                           final_yaw_mod90_degrees=float((rot[-1].as_euler('xyz',degrees=True)[2]+45)%90-45))
        released[name]['released_stable']=bool(table_only and up>=.8*m.body(name).mass*9.81 and drift<=2 and angle<=2 and face<=2 and released[name]['held_duration_s']>=2.98)
    s.update(protected_structure_metrics=metrics,protected_structure_verified=safe,upper_only_base_contacts=only,upper_minimum_support_force_n=forces,
             released_supply_targets=released,source_sequence_continuity_verified=True,local_acquisition_verified=False,
             local_coupled_feed_released=bool(safe and len(released)==2 and all(v['released_stable'] for v in released.values()) and s['checkpoint_next_step_maximum_difference']<=1e-10))
    s.pop('local_red_position_verified',None)
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    restore,seed,command,fingers,plan=prepare();check=first.precheck(command,restore,seed,target='target',finger_command=fingers,additional_targets=(SECONDARY,))
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    brief=dict(precheck_passed=check['passed'],geometry_failures=len(check['geometry_failures']),stages=len(check['stages']),precheck_wall_seconds=check['wall_seconds']);passed=check['passed']
    if passed and args.run:
        first.run(command,plan,restore,args.out,finger_command=fingers,protected_payload=receiver.fourth.UPPER,additional_targets=(SECONDARY,));brief['dynamics']=evaluate(args.out);passed=brief['dynamics']['local_coupled_feed_released']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
