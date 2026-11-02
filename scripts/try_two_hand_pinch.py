"""Try a fixed opposed-finger pickup from the retained table-supported third brick.

The candidate is offline geometry. Actual dynamics must prove lift and a flat
three-second hold; no object reset, attachment or material change is used.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_third_yellow_feed as feed

g=feed.g;a=g.a


def evaluate(out,plan):
    record=json.loads((out/'result.json').read_text());record['summary']['unintended_contacts']=record['summary']['unintended_hand_contacts']
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));s=feed.evaluate(out)
    matrix=Rotation.from_quat(np.array(record['rows'][-1]['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
    tilt=float(np.degrees(np.arccos(np.clip(np.max(np.abs(matrix[2,:])),0,1))));axis=matrix[:,int(np.argmax(np.abs(matrix[0,:])))];yaw=float((np.degrees(np.arctan2(axis[1],axis[0]))+45)%90-45)
    held=[row for row in record['rows'] if row['elapsed_s']>=plan['hold_start_s']-1e-8];normal=np.array(plan['normal'])
    bilateral=bool(held and all(all(any(any(n.startswith(prefix) for n in c['pair']) for c in row['contacts']) for prefix in ('lh_','rh_')) for row in held))
    compression={prefix:min((sum(float(np.dot(c['force_xyz_n'],-sign*normal)) for c in row['contacts'] if any(n.startswith(prefix) for n in c['pair'])) for row in held),default=0.) for prefix,sign in [('lh_',1),('rh_',-1)]}
    held_tilt=max((float(np.degrees(np.arccos(np.clip(np.max(np.abs(Rotation.from_quat(np.array(row['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()[2,:])),0,1)))) for row in held),default=float('inf'))
    lift_start=plan['hold_start_s']-3
    closed=[row for row in record['rows'] if lift_start-1<=row['elapsed_s']<lift_start]
    early_lift=[row for row in record['rows'] if lift_start+.1<=row['elapsed_s']<=lift_start+.5]
    closure_force={prefix:min((sum(float(np.dot(c['force_xyz_n'],-sign*normal)) for c in row['contacts'] if any(n.startswith(prefix) for n in c['pair'])) for row in closed),default=0.) for prefix,sign in [('lh_',1),('rh_',-1)]}
    early_up=[sum(c['force_xyz_n'][2] for c in row['contacts'] if any(n.startswith(('lh_','rh_')) for n in c['pair'])) for row in early_lift]
    s['local_acquisition_verified'] &= s['protected_structure_verified']
    s.update(final_face_tilt_degrees=tilt,final_cube_yaw_mod90_degrees=yaw,held_bilateral_contacts=bilateral,held_minimum_inward_force_n=compression,
        maximum_bottom_mm=max(row['lowest_corner_m'] for row in record['rows'])*1000,closure_minimum_inward_force_n=closure_force,
        early_lift_hand_up_force_range_n=[min(early_up),max(early_up)] if early_up else None,early_lift_diagnostic_interval_s=[lift_start+.1,lift_start+.5],
        held_maximum_face_tilt_degrees=held_tilt if held else None,local_opposed_pinch_verified=bool(s['local_acquisition_verified'] and bilateral and min(compression.values())>=.1 and held_tilt<=2 and abs(yaw)<=10))
    record['summary']=s;(out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


def prepare(candidate,squeeze_mm=.5):
    if not np.isfinite(squeeze_mm) or not .5<=squeeze_mm<=6.:raise ValueError('Nominal side compression must be 0.5 to 6 mm')
    g.restore();starts={s:(g.d.mocap_pos[i].copy(),g.d.mocap_quat[i].copy()) for s,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    initial={g.m.actuator(i).name:float(g.d.ctrl[i]) for i in range(g.m.nu)}
    normal=np.array([np.cos(np.radians(candidate['angle_degrees'])),np.sin(np.radians(candidate['angle_degrees'])),0.])
    hands=candidate['hands'];targets={n:v for h in hands.values() for n,v in h['hand_targets'].items()}
    seeds=dict(zip(('left','right'),feed.seeds())) if not isinstance(g.CHECK['stages'],dict) else {s:g.CHECK['stages'][s][-1]['final_joints_sdk'] for s in starts}
    contexts=g.p.t.routes.context();entry={}
    for side,sign in [('left',1),('right',-1)]:
        high=np.array(hands[side]['root_position_m'])+sign*normal*.020+[0,0,.080]
        checked=g.a.g.endpoint(side,high,np.array(hands[side]['root_quaternion_wxyz']))
        if not checked['accepted']:raise ValueError(f'{side} elevated pinch approach has no verified IK endpoint')
        entry[side]=checked
    right_seed=np.array(seeds['right']);right_out=right_seed.copy();right_out[0]=-.5
    right_high=np.array(entry['right']['joints']);right_high_out=right_high.copy();right_high_out[0]=-.5
    right_waypoints=[right_seed,right_out,right_high_out,right_high]
    g.DURATION=a.DURATION=26.
    def command(t):
        poses={}
        for side,sign in [('left',1),('right',-1)]:
            root=np.array(hands[side]['root_position_m']);outside=sign*normal*.020
            quat=np.array(hands[side]['root_quaternion_wxyz'])
            if t<(12 if side=='right' else 6):
                if side=='right':
                    index=0 if t<3 else 1 if t<9 else 2;start,end=[(0,3),(3,9),(9,12)][index]
                    q=right_waypoints[index]+(right_waypoints[index+1]-right_waypoints[index])*(t-start)/(end-start)
                else:q=np.array(seeds[side])+(np.array(entry[side]['joints'])-seeds[side])*t/6
                fk=np.asarray(g.p.t.routes.sdk.forward_kinematics(side,q.tolist()))@contexts[side]['mount'];pos=fk[:3,3];quat=Rotation.from_matrix(fk[:3,:3]).as_quat()[[3,0,1,2]]
            elif t<13:pos=root+outside+[0,0,.080]
            elif t<16:pos=root+outside+np.array([0,0,.080])*(16-t)/3
            elif t<18:pos=root+outside*(18-t)/2
            elif t<20:pos=root
            else:pos=root+np.array([0,0,.040])*np.clip((t-20)/3,0,1)
            pos=pos-sign*normal*((squeeze_mm-.5)/1000)*np.clip((t-16)/2,0,1)
            poses[side]=(pos,quat)
        ctrl={n:initial[n]+(v-initial[n])*np.clip((t-3)/2,0,1) for n,v in targets.items()}
        phase='joint_entry' if t<12 else 'hold_over_target' if t<13 else 'descend_outside_faces' if t<16 else 'opposed_closure' if t<18 else 'hold_contact' if t<20 else 'common_lift' if t<23 else 'hold'
        return poses,ctrl,phase
    return command,dict(source=str(g.SOURCE),target=feed.TARGET,mode='opposed-pinch',hands=hands,normal=normal.tolist(),
        duration_s=26.,hold_start_s=23.,lift_m=.040,approach_gap_per_hand_m=.020,nominal_penetration_per_hand_m=squeeze_mm/1000,
        joint_entry=entry,joint_entry_seeds=seeds,right_joint_waypoints=[q.tolist() for q in right_waypoints],approach_clearance_m=.080,
        source_sequence_continuity_verified=True,online_object_feedback=False,native_verified=False,whole_arm_collision_checked=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,default=Path('runs/third-flat-113/opposed-pinch-candidate.json'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');parser.add_argument('--squeeze-mm',type=float,default=.5);args=parser.parse_args()
    report=json.loads(args.candidate.read_text());valid=[c for c in report['candidates'] if c['endpoint_candidate']]
    if len(valid)!=1:raise ValueError('Expected one geometrically valid, reachable candidate')
    source=Path(report['source']);a.configure_continuation(source,feed.TARGET,southern_receiver=True);feed.configure(source)
    left,right=feed.seeds();g.CHECK={**g.CHECK,'stages':{'left':[dict(final_joints_sdk=left)],'right':[dict(final_joints_sdk=right)]}}
    g.TARGET=feed.TARGET;command,plan=prepare(valid[0],args.squeeze_mm);plan['candidate_source']=str(args.candidate)
    args.out.mkdir(parents=True,exist_ok=False);check=g.precheck(command)
    if check['passed']:
        check['left_stages']=check['stages']['left'];check['right_lift']=check['stages']['right'][-1]['result']
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),stage_counts={s:len(v) for s,v in check['stages'].items()},precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        a.finger_commands=lambda t:command(t)[1]
        def motion(t):
            poses,_,phase=command(t);return poses,phase
        a.run(motion,args.out,protected_payload='neighbor_5',additional_protected_payloads=(feed.SECOND,*feed.BASES))
        brief['dynamics']=evaluate(args.out,plan);passed=brief['dynamics']['local_opposed_pinch_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
