"""Continue the retained first red with second-red transfer and alignment.

Offline teaching only. Preserve the first base and all other bricks; no resets,
native simulation, or online object truth control. Local assets are required.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import try_replanned_first_red as first

m,d,mj,p=first.m,first.d,first.mj,first.p
TARGET='neighbor_6';BASE='neighbor_9'


def configure(axis,source):
    record=json.loads((source/'result.json').read_text());summary=record['summary'];check=json.loads((source/'precheck.json').read_text())
    if axis=='east':
        valid=summary.get('axis')=='finish-south' and summary.get('local_red_position_verified',False)
    else:
        valid=(summary.get('target')==TARGET and summary.get('axis') in (['east'] if axis=='south' else ['south','west']) and summary.get('held_duration_s',0)>=2.98
               and summary.get('held_table_only',False) and summary.get('held_drift_mm',1e6)<=2 and summary.get('held_angular_drift_degrees',1e6)<=2
               and summary.get('final_face_tilt_degrees',1e6)<=2 and summary.get('minimum_table_support_n',0)>=.8*m.body(TARGET).mass*9.81
               and max(summary['other_bricks_max_shift_mm'].values())<=2)
    if not valid or summary['abort'] or summary['unintended_contacts'] or not check['passed'] or summary['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source does not satisfy this stage continuation')
    state=np.load(source/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    def restore():
        mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    restore()
    if not np.allclose(d.qpos,record['rows'][-1]['qpos'],atol=1e-12) or not np.allclose(d.qvel,record['rows'][-1]['qvel'],atol=1e-12):raise ValueError('State and trace differ')
    return restore,check['stages'][-1]['result']['final_joints_sdk']


def prepare(axis,source,restore,tangent_shift_mm=0,diagonal_align=False,target_name=TARGET,goal_xy_m=None,hand_targets=None,vertical_approach=False,approach_height=.070,south_wrist_yaw=90):
    if approach_height not in [.030,.050,.070]:raise ValueError('Unsupported local approach height')
    if south_wrist_yaw not in [0,90,180,270] or (south_wrist_yaw!=90 and axis!='south'):raise ValueError('Alternate wrist orientation applies only to south pushing')
    goal=goal_xy_m or [.3025,.015]
    if diagonal_align and axis!='align-west':raise ValueError('Diagonal correction requires the adapted west face')
    if axis.startswith('align-'):
        if tangent_shift_mm:raise ValueError('Alignment uses a centered contact face')
        command,plan=first.prepare(axis.removeprefix('align-'),source,restore,approach_height=approach_height,adapt_lateral_face=axis=='align-west',target_name=target_name,goal_xy_m=goal)
        if diagonal_align:
            dy=goal[1]-plan['initial_target_center_m'][1]
            if abs(dy)>.020:raise ValueError('Diagonal correction exceeds the short local range')
            original=command
            def command(seconds):
                pos,q,phase=original(seconds);pos=pos.copy();pos[1]+=dy*np.clip((seconds-10)/6,0,1)
                return pos,q,phase
            plan['diagonal_y_displacement_m']=dy
        return command,plan
    restore();start=d.mocap_pos[p.t.LM].copy();startq=d.mocap_quat[p.t.LM].copy();target=d.xpos[m.body(target_name).id].copy()
    corners=Rotation.from_quat(d.xquat[m.body(target_name).id][[1,2,3,0]]).apply(p.CORNERS)+target
    if hand_targets is not None:
        for name,value in hand_targets.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    normal=np.array([-1.,0,0]) if axis=='east' else np.array([0.,-1. if axis=='north' else 1.,0]);tangent=np.array([-normal[1],normal[0],0.])
    quat=(Rotation.from_euler('z',0 if axis=='east' else south_wrist_yaw,degrees=True)*Rotation.from_quat(p.t.QUAT[[1,2,3,0]])).as_quat()[[3,0,1,2]]
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=quat;mj.mj_forward(m,d)
    vertices=np.concatenate([first.b.a.placement.a.g.mesh_vertices('lh_'+n) for n in ['index_distal','index_proximal','middle_distal','middle_proximal']])
    root=normal*((corners@normal).max()-(vertices@normal).min()-.0005)
    root+=tangent*(target@tangent-((vertices@tangent).min()+(vertices@tangent).max())/2);root[2]=p.t.ROOT[2]
    if tangent_shift_mm not in [0,10,15,20] or (axis=='east' and tangent_shift_mm):raise ValueError('Unsupported contact-face offset')
    root+=tangent*tangent_shift_mm/1000
    # The retained blue push moved 61.269 mm for 80 mm hand travel. Use it as
    # an explicitly recorded offline teaching estimate, not a dynamics model.
    lost_travel=.080-.06126927207210336 if axis=='east' else 0.
    distance=goal[0]-target[0]+lost_travel if axis=='east' else goal[1]-target[1] if axis=='north' else target[1]-goal[1]
    if not .001<distance<.150:raise ValueError('Move outside this trial range')
    approach=0. if vertical_approach else .020
    above=root+approach*normal+[0,0,approach_height];turn=Slerp([0.,1.],Rotation.from_quat(np.array([startq,quat])[:,[1,2,3,0]]))
    def command(seconds):
        if seconds<2:pos=start+[0,0,.030*seconds/2];q=startq;phase='rise'
        elif seconds<6:
            f=(seconds-2)/4;pos=(start+[0,0,.030])*(1-f)+above*f;q=turn(f).as_quat()[[3,0,1,2]];phase='transfer_above'
        elif seconds<9:pos=root+approach*normal+[0,0,approach_height*(9-seconds)/3];q=quat;phase='descend'
        elif seconds<10:pos=root+approach*(10-seconds)*normal;q=quat;phase='approach'
        elif seconds<16:pos=root-distance*(seconds-10)/6*normal;q=quat;phase='push_red'
        elif seconds<18:pos=root-distance*normal+[0,0,.050*(seconds-16)/2];q=quat;phase='lift_clear'
        elif seconds<20:pos=root+(-distance+.020*(seconds-18)/2)*normal+[0,0,.050];q=quat;phase='retreat'
        else:pos=root+(-distance+.020)*normal+[0,0,.050];q=quat;phase='released_hold'
        return pos,q,phase
    restore()
    return command,dict(source=str(source),target=target_name,axis=axis,initial_target_center_m=target.tolist(),component=0 if axis=='east' else 1,
                        destination_axis_m=goal[0] if axis=='east' else goal[1],goal_xy_m=[goal[0],float(target[1]) if axis=='east' else goal[1]],
                        contact_root_m=root.tolist(),distance_m=distance,tangent_shift_mm=tangent_shift_mm,offline_lost_travel_estimate_m=lost_travel,
                        duration_s=23.,hold_start_s=20.,approach_offset_m=approach,approach_height_m=approach_height,south_wrist_yaw_degrees=south_wrist_yaw if axis=='south' else None,whole_arm_collision_checked=False,native_verified=False,official_region_verified=False)


def evaluate_pair(out,plan):
    record=json.loads((out/'result.json').read_text());summary=record['summary'];state=np.load(out/'integration-state.npz',allow_pickle=False)
    mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    positions={n:d.xpos[m.body(n).id].copy() for n in [BASE,TARGET]};rot={n:Rotation.from_quat(d.xquat[m.body(n).id][[1,2,3,0]]) for n in positions}
    corners={n:rot[n].apply(p.CORNERS)+positions[n] for n in positions}
    dx=abs(float(positions[TARGET][0]-positions[BASE][0]));dy=float(positions[TARGET][1]-positions[BASE][1])
    gap=float(corners[TARGET][:,1].min()-corners[BASE][:,1].max());yaw={n:float((rot[n].as_euler('xyz',degrees=True)[2]+45)%90-45) for n in positions}
    address=int(m.jnt_qposadr[int(m.body(BASE).jntadr[0])]);base_quats=np.array([r['qpos'][address+3:address+7] for r in record['rows']])
    base_rot=Rotation.from_quat(base_quats[:,[1,2,3,0]])
    source_record=json.loads((Path(plan['source'])/'result.json').read_text())
    source_rot=Rotation.from_quat(np.array(source_record['rows'][-1]['qpos'][address+3:address+7])[[1,2,3,0]])
    base_angle=float(np.degrees((source_rot.inv()*base_rot).magnitude().max()))
    base_face=float(np.degrees(np.arccos(np.clip(np.max(np.abs(rot[BASE].as_matrix()[2,:])),0,1))))
    summary.update(pair_x_offset_mm=dx*1000,pair_y_spacing_mm=dy*1000,pair_projected_gap_mm=gap*1000,pair_yaw_degrees=yaw,
                   protected_base_angular_shift_degrees=base_angle,protected_base_face_tilt_degrees=base_face,
                   local_two_red_support_verified=bool(plan['axis']!='east' and summary['local_red_position_verified'] and dx<=.005 and .050<=dy<=.060
                                                       and 0<=gap<=.010 and all(abs(v)<=10 for v in yaw.values()) and summary['other_bricks_max_shift_mm'][BASE]<=2
                                                       and base_angle<=2 and base_face<=2))
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--axis',choices=['east','south','align-south','align-west'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--tangent-shift-mm',type=int,choices=[0,10,15,20],default=0)
    parser.add_argument('--diagonal-align',action='store_true')
    args=parser.parse_args();restore,seed=configure(args.axis,args.source);command,plan=prepare(args.axis,args.source,restore,args.tangent_shift_mm,args.diagonal_align);args.out.mkdir(parents=True,exist_ok=False)
    checked=first.precheck(command,restore,seed,target=TARGET)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),stages=len(checked['stages']),precheck_wall_seconds=checked['wall_seconds']);passed=checked['passed']
    if args.run and passed:
        first.run(command,plan,restore,args.out);brief['dynamics']=evaluate_pair(args.out,plan)
        passed=brief['dynamics']['local_red_position_verified'] if args.axis=='east' else brief['dynamics']['local_two_red_support_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
