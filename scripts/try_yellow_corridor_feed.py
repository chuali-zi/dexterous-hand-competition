"""Feed yellow 5 into the cleared supply corridor from the actual red pair.

Fixed local contact trajectories only; original physical state is continued.
This does not acquire a brick or verify any upper-layer placement.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import try_replanned_first_red as first

m,d,mj,p=first.m,first.d,first.mj,first.p
TARGET='neighbor_5'


def configure(mode,source):
    record=json.loads((source/'result.json').read_text());summary=record['summary'];check=json.loads((source/'precheck.json').read_text())
    valid=summary.get('local_two_red_support_verified',False) if mode=='diagonal' else (
        summary.get('target')==TARGET and summary.get('held_duration_s',0)>=2.98 and summary.get('held_table_only',False)
        and summary.get('held_drift_mm',1e6)<=2 and summary.get('held_angular_drift_degrees',1e6)<=2
        and summary.get('final_face_tilt_degrees',1e6)<=2 and summary.get('minimum_table_support_n',0)>=.8*m.body(TARGET).mass*9.81
        and max(summary['other_bricks_max_shift_mm'].values())<=2)
    if not valid or summary['abort'] or summary['unintended_contacts'] or not check['passed'] or summary['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Source is not a verified released continuation')
    state=np.load(source/'integration-state.npz',allow_pickle=False)
    if str(state['mujoco_version'])!=mj.__version__ or len(state['state'])!=mj.mj_stateSize(m,int(state['spec'])):raise ValueError('Checkpoint runtime mismatch')
    def restore():
        mj.mj_setState(m,d,state['state'],int(state['spec']));mj.mj_forward(m,d)
    restore()
    if not np.allclose(d.qpos,record['rows'][-1]['qpos'],rtol=0,atol=1e-12) or not np.allclose(d.qvel,record['rows'][-1]['qvel'],rtol=0,atol=1e-12):raise ValueError('State and trace differ')
    return restore,check['stages'][-1]['result']['final_joints_sdk']


def prepare(mode,source,restore,face_shift_mm=0):
    restore();start=d.mocap_pos[p.t.LM].copy();startq=d.mocap_quat[p.t.LM].copy();target=d.xpos[m.body(TARGET).id].copy()
    corners=Rotation.from_quat(d.xquat[m.body(TARGET).id][[1,2,3,0]]).apply(p.CORNERS)+target
    goal=np.array([.299,.140]) if mode=='diagonal' else np.array([target[0],.140]) if mode=='south' else np.array([.227,target[1]])
    vector=target[:2]-goal;distance=float(np.linalg.norm(vector));normal=np.r_[vector/distance,0.];tangent=np.array([-normal[1],normal[0],0.])
    if not .001<distance<.150:raise ValueError('Feed distance outside this local trial')
    if face_shift_mm not in [0,-10,-15,-20,-30] or (mode!='diagonal' and face_shift_mm):raise ValueError('Unsupported contact-face offset')
    angle=float(np.degrees(np.arctan2(normal[1],normal[0])))
    quat=(Rotation.from_euler('z',angle,degrees=True)*Rotation.from_quat(p.t.QUAT[[1,2,3,0]])).as_quat()[[3,0,1,2]]
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=quat;mj.mj_forward(m,d)
    vertices=np.concatenate([first.b.a.placement.a.g.mesh_vertices('lh_'+n) for n in ['index_distal','index_proximal','middle_distal','middle_proximal']])
    root=normal*((corners@normal).max()-(vertices@normal).min()-.0005)
    root+=tangent*(target@tangent-((vertices@tangent).min()+(vertices@tangent).max())/2+face_shift_mm/1000);root[2]=p.t.ROOT[2]
    # The west contact endpoint clears blue 7, but the extra east-side 20 mm
    # approach intersects it. Descend vertically to that contact endpoint.
    clearance_shift=0.;approach_offset=0. if mode=='west' else .020
    if mode=='south':
        obstacle=d.xpos[m.body('neighbor_3').id].copy();oq=d.xquat[m.body('neighbor_3').id]
        oc=Rotation.from_quat(oq[[1,2,3,0]]).apply(p.CORNERS)+obstacle
        low=vertices[vertices[:,2]<=oc[:,2].max()+.002];approach_offset=0.
        if root[1]+low[:,1].max()>=oc[:,1].min() and root[1]+low[:,1].min()<=oc[:,1].max():
            clearance_shift=min(0.,float(oc[:,0].min()-.002-root[0]-low[:,0].max()))
            if clearance_shift<-.040:raise ValueError('No bounded west-side clearance for north obstacle')
            root[0]+=clearance_shift
    above=root+approach_offset*normal+[0,0,.070];turn=Slerp([0.,1.],Rotation.from_quat(np.array([startq,quat])[:,[1,2,3,0]]))
    def command(seconds):
        if seconds<2:pos=start+[0,0,.030*seconds/2];q=startq;phase='rise'
        elif seconds<6:
            f=(seconds-2)/4;pos=(start+[0,0,.030])*(1-f)+above*f;q=turn(f).as_quat()[[3,0,1,2]];phase='transfer_above'
        elif seconds<9:pos=root+approach_offset*normal+[0,0,.070*(9-seconds)/3];q=quat;phase='descend'
        elif seconds<10:pos=root+approach_offset*(10-seconds)*normal;q=quat;phase='approach'
        elif seconds<16:pos=root-distance*(seconds-10)/6*normal;q=quat;phase='feed_yellow'
        elif seconds<18:pos=root-distance*normal+[0,0,.050*(seconds-16)/2];q=quat;phase='lift_clear'
        elif seconds<20:pos=root+(-distance+.020*(seconds-18)/2)*normal+[0,0,.050];q=quat;phase='retreat'
        else:pos=root+(-distance+.020)*normal+[0,0,.050];q=quat;phase='released_hold'
        return pos,q,phase
    restore()
    return command,dict(source=str(source),target=TARGET,axis='feed-'+mode,initial_target_center_m=target.tolist(),goal_xy_m=goal.tolist(),
                        component=0 if mode=='west' else 1,destination_axis_m=float(goal[0] if mode=='west' else goal[1]),
                        contact_root_m=root.tolist(),normal=normal.tolist(),distance_m=distance,face_shift_mm=face_shift_mm,
                        obstacle_clearance_shift_x_m=clearance_shift,approach_offset_m=approach_offset,
                        duration_s=23.,hold_start_s=20.,whole_arm_collision_checked=False,native_verified=False,official_region_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['diagonal','south','west'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--face-shift-mm',type=int,choices=[0,-10,-15,-20,-30],default=0)
    args=parser.parse_args();restore,seed=configure(args.mode,args.source);command,plan=prepare(args.mode,args.source,restore,args.face_shift_mm);args.out.mkdir(parents=True,exist_ok=False)
    checked=first.precheck(command,restore,seed,target=TARGET)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),stages=len(checked['stages']),precheck_wall_seconds=checked['wall_seconds']);passed=checked['passed']
    if args.run and passed:
        summary=first.run(command,plan,restore,args.out);summary['local_feed_position_verified']=summary.pop('local_red_position_verified')
        summary['local_acquisition_verified']=False;summary['local_upper_placement_verified']=False
        record=json.loads((args.out/'result.json').read_text());record['summary']=summary;(args.out/'result.json').write_text(json.dumps(record,allow_nan=False))
        brief['dynamics']=summary;passed=summary['local_feed_position_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
