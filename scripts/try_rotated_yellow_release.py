"""Local release comparison: flatten the carried brick before opening the hand.

Offline geometry chooses a wrist rotation around the measured payload center and
a collision-free lowering distance. Execution is fixed; no online object truth.
Uses the actual loaded-transfer checkpoint. An optional left guide enters after
flattening; its angle is measured from the east side of the brick.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import try_guided_yellow_placement as g


def prepare(align_yaw=False,east_guide=False,guide_angle=0.,guide_height=.075):
    if not np.isfinite([guide_angle,guide_height]).all() or not .050<=guide_height<=.150:
        raise ValueError('Guide height must be 50–150 mm and angle must be finite')
    g.restore();m,d=g.m,g.d
    starts={side:(d.mocap_pos[i].copy(),d.mocap_quat[i].copy()) for side,i in [('left',g.p.t.LM),('right',g.p.t.RM)]}
    controls={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)}
    center=d.xpos[m.body(g.TARGET).id].copy()
    cube=Rotation.from_quat(d.xquat[m.body(g.TARGET).id][[1,2,3,0]])
    wrist=Rotation.from_quat(starts['right'][1][[1,2,3,0]])
    if align_yaw:
        candidates=Rotation.create_group('O')
        angles=(candidates*cube.inv()).magnitude()
        delta=candidates[int(np.argmin(angles))]*cube.inv()
    else:
        matrix=cube.as_matrix();normal=matrix[:,int(np.argmax(np.abs(matrix[2,:])))].copy()
        if normal[2]<0:normal=-normal
        axis=np.cross(normal,[0,0,1]);angle=float(np.arccos(np.clip(normal[2],-1,1)))
        delta=Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle) if angle>1e-10 else Rotation.identity()
    rotated=delta.apply(starts['right'][0]-center)+center
    final_quat=(delta*wrist).as_quat()[[3,0,1,2]]
    away=rotated-center;away[2]=0;away=away/np.linalg.norm(away)*.100
    # Search only the hand/base geometry; never move a brick during execution.
    desired=max(0.,center[2]-g.a.g.FLOOR-.025-.060)
    candidates=np.linspace(0,desired,int(np.ceil(desired/.001))+1)
    valid=[];blocked=None
    for down in candidates:
        g.restore();d.qpos[g.p.t.RA:g.p.t.RA+3]=rotated-[0,0,down]
        d.qpos[g.p.t.RA+3:g.p.t.RA+7]=final_quat;g.mj.mj_forward(m,d)
        bad=[]
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and g.TARGET not in ns:bad.append(ns)
        if bad:blocked=dict(lowering_m=float(down),pairs=bad);break
        valid.append(float(down))
    if not valid:raise ValueError('Rotated wrist already intersects non-target geometry')
    lowering=max(0.,valid[-1]-(.005 if blocked else 0.))
    guide_root=None;guide_targets=g.p.t.CANDIDATE['hand_targets'].copy();guide_quat=starts['left'][1]
    if east_guide:
        g.restore()
        for name,value in guide_targets.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        guide_quat=(Rotation.from_euler('z',guide_angle,degrees=True)*Rotation.from_quat(starts['left'][1][[1,2,3,0]])).as_quat()[[3,0,1,2]]
        guide_turn=Slerp([10,13],Rotation.from_quat(np.array([starts['left'][1],guide_quat])[:,[1,2,3,0]]))
        d.qpos[g.p.t.LA:g.p.t.LA+3]=0;d.qpos[g.p.t.LA+3:g.p.t.LA+7]=guide_quat;g.mj.mj_forward(m,d)
        vs=np.concatenate([g.a.g.mesh_vertices('lh_'+n) for n in ['index_proximal','index_distal','middle_proximal','middle_distal']])
        corners=(delta*cube).apply(g.p.CORNERS)+center-[0,0,lowering]
        normal=np.array([np.cos(np.radians(guide_angle)),np.sin(np.radians(guide_angle)),0]);tangent=np.array([-normal[1],normal[0],0])
        guide_root=normal*(np.max(corners@normal)+.002-np.min(vs@normal))+tangent*(center@tangent-(np.min(vs@tangent)+np.max(vs@tangent))/2)
        guide_root[2]=g.a.g.FLOOR+guide_height-vs[:,2].min()
    g.DURATION=27. if east_guide else 16.
    def command(t):
        elapsed=t
        if east_guide:t=t if t<8 else 8. if t<16 else t-8
        turn=Rotation.from_rotvec(delta.as_rotvec()*np.clip(t/4,0,1))
        right=turn.apply(starts['right'][0]-center)+center
        right=right-np.array([0,0,lowering])*np.clip((t-4)/4,0,1)+away*np.clip((t-10)/3,0,1)
        quat=(turn*wrist).as_quat()[[3,0,1,2]]
        ctrl=controls.copy()
        for finger in ['index','middle','ring','pinky']:
            for suffix in ['_mcp_pitch','_dip']:
                name='rh_'+finger+suffix;ctrl[name]=controls[name]*(1-np.clip((t-8)/2,0,1))
        phase='rotate_loaded_wrist' if t<4 else 'lower_flat_candidate' if t<8 else 'open_stationary_hand' if t<10 else 'withdraw_empty_candidate' if t<13 else 'released_hold'
        left=starts['left']
        if east_guide:
            high=guide_root+[0,0,.100]
            if elapsed<8:position=starts['left'][0]
            elif elapsed<10:position=starts['left'][0];phase='fold_guide_fingers'
            elif elapsed<13:
                f=(elapsed-10)/3;position=starts['left'][0]*(1-f)+high*f;phase='guide_high_transit'
            elif elapsed<16:
                f=(elapsed-13)/3;position=high*(1-f)+guide_root*f;phase='guide_descend'
            elif elapsed<21:position=guide_root
            else:position=guide_root+(.05*normal+[0,0,.05])*np.clip((elapsed-21)/3,0,1);phase='guide_withdraw' if elapsed<24 else 'released_hold'
            left=(position,guide_turn(np.clip(elapsed,10,13)).as_quat()[[3,0,1,2]])
            for name,value in guide_targets.items():ctrl[name]=controls[name]+(value-controls[name])*np.clip((elapsed-8)/2,0,1)
        return dict(left=left,right=(right,quat)),ctrl,phase
    g.restore()
    pair=np.mean([d.xpos[m.body(n).id] for n in g.SUPPORTS],axis=0)
    plan=dict(source=str(g.SOURCE),target=g.TARGET,supports=g.SUPPORTS,pair_center_m=pair.tolist(),initial_target_center_m=center.tolist(),
              align_yaw=align_yaw,rotation_vector_rad=delta.as_rotvec().tolist(),rotation_degrees=float(np.degrees(delta.magnitude())),
              requested_lowering_m=desired,lowering_m=lowering,lowering_first_collision=blocked,withdrawal_m=away.tolist(),
              duration_s=g.DURATION,hold_start_s=g.DURATION-3,left_hand_unchanged=not east_guide,east_guide=east_guide,
              guide_root_m=guide_root.tolist() if east_guide else None,guide_angle_degrees=guide_angle,guide_min_height_m=guide_height,online_object_feedback=False,
              native_verified=False,whole_arm_collision_checked=False,source_sequence_continuity_verified=g.CONTINUOUS_SOURCE)
    return command,plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/upper-place-100/loaded-transfer'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--align-yaw',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--east-guide',action='store_true',help='Enable a guide after flattening; --guide-angle rotates it from the default east side')
    parser.add_argument('--guide-angle',type=float,default=0.);parser.add_argument('--guide-height-mm',type=float,default=75.)
    args=parser.parse_args();g.configure_continuation(args.source,'neighbor_5')
    command,plan=prepare(args.align_yaw,args.east_guide,args.guide_angle,args.guide_height_mm/1000);args.out.mkdir(parents=True,exist_ok=False)
    checked=g.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(rotation_degrees=plan['rotation_degrees'],lowering_m=plan['lowering_m'],precheck_passed=checked['passed'],
               geometry_failures=len(checked['geometry_failures']),stage_counts={s:len(v) for s,v in checked['stages'].items()},precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if passed and args.run:brief['dynamics']=g.run(command,plan,args.out);passed=brief['dynamics']['local_first_yellow_aligned']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
