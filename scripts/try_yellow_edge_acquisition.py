"""Isolated local yellow pickup through the measured left table edge.

Uses the actual settled yellow-west checkpoint, but replaces the artificial
right parking pose with SDK joint zero at initialization. This is NOT a
continuous replay of the preceding base sequence or a native controller.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import brentq
import try_taught_upper_tip as previous

g,p,m,mj,d=previous.g,previous.p,previous.m,previous.mj,previous.d
SOURCE=Path('runs/upper-layer-92/yellow-west')
STATE=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
RECORD=json.loads((SOURCE/'result.json').read_text())
CHECK=json.loads((SOURCE/'precheck.json').read_text())
CTX=p.t.routes.context()
# FK-verified candidate: retain the old loaded support reference while allowing
# 11.195 degrees of wrist orientation change. The bounded optimizer reached its
# evaluation limit; no claim of optimality or solver convergence is made.
RIGHT_Q=np.array([-.026751461001499895,.023246579787497418,-1.5027346440854896,
                  -.7458503727350062,1.8399200387960728,-.23016447860546196,.4789509733913404])
OUTSIDE=RIGHT_Q.copy();OUTSIDE[0]=-.5
WAYPOINTS=[np.zeros(7),np.array([-.5,0,0,0,0,0,0]),OUTSIDE,RIGHT_Q]
DURATION=28.
FACE_MCP=None
FACE_REPORT=None
FACE_TARGETS={}
FACE_FINGERS=['index','middle']
CUP_BEFORE_LIFT=False


def forward(q):return np.asarray(p.t.routes.sdk.forward_kinematics('right',q.tolist()))@CTX['right']['mount']


def finger_commands(seconds):
    if not CUP_BEFORE_LIFT:return {}
    value=float(.3*np.clip(seconds-22,0,1))
    return {'rh_'+f+suffix:value for f in ['index','middle','ring','pinky'] for suffix in ['_mcp_pitch','_dip']}


def restore(geometry=False):
    s=RECORD['summary']
    if (s['abort'] or not CHECK['passed'] or not s['released_table_only'] or s['released_duration_s']<2.98
            or s['released_drift_mm']>2 or max(s['other_bricks_max_shift_mm'].values())>2
            or s['minimum_table_support_n']<.8*m.body('neighbor_1').mass*9.81
            or s['checkpoint_next_step_maximum_difference']>1e-10):raise ValueError('Unsafe partial source')
    if str(STATE['mujoco_version'])!=mj.__version__ or len(STATE['state'])!=mj.mj_stateSize(m,int(STATE['spec'])):
        raise ValueError('Checkpoint runtime mismatch')
    mj.mj_setState(m,d,STATE['state'],int(STATE['spec']))
    h=forward(np.zeros(7));quat=Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]]
    d.qpos[p.t.RA:p.t.RA+3]=h[:3,3];d.qpos[p.t.RA+3:p.t.RA+7]=quat
    address=m.jnt_dofadr[m.joint('right_hand_root_free').id];d.qvel[address:address+6]=0
    d.mocap_pos[p.t.RM]=h[:3,3];d.mocap_quat[p.t.RM]=quat
    for i in range(m.njnt):
        name=m.joint(i).name
        if name.startswith('rh_'):
            d.qpos[m.jnt_qposadr[i]]=0;d.qvel[m.jnt_dofadr[i]]=0;d.ctrl[m.actuator(name).id]=0
    if FACE_MCP is not None:
        name='lh_middle_mcp_pitch';d.ctrl[m.actuator(name).id]=FACE_MCP
        if geometry:d.qpos[m.jnt_qposadr[m.joint(name).id]]=FACE_MCP
    for name,value in FACE_TARGETS.items():
        d.ctrl[m.actuator(name).id]=value
        if geometry:d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    d.qacc_warmstart[:]=0
    mj.mj_forward(m,d)


def level_finger_face():
    """Align distal leading extents offline; change no material or root gain."""
    global FACE_MCP,FACE_REPORT
    restore();d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=p.t.QUAT
    mj.mj_forward(m,d);address=m.jnt_qposadr[m.joint('lh_middle_mcp_pitch').id]
    index=float(g.mesh_vertices('lh_index_distal')[:,0].min())
    def error(angle):
        d.qpos[address]=angle;mj.mj_forward(m,d)
        return float(g.mesh_vertices('lh_middle_distal')[:,0].min()-index)
    initial=error(0.)
    FACE_MCP=float(brentq(error,0.,.08,xtol=1e-12))
    FACE_REPORT=dict(middle_mcp_target_rad=FACE_MCP,initial_leading_extent_difference_mm=initial*1000,
                     solved_leading_extent_difference_mm=error(FACE_MCP)*1000,
                     scope='Static distal support extents; not proof of two loaded contacts or yaw stability')


def four_finger_face():
    """Widen the pusher using existing fingers, with no added collision shapes."""
    global FACE_REPORT,FACE_TARGETS,FACE_FINGERS
    restore();FACE_FINGERS=['index','middle','ring','pinky']
    FACE_TARGETS={'lh_'+finger+suffix:0. for finger in FACE_FINGERS for suffix in ['_mcp_pitch','_dip']}
    for name,value in FACE_TARGETS.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=p.t.QUAT;mj.mj_forward(m,d)
    reference=float(g.mesh_vertices('lh_index_distal')[:,0].min());errors={}
    for finger in ['middle','ring','pinky']:
        name='lh_'+finger+'_mcp_pitch';address=m.jnt_qposadr[m.joint(name).id]
        def error(angle):
            d.qpos[address]=angle;mj.mj_forward(m,d)
            return float(g.mesh_vertices('lh_'+finger+'_distal')[:,0].min()-reference)
        goal=0. if abs(error(0.))<1e-8 else float(brentq(error,0.,.08,xtol=1e-12))
        FACE_TARGETS[name]=goal;errors[finger]=error(goal)*1000
    vertices=np.concatenate([g.mesh_vertices('lh_'+f+'_distal') for f in FACE_FINGERS])
    FACE_REPORT=dict(mode='four_finger_face',targets=FACE_TARGETS.copy(),leading_extent_errors_mm=errors,
                     distal_transverse_span_mm=float(np.ptp(vertices[:,1])*1000),
                     scope='Actual existing finger meshes; static alignment does not prove stable pushing')


def prepare():
    restore(geometry=True);start=d.mocap_pos[p.t.LM].copy();startq=d.mocap_quat[p.t.LM].copy()
    target=d.xpos[m.body('neighbor_1').id].copy();tq=d.xquat[m.body('neighbor_1').id].copy()
    pts=Rotation.from_quat(tq[[1,2,3,0]]).apply(p.CORNERS)+target
    d.qpos[p.t.LA:p.t.LA+3]=[0,0,p.t.ROOT[2]];d.qpos[p.t.LA+3:p.t.LA+7]=p.t.QUAT;mj.mj_forward(m,d)
    vs=np.concatenate([g.mesh_vertices('lh_'+finger+suffix) for finger in FACE_FINGERS for suffix in ['_distal','_proximal']])
    root=np.array([pts[:,0].max()-vs[:,0].min()-.0005,target[1]-(vs[:,1].min()+vs[:,1].max())/2,p.t.ROOT[2]])
    turn=Slerp([9.,12.],Rotation.from_quat(np.array([startq,p.t.QUAT])[:,[1,2,3,0]]))
    catch=forward(RIGHT_Q);rq=Rotation.from_matrix(catch[:3,:3]).as_quat()[[3,0,1,2]]
    support_local=np.array([.0349216744800285,.016086104322103473,.15425671734537963])
    support_goal=np.array([.11088333728649594,target[1]-.00311479731495617,-.5353070401395216])
    support_error=float(np.linalg.norm(catch[:3,:3]@support_local+catch[:3,3]-support_goal))
    if support_error>1e-5:raise ValueError('Recorded candidate no longer matches this support reference')
    def command(seconds):
        if seconds<9:
            index=min(int(seconds/3),2);f=(seconds-index*3)/3
            h=forward(WAYPOINTS[index]*(1-f)+WAYPOINTS[index+1]*f)
            right=h[:3,3];rightq=Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]]
        else:
            height=.020*np.clip((seconds-23)/2,0,1) if CUP_BEFORE_LIFT else .020*np.clip((seconds-22)/2,0,1)+.010*np.clip(seconds-24,0,1)
            right=catch[:3,3]+[0,0,height];rightq=rq
        if seconds<9:left=start;phase='right_outside_entry'
        elif seconds<12:
            f=(seconds-9)/3;left=start*(1-f)+(root+[.020,0,.070])*f;phase='left_transit'
        elif seconds<15:left=root+[.020,0,.070*(15-seconds)/3];phase='left_descend'
        elif seconds<16:left=root+[.020*(16-seconds),0,0];phase='left_approach'
        elif seconds<22:left=root+[-.120*(seconds-16)/6,0,0];phase='edge_push'
        elif CUP_BEFORE_LIFT and seconds<23:left=root+[-.120,0,0];phase='curl_receiving_fingers'
        elif CUP_BEFORE_LIFT and seconds<25:left=root+[-.120,0,.020*(seconds-23)/2];phase='common_lift'
        elif seconds<24:left=root+[-.120,0,.020*(seconds-22)/2];phase='common_lift'
        else:left=root+[-.120,0,.020];phase='right_clearance' if seconds<25 else 'hold'
        leftq=turn(np.clip(seconds,9,12)).as_quat()[[3,0,1,2]]
        return dict(left=(left,leftq),right=(right,rightq)),phase
    return command,dict(source=str(SOURCE),initial_target_center_m=target.tolist(),left_contact_root_m=root.tolist(),
                        right_joint_waypoints=[q.tolist() for q in WAYPOINTS],
                        support_reference_error_m=support_error,candidate_optimizer_converged=False,
                        finger_face_adjustment=FACE_REPORT,
                        cup_before_lift=CUP_BEFORE_LIFT,right_curl_schedule='22-23 seconds: MCP/DIP 0 to 0.3 rad; clock only' if CUP_BEFORE_LIFT else None,
                        initial_right_override='SDK joint zero; not the preceding artificial high parking pose',
                        source_sequence_continuity_verified=False,native_verified=False,whole_arm_collision_checked=False)


def precheck(command):
    started=time.monotonic();bad=[]
    for seconds in np.linspace(0,DURATION,561):
        restore(geometry=True);poses,phase=command(seconds)
        for side,address in [('left',p.t.LA),('right',p.t.RA)]:
            pos,quat=poses[side];d.qpos[address:address+3]=pos;d.qpos[address+3:address+7]=quat
        for name,value in finger_commands(seconds).items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in ns) and 'neighbor_1' not in ns:
                bad.append(dict(t=float(seconds),phase=phase,pair=ns,depth_mm=-float(c.dist)*1000))
    stages=[];ctx=CTX['left'];ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020
    q=CHECK['stages'][-1]['result']['final_joints_sdk']
    if not bad:
        for seconds in np.linspace(9,28,77):
            (pos,quat)=command(seconds)[0]['left'];ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            result=p.t.routes.stage('left',ctx,q,'left_acquisition',(pos-ctx['root'][:3,3]).tolist())
            stages.append(dict(t=float(seconds),result=result))
            if not result['continuous']:break
            q=result['final_joints_sdk']
    rctx=CTX['right'];rctx['limits']=rctx['intersection']+[.020,-.020];rctx['inset']=.020;rctx['root']=forward(RIGHT_Q)
    lift=p.t.routes.stage('right',rctx,RIGHT_Q.tolist(),'right_lift',[0,0,.020 if CUP_BEFORE_LIFT else .030])
    limits_ok=bool(np.all(np.array(WAYPOINTS)>=rctx['limits'][:,0]) and np.all(np.array(WAYPOINTS)<=rctx['limits'][:,1]))
    return dict(passed=not bad and len(stages)==77 and all(s['result']['continuous'] for s in stages) and lift['continuous'] and limits_ok,
                geometry_failures=bad,left_stages=stages,right_lift=lift,right_joint_entry_within_limits=limits_ok,
                wall_seconds=time.monotonic()-started,whole_arm_collision_checked=False)


def run(command,out):
    restore();start=float(d.time);names=[m.body(i).name for i in range(m.nbody) if m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')]
    initial={n:d.xpos[m.body(n).id].copy() for n in names};shifts={n:0. for n in names if n!='neighbor_1'}
    rows=[];bad=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            poses,phase=command(float(d.time)-start)
            for side,index in [('left',p.t.LM),('right',p.t.RM)]:d.mocap_pos[index],d.mocap_quat[index]=poses[side]
            for name,value in finger_commands(float(d.time)-start).items():d.ctrl[m.actuator(name).id]=value
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n in shifts:
                shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-initial[n])))
                if shifts[n]>.002:abort='other_brick_moved'
            contacts=[]
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(i)).name for i in c.geom]
                if any(n.startswith(('lh_','rh_')) for n in ns) and 'neighbor_1' not in ns and c.dist<-.0005:
                    bad.append(dict(t=float(d.time)-start,pair=ns,depth_mm=-float(c.dist)*1000));abort='unexpected_hand_contact'
                if 'neighbor_1' in ns:
                    force=np.zeros(6);mj.mj_contactForce(m,d,ci,force)
                    vector=c.frame.reshape(3,3).T@force[:3]*(1 if ns[1]=='neighbor_1' else -1)
                    contacts.append(dict(pair=ns,force_xyz_n=vector.tolist()))
            body=m.body('neighbor_1').id;rot=Rotation.from_quat(d.xquat[body][[1,2,3,0]])
            bottom=float((rot.apply(p.CORNERS)+d.xpos[body])[:,2].min()-g.FLOOR)
            if bottom<-.020:abort='target_corner_below_limit'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row=dict(elapsed_s=float(d.time)-start,phase=phase,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),
                         lowest_corner_m=bottom,contacts=contacts,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort:break
    held=[r for r in rows if r['elapsed_s']>=25-1e-8]
    summary=dict(abort=abort,duration_s=float(d.time)-start,wall_seconds=time.monotonic()-started,
                 unintended_hand_contacts=bad,other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},
                 local_acquisition_verified=False,upper_placement_verified=False,native_verified=False,
                 source_sequence_continuity_verified=False,whole_arm_collision_checked=False)
    if held:
        xyz=np.array([r['position_m'] for r in held]);drift=float(np.linalg.norm(xyz-xyz[0],axis=1).max())
        only_hands=all(r['contacts'] and all(any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in r['contacts']) for r in held)
        up=min(sum(c['force_xyz_n'][2] for c in r['contacts'] if any(n.startswith(('lh_','rh_')) for n in c['pair'])) for r in held)
        summary.update(held_duration_s=held[-1]['elapsed_s']-held[0]['elapsed_s'],held_minimum_bottom_mm=min(r['lowest_corner_m'] for r in held)*1000,
                       held_drift_mm=drift*1000,held_only_hand_contacts=only_hands,held_minimum_hand_up_force_n=up)
        summary['local_acquisition_verified']=bool(not abort and not bad and summary['held_duration_s']>=2.98
                        and summary['held_minimum_bottom_mm']>=5 and drift<=.002 and only_hands and up>=.8*m.body('neighbor_1').mass*9.81)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    restored=mj.MjData(m);mj.mj_setState(m,restored,state,spec)
    first=np.empty_like(state);second=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,restored);mj.mj_getState(m,d,first,spec);mj.mj_getState(m,restored,second,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(first-second)))
    summary['local_acquisition_verified'] &= summary['checkpoint_next_step_maximum_difference']<=1e-10
    mj.mj_setState(m,d,state,spec)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--level-finger-face',action='store_true',help='Align index/middle leading extents using a small middle-MCP change')
    group.add_argument('--four-finger-face',action='store_true',help='Open four fingers and align their leading extents')
    parser.add_argument('--cup-before-lift',action='store_true',help='With the four-finger pusher, curl receiving fingers before a common 20 mm lift')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.level_finger_face:level_finger_face()
    if args.four_finger_face:four_finger_face()
    if args.cup_before_lift:
        if not args.four_finger_face:raise ValueError('This comparison requires --four-finger-face')
        CUP_BEFORE_LIFT=True
    command,plan=prepare();checked=precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),left_stage_count=len(checked['left_stages']),precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if args.run and passed:brief['dynamics']=run(command,args.out);passed=brief['dynamics']['local_acquisition_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
