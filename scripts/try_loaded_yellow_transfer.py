"""Continue a verified right-hand yellow hold toward the first red pair.

Preserve the complete integration state and finger controls. This tests loaded
transport above the pair; it does not release the yellow brick or claim a stack.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_yellow_edge_acquisition as a

m,mj,d,p=a.m,a.mj,a.d,a.p
SOURCE=Path('runs/upper-layer-93/loaded-hold')
RECORD=json.loads((SOURCE/'result.json').read_text())
STATE=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
CHECK=json.loads(Path('runs/upper-layer-93/cupped-receiver/precheck.json').read_text())
DURATION=16.
LEFT_CLEAR=np.array([.080,.080,.050])


def restore():
    s=RECORD['summary']
    if not s['local_loaded_hold_verified'] or s['abort'] or s['checkpoint_next_step_maximum_difference']>1e-10:
        raise ValueError('Loaded source did not pass its unchanged hold')
    if str(STATE['mujoco_version'])!=mj.__version__ or len(STATE['state'])!=mj.mj_stateSize(m,int(STATE['spec'])):
        raise ValueError('Checkpoint runtime mismatch')
    mj.mj_setState(m,d,STATE['state'],int(STATE['spec']));mj.mj_forward(m,d)


def prepare():
    restore();poses={side:(d.mocap_pos[index].copy(),d.mocap_quat[index].copy()) for side,index in [('left',p.t.LM),('right',p.t.RM)]}
    target=d.xpos[m.body('neighbor_1').id].copy()
    pair=(d.xpos[m.body('neighbor_9').id]+d.xpos[m.body('neighbor_6').id])/2
    delta=pair-target;delta[2]=.080
    def command(seconds):
        left=LEFT_CLEAR*np.clip(seconds/3,0,1)
        lift=np.array([0.,0.,.080*np.clip((seconds-3)/4,0,1)])
        horizontal=np.r_[delta[:2],0.]*np.clip((seconds-7)/6,0,1)
        right=lift+horizontal
        phase='left_clear' if seconds<3 else 'raise_payload' if seconds<7 else 'transfer_over_pair' if seconds<13 else 'hold'
        return {side:(poses[side][0]+offset,poses[side][1]) for side,offset in [('left',left),('right',right)]},right,phase
    return command,dict(source=str(SOURCE),initial_target_center_m=target.tolist(),red_pair_center_m=pair.tolist(),
                        requested_payload_translation_m=delta.tolist(),left_clear_m=LEFT_CLEAR.tolist(),
                        finger_controls_unchanged=True,upper_placement_verified=False,native_verified=False,
                        whole_arm_collision_checked=False,source_sequence_continuity_verified=False)


def precheck(command):
    started=time.monotonic();bad=[];restore();target_address=m.jnt_qposadr[m.joint('neighbor_1_free').id]
    for seconds in np.linspace(0,DURATION,321):
        restore();poses,offset,phase=command(seconds)
        for side,address in [('left',p.t.LA),('right',p.t.RA)]:
            pos,quat=poses[side];d.qpos[address:address+3]=pos;d.qpos[address+3:address+7]=quat
        d.qpos[target_address:target_address+3]+=offset;mj.mj_forward(m,d)
        for c in d.contact:
            ns=[m.geom(int(i)).name for i in c.geom]
            allowed='neighbor_1' in ns and any(n.startswith('rh_') for n in ns)
            relevant='neighbor_1' in ns or any(n.startswith(('rh_','lh_')) for n in ns)
            if relevant and not allowed:bad.append(dict(t=float(seconds),phase=phase,pair=ns,depth_mm=-float(c.dist)*1000))
    restore();stages={};initial={}
    for side,index,seed in [('left',p.t.LM,CHECK['left_stages'][-1]['result']['final_joints_sdk']),('right',p.t.RM,CHECK['right_lift']['final_joints_sdk'])]:
        ctx=p.t.routes.context()[side];ctx['root'][:3,3]=d.mocap_pos[index];ctx['root'][:3,:3]=Rotation.from_quat(d.mocap_quat[index][[1,2,3,0]]).as_matrix()
        ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020
        actual=np.asarray(p.t.routes.sdk.forward_kinematics(side,seed))@ctx['mount']
        initial[side]=dict(position_error_m=float(np.linalg.norm(actual[:3,3]-ctx['root'][:3,3])),
                           angle_error_rad=float(Rotation.from_matrix(ctx['root'][:3,:3].T@actual[:3,:3]).magnitude()))
        stages[side]=[];q=seed
        if not bad and initial[side]['position_error_m']<1e-5 and initial[side]['angle_error_rad']<1e-4:
            for seconds in ([3] if side=='left' else [7,13]):
                pos,_=command(seconds)[0][side]
                result=p.t.routes.stage(side,ctx,q,'loaded_transfer',(pos-ctx['root'][:3,3]).tolist());stages[side].append(result)
                if not result['continuous']:break
                q=result['final_joints_sdk']
    return dict(passed=not bad and len(stages['left'])==1 and len(stages['right'])==2 and all(x['continuous'] for ss in stages.values() for x in ss),
                geometry_failures=bad,stages=stages,initial_seed_errors=initial,wall_seconds=time.monotonic()-started,
                frozen_payload_geometry_only=True,whole_arm_collision_checked=False)


def run(command,plan,out):
    restore();start=float(d.time);body=m.body('neighbor_1').id;target=d.xpos[body].copy()
    others={m.body(i).name:d.xpos[i].copy() for i in range(m.nbody) if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and i!=body}
    shifts={n:0. for n in others};rows=[];bad=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(DURATION/m.opt.timestep)):
            poses,offset,phase=command(float(d.time)-start)
            for side,index in [('left',p.t.LM),('right',p.t.RM)]:d.mocap_pos[index],d.mocap_quat[index]=poses[side]
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,position in others.items():
                shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[m.body(n).id]-position)))
                if shifts[n]>.002:abort='other_brick_moved'
            contacts=[]
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(i)).name for i in c.geom]
                relevant='neighbor_1' in ns or any(n.startswith(('rh_','lh_')) for n in ns)
                allowed='neighbor_1' in ns and any(n.startswith('rh_') for n in ns)
                if relevant and not allowed and c.dist<-.0005:
                    bad.append(dict(t=float(d.time)-start,pair=ns,depth_mm=-float(c.dist)*1000));abort='unexpected_contact'
                if 'neighbor_1' in ns:
                    f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);v=c.frame.reshape(3,3).T@f[:3]*(1 if ns[1]=='neighbor_1' else -1)
                    contacts.append(dict(pair=ns,force_xyz_n=v.tolist()))
            rot=Rotation.from_quat(d.xquat[body][[1,2,3,0]]);bottom=float((rot.apply(p.CORNERS)+d.xpos[body])[:,2].min()-a.g.FLOOR)
            error=float(np.linalg.norm(d.xpos[body]-target-offset))
            if error>.020:abort='payload_slipped'
            if bottom<(.055 if phase in ['transfer_over_pair','hold'] else .005):abort='payload_clearance_lost'
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                row=dict(elapsed_s=float(d.time)-start,phase=phase,position_m=d.xpos[body].tolist(),quaternion_wxyz=d.xquat[body].tolist(),
                         lowest_corner_m=bottom,payload_translation_error_m=error,contacts=contacts,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if abort:break
    held=[r for r in rows if r['elapsed_s']>=13-1e-8]
    summary=dict(abort=abort,duration_s=float(d.time)-start,wall_seconds=time.monotonic()-started,unintended_contacts=bad,
                 other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},local_loaded_transfer_verified=False,
                 upper_placement_verified=False,native_verified=False,whole_arm_collision_checked=False,source_sequence_continuity_verified=False)
    if held:
        xyz=np.array([r['position_m'] for r in held]);rot=Rotation.from_quat(np.array([r['quaternion_wxyz'] for r in held])[:,[1,2,3,0]])
        drift=float(np.linalg.norm(xyz-xyz[0],axis=1).max());angle=float(np.degrees((rot[0].inv()*rot).magnitude().max()))
        right_only=all(r['contacts'] and all(any(n.startswith('rh_') for n in c['pair']) for c in r['contacts']) for r in held)
        up=min(sum(c['force_xyz_n'][2] for c in r['contacts']) for r in held)
        summary.update(held_duration_s=held[-1]['elapsed_s']-held[0]['elapsed_s'],held_drift_mm=drift*1000,held_angular_drift_degrees=angle,
                       held_minimum_bottom_mm=min(r['lowest_corner_m'] for r in held)*1000,held_minimum_up_force_n=up,
                       held_only_right_hand_contacts=right_only,final_translation_error_mm=held[-1]['payload_translation_error_m']*1000)
        summary['local_loaded_transfer_verified']=bool(not abort and not bad and summary['held_duration_s']>=2.98 and drift<=.002
                         and angle<=2 and right_only and up>=.8*m.body('neighbor_1').mass*9.81 and summary['final_translation_error_mm']<=5)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    restored=mj.MjData(m);mj.mj_setState(m,restored,state,spec);x=np.empty_like(state);y=np.empty_like(state)
    mj.mj_step(m,d);mj.mj_step(m,restored);mj.mj_getState(m,d,x,spec);mj.mj_getState(m,restored,y,spec);mj.mj_setState(m,d,state,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(x-y)))
    summary['local_loaded_transfer_verified'] &= summary['checkpoint_next_step_maximum_difference']<=1e-10
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);command,plan=prepare();checked=precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(precheck_passed=checked['passed'],geometry_failure_count=len(checked['geometry_failures']),stage_counts={s:len(v) for s,v in checked['stages'].items()},precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if args.run and passed:brief['dynamics']=run(command,plan,args.out);passed=brief['dynamics']['local_loaded_transfer_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
