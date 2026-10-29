"""Replay the complete taught base with shorter command times, locally only.

The physics timestep, materials and actuators stay unchanged. Final released
observation remains three seconds. Requires retained rounds 77-89 assets.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import try_close_staged_pair as pair
import try_third_base_transfer as transfer

t,m,mj,d=pair.t,pair.m,pair.mj,pair.d
REDS=['neighbor_9','neighbor_6','neighbor_4','neighbor_2']
STATIONARY=['neighbor_0','neighbor_1','target']


def compile_plan():
    times=[];positions=[];quaternions=[];targets=[];segments=[]
    def append(start,duration,command,name,expected):
        local=np.r_[np.arange(0,duration,.01),duration]
        first=None;last=None
        for offset in local:
            seconds=start+float(offset);pos,quat,target=command(seconds)
            if first is None:first=(np.array(pos),np.array(quat))
            last=(np.array(pos),np.array(quat))
            if times and seconds<=times[-1]+1e-9:continue
            times.append(seconds);positions.append(pos);quaternions.append(quat);targets.append('none' if target is None else target)
        if segments:
            # Compare commands at the boundary, rather than resetting the hand.
            old=segments[-1]['last_position'];oldq=segments[-1]['last_quaternion']
            pe=float(np.linalg.norm(first[0]-old));re=float((Rotation.from_quat(first[1][[1,2,3,0]]).inv()*Rotation.from_quat(np.array(oldq)[[1,2,3,0]])).magnitude())
            if pe>1e-6 or re>1e-6:raise ValueError(f'Discontinuous source boundary: {name}: {pe}, {re}')
        segments.append(dict(name=name,start=start,end=start+duration,expected_source=str(expected),last_position=last[0].tolist(),last_quaternion=last[1].tolist()))
    previous=pair.previous
    def initial(seconds):
        pos,quat,_,target=previous.command(seconds);return pos,quat,target
    append(0,163.002,initial,'wide_staging_and_second_red','runs/wide-staging-85/segment-163-endpoint-fixed')
    for mode in ['transverse','align-first-side','close-final']:
        pair.configure(mode)
        append(pair.START_TIME,20,lambda seconds:(*pair.command(seconds)[:2],pair.TARGET),mode,pair.SOURCE)
    for target in ['neighbor_5','neighbor_4']:
        transfer.configure(target)
        append(transfer.START,58,lambda seconds:(*transfer.command(seconds)[:2],transfer.TARGET),target,transfer.SOURCE)
    pair.configure('align-third')
    append(pair.START_TIME,20,lambda seconds:(*pair.command(seconds)[:2],pair.TARGET),'align-third',pair.SOURCE)
    for target in ['neighbor_3','neighbor_2']:
        transfer.configure(target)
        append(transfer.START,58,lambda seconds:(*transfer.command(seconds)[:2],transfer.TARGET),target,transfer.SOURCE)
    pair.configure('align-fourth')
    append(pair.START_TIME,20,lambda seconds:(*pair.command(seconds)[:2],pair.TARGET),'align-fourth',pair.SOURCE)
    return np.array(times),np.array(positions),np.array(quaternions),np.array(targets),segments


def contact_windows():
    """Offline timing labels from the successful physical traces, not feedback."""
    folders=['runs/wide-staging-85/segment-163-endpoint-fixed','runs/pair-alignment-86/transverse',
             'runs/pair-alignment-87/first-red-side','runs/pair-alignment-87/final-close',
             'runs/third-base-88/yellow','runs/third-base-88/red','runs/third-base-88/aligned',
             'runs/fourth-base-89/yellow','runs/fourth-base-89/red','runs/fourth-base-89/aligned-low']
    intervals=[]
    for folder in folders:
        for row in json.loads((Path(folder)/'result.json').read_text())['rows']:
            touching=False
            for brick in row['bricks'].values():
                for contact in brick.get('contacts',[]):
                    names=contact['pair'] if isinstance(contact,dict) else contact
                    touching|=any(n.startswith(('lh_','rh_')) for n in names)
            if touching:
                lo,hi=row['t']-.25,row['t']+.25
                if intervals and lo<=intervals[-1][1]:intervals[-1][1]=max(intervals[-1][1],hi)
                else:intervals.append([lo,hi])
    return intervals


def prepare(speed,preserve_contact=False):
    ts,xyz,quat,targets,segments=compile_plan()
    slow=ts>=ts[-1]-3-1e-8;windows=contact_windows() if preserve_contact else []
    for lo,hi in windows:slow|=(ts>=lo)&(ts<=hi)
    schedule=np.r_[0,np.cumsum(np.diff(ts)/np.where(slow[:-1],1.,speed))]
    duration=float(schedule[-1])
    clock=np.arange(int(np.ceil(duration/m.opt.timestep))+1)*m.opt.timestep
    source=np.interp(clock,schedule,ts);pos=np.column_stack([np.interp(source,ts,xyz[:,i]) for i in range(3)])
    rot=Slerp(ts,Rotation.from_quat(quat[:,[1,2,3,0]]))(source).as_quat()[:,[3,0,1,2]]
    active=targets[np.maximum(0,np.searchsorted(ts,source,side='right')-1)]
    checks=dict(source_duration_s=float(ts[-1]),planned_duration_s=duration,speed_factor=speed,final_hold_s=3,
                preserve_contact=preserve_contact,contact_windows=windows,original_speed_duration_s=float(np.diff(ts)[slow[:-1]].sum()),
                retimed_segments=[dict(name=s['name'],end=float(np.interp(s['end'],ts,schedule))) for s in segments],
                source_samples=len(ts),segments=segments,physics_timestep_s=m.opt.timestep,
                maximum_command_translation_speed_m_s=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).max()/m.opt.timestep),
                native_joint_speed_validated=False,whole_arm_collision_checked=False)
    return clock,source,pos,rot,active,checks


def run(out,speed,until,resume,preserve_contact=False):
    clock,source,pos,quat,active,checks=prepare(speed,preserve_contact)
    (out/'plan.json').write_text(json.dumps(checks,allow_nan=False))
    t.initialize();d.mocap_pos[t.LM]=pos[0];d.mocap_quat[t.LM]=quat[0]
    fixed={n:d.xpos[m.body(n).id].copy() for n in STATIONARY}
    anchors=dict(fixed);last_target='none';rows=[];bad=[];abort=None;start_index=0
    if resume:
        old=json.loads((resume/'result.json').read_text());s=old['summary'];state=np.load(resume/'integration-state.npz')
        if json.loads((resume/'plan.json').read_text())!=checks:
            raise ValueError('Compiled control plan differs from the checkpoint plan')
        if s['abort'] or s['unexpected_hand_contacts'] or s['speed_factor']!=speed or s.get('preserve_contact',False)!=preserve_contact or s['checkpoint_next_step_maximum_difference']>1e-10:
            raise ValueError('Cannot resume unverified or incompatible result')
        if str(state['mujoco_version'])!=mj.__version__:raise ValueError('Runtime mismatch')
        mj.mj_setState(m,d,state['state'],int(state['spec']));start_index=round(float(d.time)/m.opt.timestep)
        anchors={n:np.array(p) for n,p in s['anchors'].items()};anchors.update(fixed);last_target=s['last_target']
    start_time=float(d.time);end=min(until,clock[-1]) if until else clock[-1]
    if end<=start_time:raise ValueError('Continuation endpoint must advance time')
    names=[m.body(i).name for i in range(m.nbody) if m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')]
    maximum_protected_shift={n:0. for n in REDS+STATIONARY}
    if resume:maximum_protected_shift={n:s['other_bricks_max_shift_mm'].get(n,0.)/1000 for n in REDS+STATIONARY}
    mj.mj_forward(m,d);started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for index in range(start_index,round(end/m.opt.timestep)):
            target=str(active[index])
            if target!=last_target:
                if last_target in REDS:
                    body=m.body(last_target).id;anchors[last_target]=d.xpos[body].copy()
                    corners=Rotation.from_quat(d.xquat[body][[1,2,3,0]]).apply(pair.CORNERS)+d.xpos[body]
                    if not (np.all(corners[:,:2]>=t.REGION[:,0]) and np.all(corners[:,:2]<=t.REGION[:,1])):
                        abort='placed_red_outside_analysis_rectangle'
                if target in anchors:del anchors[target]
                last_target=target
            d.mocap_pos[t.LM]=pos[index];d.mocap_quat[t.LM]=quat[index]
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,anchor in anchors.items():
                delta=d.xpos[m.body(n).id]-anchor
                # Unused supply bricks can settle vertically from the retained
                # initial pose; their horizontal placement must stay fixed.
                shift=float(np.linalg.norm(delta[:2] if n in STATIONARY else delta));maximum_protected_shift[n]=max(maximum_protected_shift[n],shift)
                if shift>.002:abort='protected_brick_moved'
            contacts={n:[] for n in names}
            for ci,c in enumerate(d.contact):
                ns=[m.geom(int(g)).name for g in c.geom]
                if any(n.startswith(('lh_','rh_')) for n in ns) and target not in ns:
                    bad.append(dict(t=float(d.time),pair=ns,distance_m=float(c.dist)))
                    if c.dist<-.0005:abort='unexpected_hand_contact'
                f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);vector=c.frame.reshape(3,3).T@f[:3]
                for n in contacts:
                    if n in ns:contacts[n].append(dict(pair=ns,force_xyz_n=(vector*(1 if ns[1]==n else -1)).tolist()))
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if index%10==9 or index==round(end/m.opt.timestep)-1 or abort:
                bricks={n:dict(position=d.xpos[m.body(n).id].tolist(),quaternion=d.xquat[m.body(n).id].tolist(),contacts=contacts[n]) for n in names}
                row=dict(t=float(d.time),source_t=float(source[index]),target=target,bricks=bricks,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if abort:break
    complete=not abort and d.time>=clock[-1]-1e-8
    summary=dict(speed_factor=speed,preserve_contact=preserve_contact,abort=abort,unexpected_hand_contacts=bad,unintended_hand_contacts=bad,
                 wall_seconds=time.monotonic()-started,start_time_s=start_time,end_time_s=float(d.time),complete_sequence=bool(complete),
                 other_bricks_max_shift_mm={n:v*1000 for n,v in maximum_protected_shift.items()},
                 anchors={n:p.tolist() for n,p in anchors.items()},last_target=last_target,
                 full_base_row_verified=False,native_verified=False,official_region_verified=False,full_pyramid_verified=False)
    if complete:
        pair.MODE='align-fourth';pair.START_TIME=clock[-1]-20
        pair.evaluate(summary,rows)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    restored=mj.MjData(m);mj.mj_setState(m,restored,state,spec)
    index=min(round(float(d.time)/m.opt.timestep),len(clock)-1)
    for data in [d,restored]:
        data.mocap_pos[t.LM]=pos[index];data.mocap_quat[t.LM]=quat[index];mj.mj_step(m,data);mj.mj_forward(m,data)
    a=np.empty_like(state);b=np.empty_like(state);mj.mj_getState(m,d,a,spec);mj.mj_getState(m,restored,b,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(a-b)));mj.mj_setState(m,d,state,spec)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));print(json.dumps(summary,indent=2))
    return not abort and not bad and summary['checkpoint_next_step_maximum_difference']<=1e-10 and (not complete or summary.get('local_four_brick_alignment_verified',False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--speed',type=float,choices=[1.,3.],default=3.);parser.add_argument('--until',type=float);parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--preserve-contact',action='store_true',help='Keep retained contact windows and 250ms margins at original speed')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.run:passed=run(args.out,args.speed,args.until,args.resume_from,args.preserve_contact)
    else:
        *_,checks=prepare(args.speed,args.preserve_contact);(args.out/'plan.json').write_text(json.dumps(checks,allow_nan=False));print(json.dumps(checks,indent=2));passed=True
    raise SystemExit(0 if passed else 1)
