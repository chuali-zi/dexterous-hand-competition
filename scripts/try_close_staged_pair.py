"""Continue the saved local staging state with one short alignment push.

Fixed commands use the previously recorded pose as offline teaching. No ROS,
platform calls, scene reset, material change, or live object-state feedback.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import try_spaced_base_transfer as previous

t=previous.t;m,mj,d=previous.m,previous.mj,previous.d
SOURCE=Path('runs/wide-staging-85/segment-163-endpoint-fixed')
RECORD=json.loads((SOURCE/'result.json').read_text())
CHECK=json.loads((SOURCE/'precheck.json').read_text())
STATE=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
CORNERS=previous.w.CORNERS
MODE='transverse'
TARGET='neighbor_6'


def restore():
    summary=RECORD['summary']
    usable=summary.get('two_red_staging_verified',False)
    if MODE.startswith('align-first') or MODE=='close-final':
        expected='align-first-side' if MODE=='close-final' else 'transverse'
        protected='neighbor_6' if MODE=='close-final' else 'neighbor_9'
        usable=(summary.get('mode')==expected and summary.get('abort') is None
                and summary.get('held_duration_s',0)>=2.98 and not summary.get('unintended_hand_contacts',[])
                and summary.get('other_bricks_max_shift_mm',{}).get(protected,float('inf'))<=2)
    if MODE=='align-third':
        usable=(summary.get('target')=='neighbor_4' and summary.get('target_staging_verified',False)
                and summary.get('placed_pair_preserved',False) and summary.get('abort') is None)
    if not usable or summary.get('checkpoint_next_step_maximum_difference',float('inf'))>1e-10:
        raise ValueError('Source staging or continuation equivalence is unverified')
    if str(STATE['mujoco_version'])!=mj.__version__ or len(STATE['state'])!=mj.mj_stateSize(m,int(STATE['spec'])):
        raise ValueError('Checkpoint runtime or dimensions differ')
    mj.mj_setState(m,d,STATE['state'],int(STATE['spec']))


restore()
START_TIME=float(d.time)
START_POS=d.mocap_pos[t.LM].copy()
START_ROT=Rotation.from_quat(d.mocap_quat[t.LM][[1,2,3,0]])


def parameters():
    if MODE=='align-third':return 90.,.045
    if MODE=='close-final':return 0.,.010
    if MODE.startswith('align-first'):
        if MODE=='align-first-side':return 0.,.022
        return (90. if MODE=='align-first-front' else -90.)+np.rad2deg(np.arctan(.08)),.022
    return (90.,.035) if MODE=='transverse' else (60.,.040)


def contact_root():
    angle,_=parameters();rotation=Rotation.from_euler('z',angle,degrees=True)
    # The continued transverse checkpoint already has a 90-degree world yaw.
    world_angle=angle+(90. if MODE.startswith('align-first') else 0.)+(180. if MODE=='align-first' else 0.)
    if MODE=='align-first-side':world_angle=180.+np.rad2deg(np.arctan(.08))
    if MODE=='close-final':world_angle=90.
    world_rotation=Rotation.from_euler('z',world_angle,degrees=True)
    normal=world_rotation.apply([1,0,0]);tangent=world_rotation.apply([0,1,0])
    t.initialize();d.qpos[t.LA:t.LA+3]=[0,0,t.ROOT[2]]
    d.qpos[t.LA+3:t.LA+7]=(rotation*START_ROT).as_quat()[[3,0,1,2]];mj.mj_forward(m,d)
    vertices=[]
    for name in ['lh_index_proximal','lh_index_distal','lh_middle_proximal','lh_middle_distal']:
        g=m.geom(name).id;mesh=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
        vertices.extend(m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
    vertices=np.array(vertices);source=RECORD['rows'][-1]['bricks'][TARGET];center=np.array(source['position'])
    pts=Rotation.from_quat(np.array(source['quaternion'])[[1,2,3,0]]).apply(CORNERS)+center
    rn=float((pts@normal).max()-(vertices@normal).min()-.0005)
    ru=float(center@tangent-((vertices@tangent).min()+(vertices@tangent).max())/2)
    # Move the two-finger face away from the parked blue brick for the diagonal
    # candidate; this is a fixed offline offset, not feedback during execution.
    if MODE=='diagonal-clear':ru+=.008
    root=rn*normal+ru*tangent;root[2]=t.ROOT[2]
    return root,normal


ROOT,NORMAL=contact_root()


def configure(mode):
    """Select a retained, checked source and its fixed offline teaching action."""
    global MODE,SOURCE,TARGET,RECORD,CHECK,STATE,START_TIME,START_POS,START_ROT,ROOT,NORMAL
    MODE=mode
    if mode=='align-third':SOURCE=Path('runs/third-base-88/red');TARGET='neighbor_4'
    elif mode=='close-final':SOURCE=Path('runs/pair-alignment-87/first-red-side');TARGET='neighbor_6'
    elif mode.startswith('align-first'):SOURCE=Path('runs/pair-alignment-86/transverse');TARGET='neighbor_9'
    else:SOURCE=Path('runs/wide-staging-85/segment-163-endpoint-fixed');TARGET='neighbor_6'
    RECORD=json.loads((SOURCE/'result.json').read_text());CHECK=json.loads((SOURCE/'precheck.json').read_text())
    STATE=np.load(SOURCE/'integration-state.npz',allow_pickle=False);restore()
    START_TIME=float(d.time);START_POS=d.mocap_pos[t.LM].copy()
    START_ROT=Rotation.from_quat(d.mocap_quat[t.LM][[1,2,3,0]])
    ROOT,NORMAL=contact_root()


def command(seconds):
    elapsed=max(0.,min(seconds-START_TIME,20.));angle,distance=parameters()
    if elapsed<3:pos=START_POS+[0,0,.050*elapsed/3];yaw=0.;phase='rise'
    elif elapsed<7:
        f=(elapsed-3)/4;pos=(START_POS+[0,0,.050])*(1-f)+(ROOT+.020*NORMAL+[0,0,.100])*f;yaw=angle*f;phase='turn'
    elif elapsed<10:pos=ROOT+.020*NORMAL+[0,0,.100*(10-elapsed)/3];yaw=angle;phase='descend'
    elif elapsed<11:pos=ROOT+.020*(11-elapsed)*NORMAL;yaw=angle;phase='approach'
    elif elapsed<14:pos=ROOT-distance*(elapsed-11)/3*NORMAL;yaw=angle;phase='push'
    elif elapsed<15:pos=ROOT+(-distance+.020*(elapsed-14))*NORMAL;yaw=angle;phase='retreat'
    elif elapsed<17:pos=ROOT+(-distance+.020)*NORMAL+[0,0,.050*(elapsed-15)/2];yaw=angle;phase='withdraw'
    else:pos=ROOT+(-distance+.020)*NORMAL+[0,0,.050];yaw=angle;phase='released'
    quat=(Rotation.from_euler('z',yaw,degrees=True)*START_ROT).as_quat()[[3,0,1,2]]
    return pos,quat,phase


def precheck():
    failures=[]
    for seconds in np.linspace(START_TIME,START_TIME+20,401):
        restore();pos,quat,phase=command(float(seconds));d.qpos[t.LA:t.LA+3]=pos;d.qpos[t.LA+3:t.LA+7]=quat;mj.mj_forward(m,d)
        for c in d.contact:
            pair=[m.geom(int(g)).name for g in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in pair) and TARGET not in pair:
                failures.append(dict(t=float(seconds),phase=phase,pair=pair,distance_m=float(c.dist)))
    ctx=t.routes.context()['left'];ctx['limits']=ctx['intersection']+[.020,-.020];ctx['inset']=.020
    q=CHECK['stages'][-1]['result']['final_joints_sdk'];stages=[]
    if not failures:
        for seconds in np.linspace(START_TIME,START_TIME+20,81)[1:]:
            pos,quat,phase=command(float(seconds));ctx['root'][:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            result=t.routes.stage('left',ctx,q,phase,(pos-ctx['root'][:3,3]).tolist());stages.append(dict(t=float(seconds),result=result))
            if not result['continuous']:break
            q=result['final_joints_sdk']
    return dict(passed=not failures and len(stages)==80 and all(s['result']['continuous'] for s in stages),mode=MODE,
                contact_root_m=ROOT.tolist(),geometry_failures=failures,stages=stages,whole_arm_collision_checked=False)


def evaluate(summary,rows):
    """Evaluate saved contacts and poses without running physics again."""
    tracked=['neighbor_6','neighbor_9']+(['neighbor_4'] if MODE=='align-third' else [])
    summary['local_pair_spacing_verified']=False
    if MODE=='align-third':summary['local_three_brick_alignment_verified']=False
    held=[r for r in rows if r['t']>=START_TIME+17-1e-8]
    if held:
        metrics={}
        for n in tracked:
            xyz=np.array([r['bricks'][n]['position'] for r in held]);rot=Rotation.from_quat(np.array([r['bricks'][n]['quaternion'] for r in held])[:,[1,2,3,0]])
            pts=np.concatenate([rr.apply(CORNERS)+p for rr,p in zip(rot,xyz)])
            table_force=min(sum(c['force_xyz_n'][2] for c in r['bricks'][n]['contacts'] if 'table' in c['pair']) for r in held)
            allowed=all(all(all(a in ['table']+tracked for a in c['pair']) for c in r['bricks'][n]['contacts']) for r in held)
            metrics[n]=dict(final_position=xyz[-1].tolist(),final_yaw_degrees=float(rot[-1].as_euler('xyz',degrees=True)[2]),
                            drift_mm=float(np.linalg.norm(xyz-xyz[0],axis=1).max()*1000),angle_drift_degrees=float(np.rad2deg((rot[0].inv()*rot).magnitude().max())),
                            table_support_min_n=table_force,allowed_released_contacts=allowed,
                            footprint_minimum_m=pts[:,:2].min(0).tolist(),footprint_maximum_m=pts[:,:2].max(0).tolist(),
                            inside_analysis_rectangle=bool(np.all(pts[:,:2]>=t.REGION[:,0]) and np.all(pts[:,:2]<=t.REGION[:,1])))
        first,second=metrics['neighbor_9'],metrics['neighbor_6']
        gap=second['footprint_minimum_m'][1]-first['footprint_maximum_m'][1]
        dx=second['final_position'][0]-first['final_position'][0];dy=second['final_position'][1]-first['final_position'][1]
        summary.update(held_duration_s=held[-1]['t']-held[0]['t'],bricks=metrics,x_offset_mm=dx*1000,y_center_spacing_mm=dy*1000,y_projection_gap_mm=gap*1000)
        summary['local_pair_spacing_verified']=bool(not summary['abort'] and not summary['unintended_hand_contacts'] and max(summary['other_bricks_max_shift_mm'].values())<=2 and summary['held_duration_s']>=2.98
            and abs(dx)<=.005 and .050<=dy<=.060 and -.001<=gap<=.010
            and all(v['inside_analysis_rectangle'] and v['allowed_released_contacts'] and v['table_support_min_n']>=.8*m.body(n).mass*9.81
                    and v['drift_mm']<=2 and v['angle_drift_degrees']<=2 and abs(v['final_yaw_degrees'])<=10 for n,v in metrics.items() if n in ['neighbor_9','neighbor_6']))
        if MODE=='align-third':
            third=metrics['neighbor_4'];dx3=third['final_position'][0]-second['final_position'][0]
            dy3=third['final_position'][1]-second['final_position'][1]
            gap3=third['footprint_minimum_m'][1]-second['footprint_maximum_m'][1]
            summary.update(third_x_offset_mm=dx3*1000,third_y_center_spacing_mm=dy3*1000,third_y_projection_gap_mm=gap3*1000)
            summary['local_three_brick_alignment_verified']=bool(summary['local_pair_spacing_verified'] and abs(dx3)<=.005 and .050<=dy3<=.060 and -.001<=gap3<=.010
                and third['inside_analysis_rectangle'] and third['allowed_released_contacts'] and third['table_support_min_n']>=.8*m.body('neighbor_4').mass*9.81
                and third['drift_mm']<=2 and third['angle_drift_degrees']<=2 and abs(third['final_yaw_degrees'])<=10)


def run(out):
    restore();mj.mj_forward(m,d)
    others={m.body(i).name:i for i in range(m.nbody)
            if (m.body(i).name=='target' or m.body(i).name.startswith('neighbor_')) and m.body(i).name!=TARGET}
    initial={n:d.xpos[i].copy() for n,i in others.items()};shifts={n:0. for n in others}
    tracked=['neighbor_6','neighbor_9']+(['neighbor_4'] if MODE=='align-third' else [])
    rows=[];bad=[];abort=None;started=time.monotonic()
    with (out/'trace.jsonl').open('x') as stream:
        for step in range(round(20/m.opt.timestep)):
            pos,quat,phase=command(float(d.time));d.mocap_pos[t.LM]=pos;d.mocap_quat[t.LM]=quat
            mj.mj_step(m,d);mj.mj_forward(m,d)
            for n,i in others.items():shifts[n]=max(shifts[n],float(np.linalg.norm(d.xpos[i][:2]-initial[n][:2])))
            contacts={n:[] for n in tracked}
            for ci,c in enumerate(d.contact):
                pair=[m.geom(int(g)).name for g in c.geom]
                if any(n.startswith(('lh_','rh_')) for n in pair) and TARGET not in pair:
                    bad.append(dict(t=float(d.time),pair=pair,distance_m=float(c.dist)))
                    if c.dist<-.0005:abort='unexpected_hand_contact'
                for n in contacts:
                    if n in pair:
                        f=np.zeros(6);mj.mj_contactForce(m,d,ci,f)
                        vector=c.frame.reshape(3,3).T@f[:3]*(1 if pair[1]==n else -1)
                        contacts[n].append(dict(pair=pair,force_xyz_n=vector.tolist()))
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite state')
            if step%10==9 or abort:
                bricks={n:dict(position=d.xpos[m.body(n).id].tolist(),quaternion=d.xquat[m.body(n).id].tolist(),contacts=contacts[n]) for n in contacts}
                row=dict(t=float(d.time),phase=phase,bricks=bricks,qpos=d.qpos.tolist(),qvel=d.qvel.tolist())
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if abort:break
    summary=dict(mode=MODE,abort=abort,wall_seconds=time.monotonic()-started,simulation_duration_s=float(d.time)-START_TIME,
                 source=str(SOURCE),unintended_hand_contacts=bad,other_bricks_max_shift_mm={n:v*1000 for n,v in shifts.items()},
                 local_pair_spacing_verified=False,full_base_row_verified=False,native_verified=False,official_region_verified=False,full_pyramid_verified=False)
    evaluate(summary,rows)
    spec=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec)
    np.savez(out/'integration-state.npz',state=state,spec=int(spec),mujoco_version=mj.__version__)
    restored=mj.MjData(m);mj.mj_setState(m,restored,state,spec)
    pos,quat,_=command(float(d.time))
    for data in [d,restored]:
        data.mocap_pos[t.LM]=pos;data.mocap_quat[t.LM]=quat
        mj.mj_step(m,data);mj.mj_forward(m,data)
    original_next=np.empty_like(state);restored_next=np.empty_like(state)
    mj.mj_getState(m,d,original_next,spec);mj.mj_getState(m,restored,restored_next,spec)
    summary['checkpoint_next_step_maximum_difference']=float(np.max(np.abs(original_next-restored_next)))
    mj.mj_setState(m,d,state,spec)
    (out/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));print(json.dumps(summary,indent=2))
    verified=summary.get('local_three_brick_alignment_verified',False) if MODE=='align-third' else summary['local_pair_spacing_verified']
    return verified and summary['checkpoint_next_step_maximum_difference']<=1e-10


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['transverse','diagonal','diagonal-clear','align-first','align-first-front','align-first-side','close-final','align-third'],default='transverse');parser.add_argument('--run',action='store_true')
    args=parser.parse_args();configure(args.mode);args.out.mkdir(parents=True,exist_ok=False)
    result=precheck();(args.out/'precheck.json').write_text(json.dumps(result,allow_nan=False))
    print(json.dumps(dict(passed=result['passed'],geometry_failures=len(result['geometry_failures']),first_failure=result['geometry_failures'][:1],stages=len(result['stages']))),flush=True)
    passed=result['passed']
    if passed and args.run:passed=run(args.out)
    raise SystemExit(0 if passed else 1)
