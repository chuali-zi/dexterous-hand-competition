"""Full local retained pickup plus one elevated tilted transport trial.

Requires retained runs/78-82 assets. No Rabokit calls. Object truth is only
logged/evaluated and used to terminate a failed local diagnostic, never to
adapt the fixed movement or serve as a competition controller.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path[:0]=['runs/edge-transfer-78/bimanual','scripts']
import precheck_bimanual as p
m,d,mj=p.m,p.d,p.mujoco
REFERENCE=json.loads(Path('runs/release-79/release.json').read_text())['rows']
SOURCE=min(REFERENCE,key=lambda r:abs(r['t']-16.002))
CENTER=np.array(SOURCE['position']);ROOTS=p.command(16)[0]
MOCAPS={s:int(m.body_mocapid[m.body(n).id]) for s,n in [('left','root_reference'),('right','right_root_reference')]}
FINAL_Z=-.381168478
TILT_FIRST=False
ROUTE_CHECK=None


def desired(t):
 center=CENTER.copy();pitch=0.
 if t<16:return center,pitch,'retained_pickup'
 if TILT_FIRST and t<24:
  if t<20:
   f=(t-16)/4;center[2]+=.060*f;return center,-30*f,'tilt_and_lift'
  center[2]+=.060;center[1]*=1-(t-20)/4;return center,-30.,'clear_supply_row'
 if t<20:center[1]*=1-(t-16)/4;return center,pitch,'clear_supply_row'
 center[1]=0
 if t<24:
  f=(t-20)/4;center[2]+=.060*f;return center,-30*f,'tilt_and_lift'
 if t<27:
  center[2]+=.060+(FINAL_Z-CENTER[2]-.060)*(t-24)/3;return center,-30.,'raise_to_transit'
 center[2]=FINAL_Z
 if t<35:center[0]+=(.400-CENTER[0])*(t-27)/8;return center,-30.,'inward_transfer'
 center[0]=.400;return center,-30.,'held_at_destination'


def commands(t,initial=False):
 center,pitch,phase=desired(t);rot=Rotation.from_euler('y',pitch,degrees=True)
 if t<=16:positions,_=p.command(t)
 else:positions={s:center+rot.apply(np.array(x)-CENTER) for s,x in ROOTS.items()}
 for side,address in p.roots.items():
  quat=(rot*Rotation.from_quat(np.array(p.quats[side])[[1,2,3,0]])).as_quat()[[3,0,1,2]]
  d.mocap_pos[MOCAPS[side]]=positions[side];d.mocap_quat[MOCAPS[side]]=quat
  if initial:d.qpos[address:address+3]=positions[side];d.qpos[address+3:address+7]=quat
 for name,value in p.targets.items():
  d.ctrl[m.actuator(name).id]=value
  if initial:d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
 return center,pitch,phase


def geometry(require_route=True):
 if require_route:
  path='runs/transport-82/tilt-first-route-v2.json' if TILT_FIRST else 'runs/transport-82/high-route.json'
  route=json.loads((ROUTE_CHECK or Path(path)).read_text())
  if route.get('tilt_first',False)!=TILT_FIRST:raise ValueError('Route order differs from the requested trial')
  if not route['continuous'] or not route['geometry_passed']:raise ValueError('Retained numerical/frozen route did not pass')
 bad=[]
 for t in np.linspace(16,35,381):
  center,pitch,phase=desired(t);rot=Rotation.from_euler('y',pitch,degrees=True);d.qpos[:]=SOURCE['qpos']
  for name,address in list(p.roots.items())+[('target',int(m.jnt_qposadr[m.joint('target_free').id]))]:
   old_pos=np.array(SOURCE['qpos'][address:address+3]);old_q=np.array(SOURCE['qpos'][address+3:address+7])
   d.qpos[address:address+3]=center+rot.apply(old_pos-CENTER)
   d.qpos[address+3:address+7]=(rot*Rotation.from_quat(old_q[[1,2,3,0]])).as_quat()[[3,0,1,2]]
  mj.mj_forward(m,d)
  for c in d.contact:
   names=[m.geom(int(g)).name for g in c.geom];hand=any(n.startswith(('lh_','rh_')) for n in names)
   if (hand and 'target' not in names) or ('target' in names and not hand):bad.append(dict(t=float(t),phase=phase,pair=names,depth_mm=-float(c.dist)*1000))
 return dict(passed=not bad,samples=381,unexpected_contacts=bad,scope='Frozen loaded hand geometry; not a physical prediction or full-arm collision check')


def run(out):
 out.mkdir(parents=True,exist_ok=False);check=geometry();(out/'precheck.json').write_text(json.dumps(check,indent=2))
 if not check['passed']:print(json.dumps(dict(geometry_passed=False,unexpected_contacts=len(check['unexpected_contacts']))));return False
 mj.mj_resetData(m,d);commands(0,True);mj.mj_forward(m,d)
 target=m.body('target').id;rows=[];bad=[];abort=None;start=time.monotonic();prefix_error=0.
 with (out/'trace.jsonl').open('x') as stream:
  for step in range(round(38/m.opt.timestep)+1):
   planned,pitch,phase=commands(float(d.time));mj.mj_step(m,d);mj.mj_forward(m,d)
   for c in d.contact:
    names=[m.geom(int(g)).name for g in c.geom]
    if any(n.startswith(('lh_','rh_')) for n in names) and 'target' not in names:
     bad.append(dict(t=float(d.time),pair=names,depth_mm=-float(c.dist)*1000))
     if c.dist<-.0005:abort='unintended_hand_collision'
   if d.time>=16 and np.linalg.norm(d.xpos[target]-planned)>.050:abort='object_left_50mm_transport_envelope'
   if step%10==0 or abort:
    contacts=[]
    for ci,c in enumerate(d.contact):
     names=[m.geom(int(g)).name for g in c.geom]
     if 'target' not in names:continue
     f=np.zeros(6);mj.mj_contactForce(m,d,ci,f)
     force=c.frame.reshape(3,3).T@f[:3]*(1 if names[1]=='target' else -1)
     contacts.append(dict(pair=names,normal_n=float(f[0]),force_xyz_n=force.tolist()))
    row=dict(t=float(d.time),phase=phase,planned_center=planned.tolist(),planned_pitch_degrees=pitch,position=d.xpos[target].tolist(),quaternion=d.xquat[target].tolist(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),contacts=contacts)
    if len(rows)<800:prefix_error=max(prefix_error,float(np.max(np.abs(d.qpos-REFERENCE[len(rows)]['qpos']))))
    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
   if abort:break
 held=[r for r in rows if r['t']>=35.002-1e-8]
 summary=dict(abort=abort,tilt_first=TILT_FIRST,wall_seconds=time.monotonic()-start,simulation_seconds=float(d.time),prefix_max_qpos_difference=prefix_error,unintended_hand_contacts=bad,local_transport_verified=False,native_verified=False,full_pyramid_verified=False,official_region_verified=False)
 if held:
  positions=np.array([r['position'] for r in held]);goal=np.array([.400,0.,FINAL_Z])
  drift=float(np.linalg.norm(positions-positions[0],axis=1).max());error=float(np.linalg.norm(positions-goal,axis=1).max())
  only_hands=all(all(n=='target' or n.startswith(('lh_','rh_')) for n in c['pair']) for r in held for c in r['contacts'])
  both_hands=all(all(any(any(n.startswith(prefix) for n in c['pair']) and c['normal_n']>.05 for c in r['contacts']) for prefix in ['lh_','rh_']) for r in held)
  expected=Rotation.from_euler('y',-30,degrees=True)*Rotation.from_quat(np.array(SOURCE['quaternion'])[[1,2,3,0]])
  orientation_error=float(max((expected.inv()*Rotation.from_quat(np.array(r['quaternion'])[[1,2,3,0]])).magnitude() for r in held))
  duration=held[-1]['t']-held[0]['t']
  summary.update(hold_duration_s=duration,position_drift_mm=drift*1000,maximum_center_error_mm=error*1000,maximum_orientation_error_degrees=float(np.rad2deg(orientation_error)),only_hand_support=only_hands,both_hands_loaded=both_hands)
  summary['local_transport_verified']=bool(not abort and not bad and duration>=2.98 and drift<=.005 and error<=.020 and orientation_error<=np.deg2rad(10) and only_hands and both_hands)
 (out/'transport.json').write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False));print(json.dumps(summary,indent=2));return summary['local_transport_verified']

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--tilt-first',action='store_true');parser.add_argument('--route-check',type=Path);args=parser.parse_args();TILT_FIRST=args.tilt_first;ROUTE_CHECK=args.route_check
 raise SystemExit(0 if run(args.out) else 1)
