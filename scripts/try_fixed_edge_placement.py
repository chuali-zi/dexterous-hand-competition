"""Local fixed diagonal placement experiment, using retained runs/76-79 assets.

Run from the repository root. Requires the retained MuJoCo model, meshes,
SDK cache and numerical helper scripts; not a standalone installation or a
Rabokit controller. Object poses are used for evaluation, never for control.
"""
import argparse,sys,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path[:0]=['runs/edge-transfer-78/bimanual','runs/edge-transfer-78/bimanual-routes']
import precheck_bimanual as p
import check_routes as routes
m,d,mj=p.m,p.d,p.mujoco
OUT=None
ALIGN_ON_SUPPORT=False
BRIDGE_ADJACENT=False
PUSHER_DROP_MM=0.
PLACEMENT_SHIFT=np.array([.014,.019,0.])
old=json.loads(Path('runs/edge-transfer-78/bimanual/bimanual.json').read_text())
old['rows']=old['rows'][:800]  # Only samples strictly before the changed t=16 segment.
shifts={'left':np.array([.04,0,0]),'right':np.array([0,-.05,0])}
mocaps={s:int(m.body_mocapid[m.body(n).id]) for s,n in [('left','root_reference'),('right','right_root_reference')]}

ALIGN_STAGES=[
 (27.,30.5,'right',[0.,.055,0.],'push_y'),
 (30.5,33.5,'right',[0.,-.055,0.],'retreat_y'),
 (33.5,35.5,'left',[0.,.020,-.020],'approach_x'),
 (35.5,38.5,'left',[-.035,0.,0.],'push_x'),
 (38.5,40.5,'left',[.040,0.,0.],'retreat_x'),
 (40.5,41.5,'left',[0.,0.,.020],'raise_clear'),
]


def alignment_offsets(t):
 offsets={s:np.zeros(3) for s in p.roots}
 for start,end,side,delta,_ in ALIGN_STAGES:
  offsets[side]+=np.array(delta)*np.clip((t-start)/(end-start),0,1)
 return offsets


def commands(t,initial=False):
 positions,phase=p.command(min(t,22))
 for side in positions:
  positions[side]=np.array(positions[side])+PLACEMENT_SHIFT*np.clip((t-16)/3,0,1)
 if t>=22:
  phase='withdraw' if t<24 else 'released'
  for s in positions:positions[s]=np.array(positions[s])+shifts[s]*min((t-22)/2,1)
 if ALIGN_ON_SUPPORT and t>=27:
  for side,delta in alignment_offsets(t).items():positions[side]+=delta
  phase=next((name for start,end,_,_,name in ALIGN_STAGES if start<=t<end),'aligned_released')
 for s,a in p.roots.items():
  d.mocap_pos[mocaps[s]]=positions[s];d.mocap_quat[mocaps[s]]=p.quats[s]
  if initial:d.qpos[a:a+3]=positions[s];d.qpos[a+3:a+7]=p.quats[s]
 for n,v in p.targets.items():
  d.ctrl[m.actuator(n).id]=v
  if initial:d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
 return phase

def check():
 ctx=routes.context()
 left=json.loads(Path('runs/edge-transfer-78/bimanual-routes/direct-results.json').read_text())['stages'][1]['arms']['left']['final_joints_sdk']
 right=json.loads(Path('runs/edge-transfer-78/bimanual-routes/right-extra10-results.json').read_text())['stages'][1]['final_joints_sdk']
 checks={}
 for side,q,z in [('left',left,.080),('right',right,.090)]:
  offset=np.array([.100,0.,z])+PLACEMENT_SHIFT
  lift=routes.stage(side,ctx[side],q,'diagonal_place',offset.tolist())
  checks[side]=[lift]
  if lift['continuous']:
   checks[side].append(routes.stage(side,ctx[side],lift['final_joints_sdk'],'withdraw',(offset+shifts[side]).tolist()))
 bad=[]
 retained=json.loads(Path('runs/release-79/release.json').read_text())['rows']
 for row in retained:
  if row['t']<16:continue
  d.qpos[:]=row['qpos'];d.qvel[:]=0
  delta=PLACEMENT_SHIFT*np.clip((row['t']-16)/3,0,1)
  for address in p.roots.values():d.qpos[address:address+3]+=delta
  address=int(m.jnt_qposadr[m.joint('target_free').id]);d.qpos[address:address+3]+=delta
  mj.mj_forward(m,d)
  for c in d.contact:
   names=[m.geom(int(g)).name for g in c.geom]
   if any(n.startswith(('lh_','rh_')) for n in names) and 'target' not in names:
    bad.append(dict(t=row['t'],pair=names,depth_mm=-float(c.dist)*1000))
 result=dict(ik=checks,non_target_contacts=bad,passed=all(len(v)==2 and all(s['continuous'] for s in v) for v in checks.values()) and not bad,
 scope='Changed diagonal and withdrawal paths only; frozen previous loaded shape. No full-arm collision or native verification.')
 (OUT/'precheck.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(precheck_passed=result['passed'],non_target_contact_count=len(bad))))
 return result['passed']


def check_alignment():
 retained=json.loads(Path('runs/placement-80/diagonal/release.json').read_text())
 source=json.loads(Path('runs/placement-80/diagonal/precheck.json').read_text())
 ctx=routes.context();checks=[];bad=[]
 q={s:source['ik'][s][-1]['final_joints_sdk'] for s in p.roots}
 for start,end,side,delta,name in ALIGN_STAGES:
  offset=np.array([.100,0.,.080 if side=='left' else .090])+PLACEMENT_SHIFT+shifts[side]+alignment_offsets(end)[side]
  checked=routes.stage(side,ctx[side],q[side],name,offset.tolist());checks.append(checked)
  if not checked['continuous']:break
  q[side]=checked['final_joints_sdk']
 # Preserve actual unloaded finger geometry. Target contact is intentional;
 # support bricks and table remain at their observed final positions.
 for t in np.linspace(27,ALIGN_STAGES[-1][1],round((ALIGN_STAGES[-1][1]-27)/.05)+1):
  d.qpos[:]=retained['rows'][-1]['qpos'];d.qvel[:]=0
  for side,delta in alignment_offsets(t).items():
   address=p.roots[side];d.qpos[address:address+3]+=delta
  mj.mj_forward(m,d)
  for c in d.contact:
   names=[m.geom(int(g)).name for g in c.geom]
   if any(n.startswith(('lh_','rh_')) for n in names) and 'target' not in names:
    bad.append(dict(t=float(t),pair=names,depth_mm=-float(c.dist)*1000))
 passed=len(checks)==len(ALIGN_STAGES) and all(x['continuous'] for x in checks) and not bad
 (OUT/'alignment-precheck.json').write_text(json.dumps(dict(passed=passed,ik=checks,non_target_contacts=bad),indent=2))
 print(json.dumps(dict(alignment_precheck_passed=passed,non_target_contact_count=len(bad))))
 return passed


def run():
 out=OUT/'release.json'
 if out.exists():raise RuntimeError('Result already exists')
 mj.mj_resetData(m,d);commands(0,True);mj.mj_forward(m,d)
 target=m.body('target').id;support=m.body('neighbor_6').id;second_support=m.body('neighbor_4').id
 corners=np.array([[x,y,z] for x in [-.025,.025] for y in [-.025,.025] for z in [-.025,.025]])
 rows=[];unexpected=[];start=time.monotonic();abort=None;prefix_error=0.
 with (OUT/'trace.jsonl').open('x') as stream:
  for step in range(round(((ALIGN_STAGES[-1][1]+3) if ALIGN_ON_SUPPORT else 27)/m.opt.timestep)+1):
   phase=commands(float(d.time));mj.mj_step(m,d);mj.mj_forward(m,d)
   for c in d.contact:
    names=[m.geom(int(g)).name for g in c.geom]
    if any(n.startswith(('lh_','rh_')) for n in names) and 'target' not in names:
     unexpected.append(dict(t=float(d.time),pair=names,depth_mm=-float(c.dist)*1000))
     if c.dist<-.0005:abort='unintended_hand_collision'
   if step%10==0 or abort:
    contact=[]
    for ci,c in enumerate(d.contact):
     names=[m.geom(int(g)).name for g in c.geom]
     if 'target' not in names:continue
     f=np.zeros(6);mj.mj_contactForce(m,d,ci,f)
     force=c.frame.reshape(3,3).T@f[:3]*(1 if names[1]=='target' else -1)
     contact.append(dict(pair=names,normal_n=float(f[0]),up_n=float(force[2])))
    pos=d.xpos[target].copy();quat=d.xquat[target].copy()
    row=dict(t=float(d.time),phase=phase,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),position=pos.tolist(),quaternion=quat.tolist(),
       support_position=d.xpos[support].tolist(),support_quaternion=d.xquat[support].tolist(),second_support_position=d.xpos[second_support].tolist(),contacts=contact,
       bottom_m=float((corners@d.xmat[target].reshape(3,3).T+pos)[:,2].min()-p.FLOOR))
    if len(rows)<len(old['rows']):prefix_error=max(prefix_error,float(np.max(np.abs(d.qpos-old['rows'][len(rows)]['qpos']))))
    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
   if abort:break
 held=[r for r in rows if r['t']>=((ALIGN_STAGES[-1][1]+.002) if ALIGN_ON_SUPPORT else 24.002)-1e-8]
 if not held:
  summary=dict(abort=abort or 'missing_released_window',wall_seconds=time.monotonic()-start,simulation_seconds=float(d.time),unintended_hand_contacts=unexpected,local_released_two_brick_stack=False,centered_placement=False,native_verified=False,full_pyramid_verified=False)
  out.write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False))
  print(json.dumps(summary,indent=2));return False
 positions=np.array([r['position'] for r in held]);rot=Rotation.from_quat(np.array([r['quaternion'] for r in held])[:,[1,2,3,0]])
 drift=float(np.linalg.norm(positions-positions[0],axis=1).max()*1000)
 angular=float(np.rad2deg((rot[0].inv()*rot).magnitude().max()))
 hand_clear=all(not any(any(n.startswith(('lh_','rh_')) for n in c['pair']) for c in r['contacts']) for r in held)
 supported=all(any('neighbor_6' in c['pair'] and c['up_n']>.1 for c in r['contacts']) for r in held)
 offset=positions[-1,:2]-np.array(held[-1]['support_position'][:2])
 summary=dict(abort=abort,wall_seconds=time.monotonic()-start,simulation_seconds=float(d.time),prefix_max_qpos_difference=prefix_error,
 released_duration_s=held[-1]['t']-held[0]['t'],released_samples=len(held),position_drift_mm=drift,rotation_drift_deg=angular,
 hand_contact_absent=hand_clear,red_brick_support_continuous=supported,final_xy_offset_from_support_mm=(offset*1000).tolist(),
 final_bottom_above_table_mm=held[-1]['bottom_m']*1000,unintended_hand_contacts=unexpected,
 local_released_two_brick_stack=bool(not abort and not unexpected and hand_clear and supported and drift<=.002*1000 and angular<=2 and held[-1]['t']-held[0]['t']>=2.98),
 native_verified=False,full_pyramid_verified=False,official_build_region_verified=False)
 summary['planned_diagonal_shift_m']=PLACEMENT_SHIFT.tolist()
 summary['alignment_on_support']=ALIGN_ON_SUPPORT
 summary['center_tolerance_mm']=5.
 summary['centered_placement']=bool(summary['local_released_two_brick_stack'] and np.linalg.norm(offset)<=.005)
 summary['bridge_adjacent']=BRIDGE_ADJACENT
 summary['pusher_drop_mm']=PUSHER_DROP_MM
 if BRIDGE_ADJACENT:
  second_supported=all(any('neighbor_4' in c['pair'] and c['up_n']>.1 for c in r['contacts']) for r in held)
  center=(np.array(held[-1]['support_position'][:2])+held[-1]['second_support_position'][:2])/2
  bridge_offset=positions[-1,:2]-center
  summary['second_support_continuous']=second_supported
  summary['bridge_center_error_mm']=float(np.linalg.norm(bridge_offset)*1000)
  summary['bridge_center_offset_mm']=(bridge_offset*1000).tolist()
  summary['released_bridge']=bool(summary['local_released_two_brick_stack'] and second_supported)
  summary['centered_placement']=bool(summary['released_bridge'] and np.linalg.norm(bridge_offset)<=.005)
 out.write_text(json.dumps(dict(summary=summary,rows=rows),allow_nan=False))
 print(json.dumps(summary,indent=2))
 return summary['centered_placement']

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=Path,required=True,help='New local result directory; existing results are never overwritten.')
 parser.add_argument('--align-on-support',action='store_true',help='Append two fixed side pushes after the retained diagonal placement.')
 parser.add_argument('--bridge-adjacent',action='store_true',help='Push toward the midpoint of adjacent lower bricks 6 and 4; implies --align-on-support.')
 parser.add_argument('--bridge-pusher-drop-mm',type=float,default=0.,help='Lower the right pusher before bridging (0 to 20 mm); requires --bridge-adjacent.')
 args=parser.parse_args();OUT=args.out;BRIDGE_ADJACENT=args.bridge_adjacent;ALIGN_ON_SUPPORT=args.align_on_support or BRIDGE_ADJACENT
 PUSHER_DROP_MM=args.bridge_pusher_drop_mm
 if not 0<=PUSHER_DROP_MM<=20 or (PUSHER_DROP_MM and not BRIDGE_ADJACENT):parser.error('Pusher drop requires bridge mode and a finite value from 0 to 20 mm.')
 if BRIDGE_ADJACENT:
  ALIGN_STAGES[0]=(27.,30.5,'right',[0.,.085,0.],'push_y')
  ALIGN_STAGES[1]=(30.5,33.5,'right',[0.,-.085,0.],'retreat_y')
  ALIGN_STAGES[2]=(33.5,35.5,'left',[0.,.050,-.020],'approach_x')
  if PUSHER_DROP_MM:
   push=ALIGN_STAGES[:2];rest=ALIGN_STAGES[2:]
   ALIGN_STAGES=[(27.,28.,'right',[0.,0.,-PUSHER_DROP_MM/1000],'lower_pusher')]
   ALIGN_STAGES += [(a+1,b+1,s,v,n) for a,b,s,v,n in push]
   ALIGN_STAGES += [(34.5,35.5,'right',[0.,0.,PUSHER_DROP_MM/1000],'raise_pusher')]
   ALIGN_STAGES += [(a+2,b+2,s,v,n) for a,b,s,v,n in rest]
 OUT.mkdir(parents=True,exist_ok=False)
 if ALIGN_ON_SUPPORT:
  old=json.loads(Path('runs/placement-80/diagonal/release.json').read_text())
  old['rows']=old['rows'][:1350]
  if not check_alignment():raise SystemExit('Alignment precheck failed; no dynamics')
 elif not check():raise SystemExit('Placement precheck failed; no dynamics')
 raise SystemExit(0 if run() else 1)
