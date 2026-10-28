"""Read-only loaded geometry and SDK paths for the fixed transport controller.

Requires local retained assets; never starts dynamics, ROS or Rabokit.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import check_transport_yaw as study
import try_tilted_transport as trial


def check(output,tilt_first):
 trial.TILT_FIRST=tilt_first
 geometry=trial.geometry(require_route=False)
 report=dict(tilt_first=tilt_first,geometry_passed=geometry['passed'],geometry=geometry,continuous=False,stages=[],dynamics_run=False,native_verified=False)
 output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(output)
 if not geometry['passed']:output.write_text(json.dumps(report));return report
 ctx,_,_,_=study.inputs()
 q={'left':json.loads(Path('runs/edge-transfer-78/bimanual-routes/direct-results.json').read_text())['stages'][1]['arms']['left']['final_joints_sdk'],'right':json.loads(Path('runs/edge-transfer-78/bimanual-routes/right-extra10-results.json').read_text())['stages'][1]['final_joints_sdk']}
 for side in ctx:ctx[side]['limits']=ctx[side]['intersection']+np.array([.020,-.020]);ctx[side]['inset']=.020
 started=time.monotonic()
 for t in np.linspace(16,35,77)[1:]:
  center,pitch,phase=trial.commands(float(t));row=dict(t=float(t),phase=phase,arms={})
  for side in ['left','right']:
   mid=trial.MOCAPS[side];goal=ctx[side]['root'].copy();goal[:3,3]=trial.d.mocap_pos[mid]
   goal[:3,:3]=trial.Rotation.from_quat(trial.d.mocap_quat[mid][[1,2,3,0]]).as_matrix()
   context=dict(ctx[side]);context['root']=ctx[side]['root'].copy();context['root'][:3,:3]=goal[:3,:3]
   result=study.routes.stage(side,context,q[side],phase,(goal[:3,3]-context['root'][:3,3]).tolist());row['arms'][side]=result
   if not result['continuous']:break
   q[side]=result['final_joints_sdk']
  report['stages'].append(row)
  if len(row['arms'])<2 or not all(r['continuous'] for r in row['arms'].values()):break
 else:report['continuous']=True
 report['wall_seconds']=time.monotonic()-started;output.write_text(json.dumps(report,allow_nan=False))
 return report

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--tilt-first',action='store_true')
 args=parser.parse_args();r=check(args.out,args.tilt_first)
 print(json.dumps(dict(geometry_passed=r['geometry_passed'],continuous=r['continuous'],stages=len(r['stages']),last_phase=r['stages'][-1]['phase'] if r['stages'] else None)))
 raise SystemExit(0 if r['continuous'] else 1)
