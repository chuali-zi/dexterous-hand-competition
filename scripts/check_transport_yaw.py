"""Local SDK endpoint screen for rigid yaw of the retained bimanual carry.

No motion, collision claim, dynamics, or authenticated competition region.
The retained MuJoCo/SDK assets from rounds 78-81 must be available locally.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,'runs/edge-transfer-78/bimanual-routes')
import check_routes as routes


def inputs():
 ctx=routes.context()
 old=json.loads(Path('runs/release-79/release.json').read_text())['rows']
 row=min(old,key=lambda r:abs(r['t']-16.002));center=np.array(row['position'])
 source={s:ctx[s]['root'].copy() for s in ctx}
 for side in source:source[side][:3,3]+=[.100,0.,.020 if side=='left' else .030]
 cleared=json.loads(Path('runs/placement-81/open-table-route.json').read_text())['stages'][:2]
 seeds={s['side']:s['final_joints_sdk'] for s in cleared}
 return ctx,center,source,seeds


def rigid_goal(root,source_center,destination,yaw_degrees,pitch_degrees=0.):
 rotation=Rotation.from_euler('zy',[yaw_degrees,pitch_degrees],degrees=True).as_matrix()
 goal=root.copy();goal[:3,:3]=rotation@root[:3,:3]
 goal[:3,3]=np.asarray(destination)+rotation@(root[:3,3]-source_center)
 return goal


def screen(out,mode='yaw'):
 if out.exists():raise FileExistsError(out)
 ctx,center,source,seeds=inputs();results=[];start=time.monotonic()
 if mode=='yaw':cases=[(a,0.,0.) for a in [0.,-90.,90.,180.]]
 elif mode=='tilt':cases=[(0.,a,.060) for a in [-15.,-30.,-45.]]
 else:cases=[(0.,a,-.381168478-center[2]) for a in [-15.,-30.]]
 for angle,pitch,lift in cases:
  destination=[.400,0.,float(center[2]+lift)]
  arms={}
  for side in ['left','right']:
   goal=rigid_goal(source[side],center,destination,angle,pitch)
   target=goal@ctx[side]['inverse_mount'];limits=ctx[side]['intersection'];lo,hi=limits[:,0]+.020,limits[:,1]-.020
   def residual(q):
    actual=np.asarray(routes.sdk.forward_kinematics(side,q.tolist()))
    return np.r_[(actual[:3,3]-target[:3,3])/.001,Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec()/.01]
   tick=time.monotonic();solved=least_squares(residual,np.clip(seeds[side],lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=500 if mode=='high-tilt' else 250,ftol=1e-11,xtol=1e-11,gtol=1e-11)
   error=residual(solved.x);pe=float(np.linalg.norm(error[:3])*.001);re=float(np.linalg.norm(error[3:])*.01);margin=float(np.minimum(solved.x-limits[:,0],limits[:,1]-solved.x).min())
   ok=pe<1e-5 and re<1e-4 and margin>=.020-1e-10
   arms[side]=dict(accepted=ok,root_pose=goal.tolist(),joints_sdk=solved.x.tolist(),position_error_m=pe,orientation_error_rad=re,minimum_joint_margin_rad=margin,nfev=int(solved.nfev),wall_seconds=time.monotonic()-tick)
   print(json.dumps(dict(yaw=angle,pitch=pitch,side=side,accepted=ok,position_error_mm=pe*1000,orientation_error_rad=re,margin_rad=margin)),flush=True)
  results.append(dict(yaw_degrees=angle,pitch_degrees=pitch,destination_object_center_m=destination,arms=arms,both_accepted=all(x['accepted'] for x in arms.values())))
 result=dict(mode=mode,source_object_center_m=center.tolist(),endpoint_results=results,wall_seconds=time.monotonic()-start,scope=__doc__,native_verified=False,dynamics_run=False)
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(result,indent=2,allow_nan=False))
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--mode',choices=['yaw','tilt','high-tilt'],default='yaw')
 args=parser.parse_args();screen(args.out,args.mode)
