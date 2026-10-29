"""Compare fixed-pose endpoints across an entire 4+3+2+1 workcell, locally.

No objects are moved and no dynamics run. Passing means only bounded numerical
FK agreement, not collision-free routes, a legal region, or a buildable stack.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import align_released_yellow as a


LOCATIONS=[('old_location',None,None),('nearer_x',.35,None),('centered_south',.35,-.08),
           ('near_south',.30,-.08),('balanced',.30,-.04)]


def compare(only=None):
    if only is not None and only not in [x[0] for x in LOCATIONS]:raise ValueError('Unknown comparison location')
    started=time.monotonic();source,restore,left_seed,_=a.configure('y');restore()
    p,m,d=a.p,a.m,a.d;contexts=p.t.routes.context()
    check=json.loads((source/'precheck.json').read_text())
    right_seed=check['stages']['right'][-1]['result']['final_joints_sdk']
    red=d.xpos[m.body('neighbor_9').id].copy();yellow=d.xpos[m.body(a.TARGET).id].copy()
    _,upper=a.prepare('y',source,restore);restore()
    left_offset=np.array(upper['contact_root_m'])-yellow
    left_rotation=Rotation.from_euler('z',90,degrees=True)*Rotation.from_quat(p.t.QUAT[[1,2,3,0]])
    current_rotation=Rotation.from_quat(d.mocap_quat[p.t.RM][[1,2,3,0]])
    # Recompute the original, fixed-orientation, low brace geometry. Alternative
    # orientations are deliberately not searched in this endpoint comparison.
    for n,v in p.t.CANDIDATE['hand_targets'].items():d.qpos[m.jnt_qposadr[m.joint(n.replace('lh_','rh_')).id]]=v
    d.qpos[p.t.RA:p.t.RA+3]=0;d.qpos[p.t.RA+3:p.t.RA+7]=current_rotation.as_quat()[[3,0,1,2]];a.mj.mj_forward(m,d)
    vertices=np.concatenate([a.placement.a.g.mesh_vertices('rh_'+n) for n in ['index_proximal','index_distal','middle_proximal','middle_distal']])
    restore();corners=Rotation.from_quat(d.xquat[m.body('neighbor_9').id][[1,2,3,0]]).apply(p.CORNERS)+red
    brace_root=np.array([red[0]-(vertices[:,0].min()+vertices[:,0].max())/2,corners[:,1].min()-vertices[:,1].max(),a.placement.a.g.FLOOR+.005-vertices[:,2].min()])
    brace_offset=brace_root-red
    hover_check=json.loads(Path('runs/upper-layer-93/loaded-transfer/precheck.json').read_text())
    hover_seed=hover_check['stages']['right'][-1]['final_joints_sdk']
    hover_pose=np.asarray(p.t.routes.sdk.forward_kinematics('right',hover_seed))@contexts['right']['mount']
    hover_target=json.loads(Path('runs/upper-layer-93/loaded-transfer/result.json').read_text())['rows'][-1]['position_m']
    hover_offset=hover_pose[:3,3]-np.array([hover_target[0],hover_target[1],a.placement.a.g.FLOOR+.075])
    hover_rotation=Rotation.from_matrix(hover_pose[:3,:3])
    candidates=[]
    for label,x,y in LOCATIONS:
        if only is not None and label!=only:continue
        x=red[0] if x is None else x;y=red[1] if y is None else y
        checks=[];slots=[]
        for layer in range(4):
            for index in range(4-layer):
                center=np.array([x,y+(index+layer/2)*.055,a.placement.a.g.FLOOR+.025+layer*.05])
                slots.append(dict(layer=layer,index=index,center_m=center.tolist()))
                requests=[('left_push','left',left_offset,left_rotation,left_seed)]
                if layer<3:requests.append(('right_brace','right',brace_offset,current_rotation,right_seed))
                if layer>0:
                    requests.extend([('right_hover','right',hover_offset,hover_rotation,hover_seed),('right_lower_50mm','right',hover_offset-[0,0,.05],hover_rotation,hover_seed)])
                for name,side,offset,rotation,seed in requests:
                    ctx=contexts[side];bounds=ctx['intersection']+[.02,-.02];target=np.eye(4);target[:3,3]=center+offset;target[:3,:3]=rotation.as_matrix()
                    def forward(q):return np.asarray(p.t.routes.sdk.forward_kinematics(side,q.tolist()))@ctx['mount']
                    def residual(q):
                        h=forward(q);return np.r_[(h[:3,3]-target[:3,3])/1e-5,Rotation.from_matrix(target[:3,:3].T@h[:3,:3]).as_rotvec()/1e-4,(q-np.array(seed))*.01]
                    best=None
                    for guess in [np.array(seed),np.clip(np.zeros(7),bounds[:,0]+.001,bounds[:,1]-.001)]:
                        solved=least_squares(residual,guess,bounds=(bounds[:,0],bounds[:,1]),max_nfev=300);h=forward(solved.x)
                        pe=float(np.linalg.norm(h[:3,3]-target[:3,3]));ae=float(Rotation.from_matrix(target[:3,:3].T@h[:3,:3]).magnitude())
                        item=dict(name=name,side=side,layer=layer,index=index,accepted=pe<1e-5 and ae<1e-4,
                                  position_error_m=pe,angle_error_rad=ae,q=solved.x.tolist(),root=target[:3,3].tolist(),
                                  quaternion_wxyz=rotation.as_quat()[[3,0,1,2]].tolist(),
                                  minimum_margin_rad=float(np.minimum(solved.x-ctx['intersection'][:,0],ctx['intersection'][:,1]-solved.x).min()))
                        if item['accepted'] or best is None or pe+ae*.1<best['position_error_m']+best['angle_error_rad']*.1:best=item
                        if item['accepted']:break
                    checks.append(best)
        result=dict(label=label,base_x_m=x,base_start_y_m=y,spacing_m=.055,slots=slots,checks=checks,
                    accepted_count=sum(c['accepted'] for c in checks),total=len(checks))
        candidates.append(result)
        print(json.dumps({k:result[k] for k in ['label','accepted_count','total']}),flush=True)
    restore()
    return dict(candidates=candidates,wall_seconds=time.monotonic()-started,
                scope='Endpoints only; no scene reset, dynamics, collision checks, continuous relocation, scoring or official-region verification',native_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--candidate',choices=[x[0] for x in LOCATIONS]);args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    report=compare(args.candidate);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,allow_nan=False))
