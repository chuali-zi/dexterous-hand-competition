"""Read-only upper-layer tipping contact geometry and SDK endpoint check.

Uses retained local base state. No dynamics or platform calls. Endpoint IK
alone does not validate an approach or prove that a brick will tip or lift.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import try_close_staged_pair as p

m,mj,d=p.m,p.mj,p.d
SOURCE=Path('runs/compact-base-90/contact-final')
STATE=np.load(SOURCE/'integration-state.npz',allow_pickle=False)
SOURCE_SUMMARY=json.loads((SOURCE/'result.json').read_text())['summary']
FLOOR=p.previous.w.t.p.FLOOR


def restore():
    if (not SOURCE_SUMMARY.get('complete_sequence') or not SOURCE_SUMMARY.get('local_four_brick_alignment_verified')
            or SOURCE_SUMMARY.get('abort') or SOURCE_SUMMARY.get('unexpected_hand_contacts')
            or SOURCE_SUMMARY.get('checkpoint_next_step_maximum_difference',float('inf'))>1e-10):
        raise ValueError('Source is not a verified local four-red checkpoint')
    if str(STATE['mujoco_version'])!=mj.__version__ or len(STATE['state'])!=mj.mj_stateSize(m,int(STATE['spec'])):
        raise ValueError('Checkpoint runtime or dimensions differ')
    mj.mj_setState(m,d,STATE['state'],int(STATE['spec']));mj.mj_forward(m,d)


def mesh_vertices(name):
    g=m.geom(name).id;mesh=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
    return m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]


def endpoint(side,root,quat):
    ctx=p.t.routes.context()[side];goal=np.eye(4);goal[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();goal[:3,3]=root
    target=goal@ctx['inverse_mount'];limits=ctx['intersection']+[.020,-.020]
    seed=ctx['seed'] if side=='right' else json.loads(Path('runs/fourth-base-89/aligned-low/precheck.json').read_text())['stages'][-1]['result']['final_joints_sdk']
    def residual(q):
        actual=np.array(p.t.routes.sdk.forward_kinematics(side,q.tolist()))
        return np.r_[(actual[:3,3]-target[:3,3])/1e-5,Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec()/1e-4,
                     (q-limits.mean(1))/((limits[:,1]-limits[:,0])/2)]
    solved=least_squares(residual,np.clip(seed,limits[:,0],limits[:,1]),bounds=(limits[:,0],limits[:,1]),max_nfev=200)
    err=residual(solved.x);pe=float(np.linalg.norm(err[:3])*1e-5);ae=float(np.linalg.norm(err[3:6])*1e-4)
    return dict(accepted=pe<1e-5 and ae<1e-4,position_error_m=pe,angle_error_rad=ae,joints=solved.x.tolist(),
                minimum_joint_margin_rad=float(np.minimum(solved.x-ctx['intersection'][:,0],ctx['intersection'][:,1]-solved.x).min()),
                continuous_approach_checked=False)


def check(target='neighbor_5'):
    restore();center=d.xpos[m.body(target).id].copy();q=d.xquat[m.body(target).id].copy()
    target_points=Rotation.from_quat(q[[1,2,3,0]]).apply(p.CORNERS)+center
    supports={n:d.xpos[m.body(n).id].copy() for n in ['neighbor_4','neighbor_2']}
    goal=(supports['neighbor_4']+supports['neighbor_2'])/2+[0,0,.050]
    results=[];started=time.monotonic()
    for side,prefix,address in [('left','lh',p.t.LA),('right','rh',p.t.RA)]:
        restore()
        targets={name.replace('lh_',prefix+'_'):v for name,v in p.t.CANDIDATE['hand_targets'].items()}
        for name,value in targets.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        quat=p.t.QUAT.copy();d.qpos[address:address+3]=0.;d.qpos[address+3:address+7]=quat;mj.mj_forward(m,d)
        vertices=np.concatenate([mesh_vertices(prefix+'_'+name) for name in ['index_distal','index_proximal','middle_distal','middle_proximal']])
        # Meet the outside (+X) surface, centred in Y, with the lowest fingertip
        # 40 mm above the table to avoid pushing near the brick's bottom.
        root=np.array([target_points[:,0].max()-vertices[:,0].min()-.0005,center[1]-(vertices[:,1].min()+vertices[:,1].max())/2,
                       FLOOR+.040-vertices[:,2].min()])
        d.qpos[address:address+3]=root;mj.mj_forward(m,d)
        contacts=[dict(pair=[m.geom(int(g)).name for g in c.geom],distance_m=float(c.dist),position_m=c.pos.tolist()) for c in d.contact
                  if any(m.geom(int(g)).name.startswith(prefix+'_') for g in c.geom)]
        bad=[c for c in contacts if target not in c['pair']]
        results.append(dict(side=side,root_position_m=root.tolist(),root_quaternion_wxyz=quat.tolist(),hand_targets=targets,
                            contacts=contacts,non_target_contacts=bad,endpoint=endpoint(side,root,quat)))
    return dict(source=str(SOURCE),target=target,target_center_m=center.tolist(),upper_goal_center_m=goal.tolist(),
                required_translation_m=(goal-center).tolist(),results=results,wall_seconds=time.monotonic()-started,
                dynamics_run=False,native_verified=False,whole_arm_collision_checked=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--target',choices=['neighbor_5','neighbor_1'],default='neighbor_5');args=parser.parse_args()
    result=check(args.target);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
