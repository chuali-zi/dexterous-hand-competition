"""Read-only opposed two-hand finger contact candidates at the real third brick.

Preserve the retained six-brick scene. Endpoint IK and contact geometry alone
do not prove a continuous approach, force closure, acquisition or native motion.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
import try_third_yellow_feed as feed

g=feed.g


def check(source,clear_unused=False,right_outer_fingers=False,thumb_pitch=(.25,.25)):
    if not np.isfinite(thumb_pitch).all() or len(thumb_pitch)!=2 or any(not 0<=v<=.5 for v in thumb_pitch):raise ValueError('Two thumb pitch targets must be between zero and 0.5 rad')
    feed.configure(source);g.restore();started=time.monotonic()
    center=g.d.xpos[g.m.body(feed.TARGET).id].copy()
    cube=Rotation.from_quat(g.d.xquat[g.m.body(feed.TARGET).id][[1,2,3,0]])
    corners=cube.apply(g.p.CORNERS)+center
    axis=cube.as_matrix()[:,int(np.argmax(np.abs(cube.as_matrix()[0,:])))];yaw=float(np.degrees(np.arctan2(axis[1],axis[0])))
    yaw=(yaw+45)%90-45
    candidates=[]
    for angle in [yaw,yaw+90,yaw-90,yaw+180]:
        g.restore();normal=np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle)),0.]);tangent=np.array([-normal[1],normal[0],0.])
        hands={}
        for side,prefix,address,sign in [('left','lh',g.p.t.LA,1),('right','rh',g.p.t.RA,-1)]:
            fingers=('ring','pinky') if right_outer_fingers and side=='right' else ('index','middle')
            targets={name.replace('lh_',prefix+'_'):value for name,value in g.p.t.CANDIDATE['hand_targets'].items()}
            if clear_unused:
                for finger in ('ring','pinky'):
                    for suffix in ('_mcp_pitch','_dip'):targets[prefix+'_'+finger+suffix]=0.
                targets.update({prefix+'_thumb_cmc_yaw':1.2,prefix+'_thumb_cmc_pitch':.5,prefix+'_thumb_ip':1.})
            if right_outer_fingers:
                for name in targets:targets[name]=0.
                targets[prefix+'_thumb_cmc_pitch']=thumb_pitch[0 if side=='left' else 1]
                targets[prefix+'_thumb_ip']=.5
            for name,value in targets.items():g.d.qpos[g.m.jnt_qposadr[g.m.joint(name).id]]=value
            quat=(Rotation.from_euler('z',angle+(180 if side=='right' else 0),degrees=True)*Rotation.from_quat(g.p.t.QUAT[[1,2,3,0]])).as_quat()[[3,0,1,2]]
            g.d.qpos[address:address+3]=0;g.d.qpos[address+3:address+7]=quat;g.mj.mj_forward(g.m,g.d)
            if clear_unused:
                toward=sign*normal;reference=min(np.min(g.a.g.mesh_vertices(prefix+'_'+finger+'_distal')@toward) for finger in fingers)
                for finger in fingers:
                    name=prefix+'_'+finger+'_mcp_pitch';adr=g.m.jnt_qposadr[g.m.joint(name).id]
                    def error(value):
                        g.d.qpos[adr]=value;g.mj.mj_forward(g.m,g.d)
                        return float(np.min(g.a.g.mesh_vertices(prefix+'_'+finger+'_distal')@toward)-reference)
                    if error(0.)*error(.08)<0:targets[name]=float(brentq(error,0.,.08))
                    g.d.qpos[adr]=targets[name];g.mj.mj_forward(g.m,g.d)
            vertices=np.concatenate([g.a.g.mesh_vertices(prefix+'_'+finger+suffix) for finger in fingers for suffix in ('_proximal','_distal')])
            toward=sign*normal
            root=toward*(np.max(corners@toward)-np.min(vertices@toward)-.0005)
            root+=tangent*(center@tangent-(np.min(vertices@tangent)+np.max(vertices@tangent))/2)
            root[2]=g.a.g.FLOOR+.010-vertices[:,2].min()
            hands[side]=dict(root_position_m=root.tolist(),root_quaternion_wxyz=quat.tolist(),hand_targets=targets,contact_fingers=list(fingers))
        g.restore()
        for side,address in [('left',g.p.t.LA),('right',g.p.t.RA)]:
            h=hands[side];g.d.qpos[address:address+3]=h['root_position_m'];g.d.qpos[address+3:address+7]=h['root_quaternion_wxyz']
            for name,value in h['hand_targets'].items():g.d.qpos[g.m.jnt_qposadr[g.m.joint(name).id]]=value
        g.mj.mj_forward(g.m,g.d);contacts=[];bad=[]
        for c in g.d.contact:
            names=[g.m.geom(int(i)).name for i in c.geom]
            if any(n.startswith(('lh_','rh_')) for n in names):
                item=dict(pair=names,depth_mm=-float(c.dist)*1000,position_m=c.pos.tolist(),normal=c.frame[:3].tolist());contacts.append(item)
                if feed.TARGET not in names:bad.append(item)
        for side,h in hands.items():h['endpoint']=g.a.g.endpoint(side,np.array(h['root_position_m']),np.array(h['root_quaternion_wxyz']))
        bilateral=all(any(feed.TARGET in c['pair'] and any(n.startswith(prefix) for n in c['pair']) for c in contacts) for prefix in ['lh_','rh_'])
        candidates.append(dict(angle_degrees=angle,hands=hands,contacts=contacts,non_target_contacts=bad,bilateral_contact=bilateral,
            endpoint_candidate=bool(not bad and bilateral and all(h['endpoint']['accepted'] for h in hands.values()))))
    g.restore()
    return dict(source=str(source),target=feed.TARGET,target_center_m=center.tolist(),clear_unused=clear_unused,right_outer_fingers=right_outer_fingers,thumb_pitch=list(thumb_pitch),candidates=candidates,wall_seconds=time.monotonic()-started,
        dynamics_run=False,continuous_approach_checked=False,native_verified=False,whole_arm_collision_checked=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=Path('runs/third-pickup-111/right-return'));parser.add_argument('--out',type=Path,required=True);parser.add_argument('--clear-unused',action='store_true');parser.add_argument('--right-outer-fingers',action='store_true');parser.add_argument('--thumb-pitch',type=float,nargs=2,default=[.25,.25]);args=parser.parse_args()
    result=check(args.source,args.clear_unused,args.right_outer_fingers,args.thumb_pitch);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(result,stream,allow_nan=False)
    print(json.dumps(dict(wall_seconds=result['wall_seconds'],candidates=[dict(angle=c['angle_degrees'],bilateral_contact=c['bilateral_contact'],non_target_contacts=len(c['non_target_contacts']),endpoints={s:h['endpoint']['accepted'] for s,h in c['hands'].items()},candidate=c['endpoint_candidate']) for c in result['candidates']]),indent=2))
