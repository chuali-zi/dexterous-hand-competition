"""Compare local payload detachment by tipping or folding support fingers.

Starts from the retained loaded transfer and reuses the verified flattening
motion. The brick is never repositioned or attached to the hand. Intended
release must still pass the existing two-base support and alignment checks.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import try_rotated_yellow_release as r
g=r.g


def predict_aim(source):
    record=json.loads((Path(source)/'result.json').read_text());check=json.loads((Path(source)/'precheck.json').read_text())
    if not check['passed'] or record['summary']['target']!=g.TARGET or record['summary']['abort']!='payload_below_base_top':
        raise ValueError('Aim requires a measured missed release at base height')
    free=[row for row in record['rows'] if not row['contacts']]
    if len(free)<2:raise ValueError('No measured free-flight interval')
    previous,last=free[-2:];dt=last['elapsed_s']-previous['elapsed_s']
    if not 0<dt<=.021:raise ValueError('Free-flight samples are not adjacent')
    p0=np.array(last['position_m']);gravity=np.asarray(g.m.opt.gravity)
    velocity=(p0-previous['position_m'])/dt+.5*gravity*dt
    rotations=Rotation.from_quat(np.array([previous['quaternion_wxyz'],last['quaternion_wxyz']])[:,[1,2,3,0]])
    angular=(rotations[0].inv()*rotations[1]).as_rotvec()/dt
    g.restore();pair=np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS],axis=0)
    top=max((Rotation.from_quat(g.d.xquat[g.m.body(n).id][[1,2,3,0]]).apply(g.p.CORNERS)+g.d.xpos[g.m.body(n).id])[:,2].max() for n in g.SUPPORTS)
    def position(t):return p0+velocity*t+.5*gravity*t*t
    def bottom(t):return ((rotations[1]*Rotation.from_rotvec(angular*t)).apply(g.p.CORNERS)+position(t))[:,2].min()
    times=np.linspace(0,.1,201);cross=next((i for i in range(1,len(times)) if bottom(times[i])<=top<bottom(times[i-1])),None)
    if cross is None:raise ValueError('No near-term downward crossing of base height')
    low,high=times[cross-1],times[cross]
    for _ in range(35):
        mid=(low+high)/2
        if bottom(mid)>top:low=mid
        else:high=mid
    impact=position(high);offset=np.r_[pair[:2]-impact[:2],0.]
    if np.linalg.norm(offset)>.060:raise ValueError('Predicted aim exceeds this local comparison range')
    return offset,dict(source=str(source),free_sample_times_s=[previous['elapsed_s'],last['elapsed_s']],
                       predicted_flight_s=float(high),predicted_center_m=impact.tolist(),offset_m=offset.tolist(),
                       assumption='Free-flight extrapolation before first base-height crossing; landing dynamics unverified')


def prepare(mode='tip',axis_degrees=135.,tip_degrees=60.,extra_clearance_m=0.,keep_fingers=False,aim_from=None,landing_correction_from=None):
    if mode not in ['tip','fold'] or not np.isfinite([axis_degrees,tip_degrees,extra_clearance_m]).all() or not 0<tip_degrees<=90 or not 0<=extra_clearance_m<=.030:
        raise ValueError('Invalid local detachment comparison')
    if keep_fingers and mode!='tip':raise ValueError('Keeping fingers applies only to tipping')
    if landing_correction_from is not None and aim_from is None:raise ValueError('Landing correction requires its measured free-flight aim')
    original,base_plan=r.prepare()
    def before(t):
        poses,ctrl,phase=original(t);pos,quat=poses['right']
        poses['right']=(pos+np.array([0,0,extra_clearance_m])*np.clip((t-4)/4,0,1),quat)
        return poses,ctrl,phase
    poses,controls,_=before(8)
    center=np.array(base_plan['initial_target_center_m'])-[0,0,base_plan['lowering_m']-extra_clearance_m]
    pos,quat=poses['right'];wrist=Rotation.from_quat(quat[[1,2,3,0]])
    axis=np.array([np.cos(np.radians(axis_degrees)),np.sin(np.radians(axis_degrees)),0.])
    final_turn=Rotation.from_rotvec(axis*np.radians(tip_degrees)) if mode=='tip' else Rotation.identity()
    final_pos=final_turn.apply(pos-center)+center
    away=final_pos-center;away[2]=0;away=away/np.linalg.norm(away)*.100
    g.DURATION=21.
    def command(t):
        if t<8:return before(t)
        turn=Rotation.from_rotvec(axis*np.radians(tip_degrees)*np.clip((t-8)/4,0,1)) if mode=='tip' else Rotation.identity()
        right=turn.apply(pos-center)+center+away*np.clip((t-15)/3,0,1)
        q=(turn*wrist).as_quat()[[3,0,1,2]];ctrl=controls.copy()
        for finger in ['index','middle','ring','pinky']:
            start=10. if mode=='tip' or finger in ['ring','pinky'] else 8.
            goal=0. if mode=='tip' else 1.45
            for suffix in ['_mcp_pitch','_dip']:
                name='rh_'+finger+suffix;ctrl[name]=controls[name] if keep_fingers else controls[name]+(goal-controls[name])*np.clip((t-start)/2,0,1)
        phase=('tip_support' if mode=='tip' else 'fold_support_fingers') if t<12 else 'observe_detachment' if t<15 else 'withdraw_candidate' if t<18 else 'released_hold'
        return dict(left=poses['left'],right=(right,q)),ctrl,phase
    plan={**base_plan,'mode':mode,'tip_axis_degrees':axis_degrees if mode=='tip' else None,'tip_degrees':tip_degrees if mode=='tip' else 0,
          'extra_clearance_m':extra_clearance_m,'effective_lowering_m':base_plan['lowering_m']-extra_clearance_m,'keep_fingers':keep_fingers,
          'duration_s':g.DURATION,'hold_start_s':18.,'detachment_observation_s':[12,15],'withdrawal_m':away.tolist()}
    if aim_from is not None:
        if mode!='tip' or not keep_fingers:raise ValueError('Measured aim comparison preserves tipping and fixed fingers')
        prior=json.loads((Path(aim_from)/'plan.json').read_text())
        if (prior.get('mode')!='tip' or not prior.get('keep_fingers') or prior.get('aim_calibration')
                or Path(prior['source']).resolve()!=g.SOURCE.resolve()
                or prior['tip_axis_degrees']!=axis_degrees or prior['tip_degrees']!=tip_degrees or prior['extra_clearance_m']!=extra_clearance_m):
            raise ValueError('Aim source must use the same unshifted loaded state and tipping command')
        offset,calibration=predict_aim(aim_from);original_command=command
        if landing_correction_from is not None:
            path=Path(landing_correction_from);landed=json.loads((path/'result.json').read_text())['summary'];prior_plan=json.loads((path/'plan.json').read_text())
            if (not landed['local_released_bridge_verified'] or landed['target']!=g.TARGET or landed['abort']
                    or landed['held_duration_s']<2.98 or not landed['held_only_base_contacts'] or min(landed['support_minimum_up_force_n'].values())<.1
                    or max(landed['other_bricks_max_shift_mm'].values())>2 or landed['checkpoint_next_step_maximum_difference']>1e-10
                    or prior_plan['aim_calibration']!=calibration or prior_plan.get('landing_correction')
                    or any(prior_plan[k]!=plan[k] for k in ['source','mode','tip_axis_degrees','tip_degrees','extra_clearance_m','keep_fingers'])):
                raise ValueError('Landing correction requires the same actual stable bridge trial')
            correction=-np.r_[landed['final_center_offset_mm'],0.]/1000
            if np.linalg.norm(correction)>.010:raise ValueError('Landing correction exceeds the 10 mm local comparison')
            offset=offset+correction
            plan['landing_correction']=dict(source=str(path),offset_m=correction.tolist(),combined_aim_m=offset.tolist())
        def aimed_command(t):
            poses,ctrl,phase=original_command(max(0.,t-3))
            pos,quat=poses['right'];poses['right']=(pos+offset*np.clip(t/3,0,1),quat)
            return poses,ctrl,'translate_loaded_aim' if t<3 else phase
        command=aimed_command;g.DURATION+=3
        plan.update(aim_calibration=calibration,duration_s=g.DURATION,hold_start_s=21.,detachment_observation_s=[15,18])
    return command,plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/upper-place-100/loaded-transfer'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--mode',choices=['tip','fold'],default='tip');parser.add_argument('--axis-degrees',type=float,default=135.)
    parser.add_argument('--tip-degrees',type=float,default=60.)
    parser.add_argument('--extra-clearance-mm',type=float,default=0.)
    parser.add_argument('--keep-fingers',action='store_true',help='Isolate wrist tipping while preserving the initial finger commands')
    parser.add_argument('--aim-from',type=Path,help='Use a previous measured free-flight interval for one offline aim comparison')
    parser.add_argument('--landing-correction-from',type=Path,help='Compare one small correction measured from an actual stable two-base release')
    args=parser.parse_args();g.configure_continuation(args.source,'neighbor_5');command,plan=prepare(args.mode,args.axis_degrees,args.tip_degrees,args.extra_clearance_mm/1000,args.keep_fingers,args.aim_from,args.landing_correction_from)
    args.out.mkdir(parents=True,exist_ok=False);checked=g.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    brief=dict(mode=args.mode,precheck_passed=checked['passed'],geometry_failures=len(checked['geometry_failures']),
               stages={side:len(value) for side,value in checked['stages'].items()},precheck_wall_seconds=checked['wall_seconds'])
    passed=checked['passed']
    if passed and args.run:brief['dynamics']=g.run(command,plan,args.out);passed=brief['dynamics']['local_first_yellow_aligned']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
