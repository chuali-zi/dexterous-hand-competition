"""Independent legal-path reconstruction from decoded RGB/action records.

No controller implementation or success labels are used as transition proofs.
A failed record may be structurally consistent yet lack reconstructible cause.
"""
from .geometry import distance, sub, add
from .contracts import placement_valid, finite
from . import bundle_replay


class PathVerifier:
    def __init__(self, config, robust, cameras):
        self.config=config; self.robust=robust; self.cameras=cameras
        self.state='before_start'; self.active=None; self.attempt=0; self.cycle=None
        self.source=None; self.close=None; self.lift=None; self.transit=None; self.descent=None
        self.release=None; self.retreat=None; self.anchor=None; self.anchor_start=None; self.held_color=None; self.held_offset=None
        self.failure=None; self.recovery=None; self.recoveries=0; self.blocked=None; self.block_status_seen=False
        self.terminal=None; self.terminal_payload=None; self.terminal_state=None; self.success_before_stop=False
        self.last_sample=None; self.last_bundle=None; self.stability_points=[]

    def evidence(self, samples, bundles):
        if self.robust:
            b=next(reversed(bundles.values()),None) if bundles else None
            return b
        if not samples or samples[-1]['point'] is None: return None
        s=samples[-1]
        return {'position_m':s['point'],'tip':s['tip'],'color':s['color'],'radius_m':0,
                'uncertainty_xyz_m':(0,0,0),'sensor_end_s':s['sensor_time_s'],
                'sample_indices':(len(samples)-1,)}

    def is_held(self, e):
        return e is not None and (bundle_replay.held(e,self.config) if self.robust else
                                 distance(e['position_m'],e['tip']) <= self.config.grasp_distance_m)

    def on_table(self,e):
        return e is not None and abs(e['position_m'][2]-self.config.table_center_height_m)+e['uncertainty_xyz_m'][2] <= self.config.table_height_tolerance_m

    def lift_proof(self,e):
        if not self.close or not self.lift or not e: return False
        pre=self.close['evidence']
        return (pre is not None and self.on_table(pre) and pre['color']==e['color'] and self.is_held(e)
                and e['position_m'][2]-pre['position_m'][2]-e['uncertainty_xyz_m'][2]-pre['uncertainty_xyz_m'][2] >= self.config.lift_min_m
                and e['sample_indices'][-1] >= self.lift['after']['index'])

    def endpoint_held(self,e,record):
        return (record is not None and self.is_held(e) and e['color']==self.held_color
                and (not self.robust or e['sample_indices'][0] > record['after']['index']))

    def accept(self,event,samples,bundles,actions):
        kind=event['kind'];p=event['payload'];stamp=event['time_s'];e=self.evidence(samples,bundles)
        frame_stability=kind in ('settle_started','stability_verified') and p.get('stability_basis')=='raw_rgb_per_frame'
        if frame_stability:
            sample=samples[-1] if samples else None
            e=None if not sample or sample['point'] is None else {'position_m':sample['point'],'tip':sample['tip'],'color':sample['color'],
                 'sensor_end_s':sample['sensor_time_s'],'radius_m':0,'uncertainty_xyz_m':(0,0,0)}
        if self.state=='stopped' and kind!='result': raise ValueError('events after irreversible stop')
        if self.state=='terminated' and kind not in ('backend_stopped','stop_request_succeeded','stop_request_failed'):
            raise ValueError('events after irreversible trial termination')
        if self.state=='stable' and kind not in ('backend_stopped','stop_request_succeeded','stop_request_failed'):
            raise ValueError('events/actions after stability terminal')
        if self.blocked and kind not in ('action_status','trial_stopped','backend_stopped','stop_request_succeeded','stop_request_failed','result'):
            raise ValueError('continued control after rejected carrying/sensor evidence')
        if kind=='trial_started':
            if self.state!='before_start': raise ValueError('duplicate/out-of-order trial start')
            self.state='initial'
        elif self.state=='before_start': raise ValueError('event before trial start')
        elif kind=='observation':
            s=samples[-1]
            if self.active and self.active['phase'] in ('target_transit','target_descent'):
                bad=None
                if self.robust:
                    if s['pixels'] is None: bad='carried_object_unobserved'
                    elif s['color']!=self.held_color or self.held_offset is None: bad='carried_object_lost'
                    else:
                        predicted=add(s['tip'],self.held_offset)
                        for name,pixel in zip(s['camera_ids'],s['pixels']):
                            expected=next(c for c in self.cameras if c.name==name).project(predicted)
                            if sum((a-b)**2 for a,b in zip(pixel,expected))**.5 > self.config.robust_tracking_pixel_px: bad='carried_object_lost'
                elif s['point'] is None: bad='carried_object_unobserved'
                elif distance(s['point'],s['tip'])>self.config.grasp_distance_m: bad='carried_object_lost'
                if self.last_sample and s['sensor_time_s']-self.last_sample['sensor_time_s']>self.config.max_observation_gap_s+1e-9:
                    bad='carried_object_unobserved'
                if bad: self.blocked=bad
            if self.state=='settling' and s['point'] is not None: self.stability_points.append(s['point'])
            self.last_sample=s
        elif kind=='robust_observation': self.last_bundle=e
        elif kind=='bundle_reset':
            if self.state=='settling': self.anchor_start=None
        elif kind=='observation_rejected': self.blocked='invalid_observation'
        elif kind=='action_submitted':
            phase=p['phase'];record=actions[p['action_id']]
            if self.active is not None: raise ValueError('overlapping path actions')
            allowed={'raise_for_approach':('initial','retry_source'),'above_grasp':('raised',),
                     'grasp_descent':('above',),'close':('grasp_ready',),'lift':('closed',),
                     'target_transit':('lift_done','withdrawn'),'target_descent':('at_target_high',),
                     'release':('at_target_low',),'retreat':('released',),'reset_empty_grasp':('lift_done',)}
            if self.state not in allowed[phase]: raise ValueError('illegal control phase transition: '+self.state+' -> '+phase)
            if phase=='raise_for_approach':
                if e is None: raise ValueError('approach lacks visual source')
                self.attempt+=1;self.source=e;self.cycle=None;self.close=None;self.lift=None
                self.transit=None;self.descent=None;self.release=None;self.retreat=None;self.anchor=None;self.anchor_start=None;self.recovery=None;self.stability_points=[]
            elif phase=='close':
                if not self.is_held(e): raise ValueError('close lacks fresh visual proximity')
            elif phase=='target_transit':
                if self.state=='lift_done':
                    if not self.lift_proof(e): raise ValueError('transit lacks same-attempt close/lift proof')
                elif self.recovery!='release_not_verified' or not self.is_held(e):
                    raise ValueError('re-placement lacks release-stuck recovery proof')
                self.recovery=None;self.held_color=e['color'];self.held_offset=sub(e['position_m'],e['tip'])
            elif phase=='target_descent':
                if not self.endpoint_held(e,self.transit): raise ValueError('descent lacks same-cycle measured transit and held proof')
                self.held_offset=sub(e['position_m'],e['tip'])
            elif phase=='release':
                if not self.endpoint_held(e,self.descent): raise ValueError('release lacks same-cycle descent and held proof')
                if distance(samples[-1]['tip'],self.config.target)>self.config.motion_completion_m:
                    raise ValueError('normal release away from measured target')
            elif phase=='reset_empty_grasp':
                if self.recovery!='grasp_not_verified' or not self.on_table(e) or self.lift_proof(e):
                    raise ValueError('empty reset lacks failed table-grasp recovery evidence')
            if record['action']['kind']=='move':
                target=record['action']['target_m'];tip=samples[-1]['tip']
                if phase=='raise_for_approach': expected=(*tip[:2],self.config.safe_height_m)
                elif phase in ('target_transit','retreat'): expected=(*self.config.target[:2],self.config.safe_height_m)
                elif phase=='target_descent': expected=self.config.target
                elif phase=='grasp_descent': expected=(*self.above_target[:2],self.config.table_center_height_m)
                elif phase=='lift': expected=(*self.grasp_target[:2],self.config.safe_height_m)
                else: expected=(*target[:2],self.config.safe_height_m)
                if distance(target,expected)>1e-9: raise ValueError('phase waypoint does not match control path')
            record['evidence']=e;record['attempt']=self.attempt;self.active=record
        elif kind=='action_status':
            record=actions[p['action_id']]
            if self.active is not record: raise ValueError('status outside current path action')
            if self.blocked:
                if self.block_status_seen: raise ValueError('continued status polling after lost carry evidence')
                self.block_status_seen=True
            if p['state']=='complete':
                phase=record['phase']
                self.state={'raise_for_approach':'raised','above_grasp':'above','grasp_descent':'grasp_ready','close':'closed',
                            'lift':'lift_done','target_transit':'at_target_high','target_descent':'at_target_low',
                            'release':'released','retreat':'withdrawn','reset_empty_grasp':'retry_source'}[phase]
                if phase=='above_grasp': self.above_target=record['action']['target_m']
                elif phase=='grasp_descent': self.grasp_target=record['action']['target_m']
                elif phase=='close': self.close=record
                elif phase=='lift': self.lift=record
                elif phase=='target_transit': self.transit=record
                elif phase=='target_descent': self.descent=record
                elif phase=='release': self.release=record;self.cycle=self.attempt;self.stability_points=[]
                elif phase=='retreat': self.retreat=record
                elif phase=='reset_empty_grasp': self.recovery=None
                self.active=None
                if not record.get('motion_verified',True): self.blocked='motion_not_verified'
            elif p['state']=='failed': self.blocked='actuator_failure';self.active=None
        elif kind=='failure_detected':
            category=p['category'];confirmed=False
            if category=='target_moved' and self.state=='above' and e and self.source:
                confirmed=distance(e['position_m'],self.source['position_m'])-e['radius_m']-self.source['radius_m'] > self.config.motion_completion_m
            elif category=='grasp_not_verified' and self.state=='lift_done' and e: confirmed=not self.lift_proof(e)
            elif category=='release_not_verified' and self.state=='withdrawn' and e:
                confirmed=distance(e['position_m'],e['tip'])+e['radius_m'] <= self.config.grasp_distance_m
            elif category=='placement_drift' and self.state in ('withdrawn','settling') and e:
                if self.robust: confirmed=distance(e['position_m'],self.config.target)-e['radius_m']-self.config.robust_absolute_floor_m > self.config.goal_tolerance_m
                else: confirmed=not placement_valid(e['position_m'],e['tip'],self.config) or any(distance(e['position_m'],point)>self.config.stability_m for point in self.stability_points)
            self.failure=(category,confirmed)
        elif kind=='recovery_started':
            if (not self.config.recovery_enabled or self.recovery is not None or self.failure!=(p['category'],True)
                    or type(p['recovery_number']) is not int or p['recovery_number']!=self.recoveries+1):
                raise ValueError('recovery not justified by current sensor/path evidence')
            self.recoveries+=1;self.recovery=p['category'];self.failure=None
            if self.recovery in ('target_moved','placement_drift'): self.state='retry_source';self.cycle=None
        elif kind=='settle_started':
            if (self.state not in ('withdrawn','settling') or self.active is not None or self.release is None or self.retreat is None
                    or self.cycle!=self.attempt or self.release['attempt']!=self.retreat['attempt']):
                raise ValueError('stability start lacks same-attempt release/retreat path')
            if e is None or p.get('sensor_start_s',stamp)<self.retreat['after']['sensor_time_s']-1e-9:
                raise ValueError('stable captures precede measured withdrawal')
            start=p.get('sensor_start_s',stamp)
            expected_start=e['sensor_start_s'] if self.robust and not frame_stability else e['sensor_end_s']
            if abs(start-expected_start)>1e-9 or distance(p['anchor_m'],e['position_m'])>1e-9:
                raise ValueError('settle anchor/time contradicts latest primary evidence')
            valid=bundle_replay.placement(e,self.config) if self.robust and not frame_stability else placement_valid(e['position_m'],e['tip'],self.config)
            if not valid: raise ValueError('settle anchor is not a verified supported placement')
            self.anchor=e;self.anchor_start=start;self.stability_points=[e['position_m']];self.state='settling'
        elif kind=='settle_reset':
            if self.state not in ('withdrawn','settling'): raise ValueError('settle reset outside placed state')
            # Keep the anchor for a following factual placement_drift check.
            self.state='withdrawn';self.anchor_start=None
        elif kind=='stability_verified':
            if self.state!='settling' or self.active is not None or self.cycle!=self.attempt:
                raise ValueError('stability has no legal current placement cycle')
            end=p.get('sensor_end_s',stamp);start=p.get('sensor_start_s',end-p['duration_s'])
            if self.anchor_start is None or abs(start-self.anchor_start)>1e-9:
                raise ValueError('stability interval is not tied to latest unrestarted window')
            if e is None or abs(end-e['sensor_end_s'])>1e-9 or distance(p['position_m'],e['position_m'])>1e-9:
                raise ValueError('stability end/position contradicts latest primary evidence')
            self.state='stable'
        elif kind=='trial_stopped':
            self.terminal_state=self.state; self.terminal=p['category'];self.terminal_payload=p;self.state='terminated'
        elif kind in ('backend_stopped','stop_request_succeeded','stop_request_failed'):
            if self.state not in ('stable','terminated'): raise ValueError('stop without preceding success/failure terminal')
            self.success_before_stop=self.state=='stable';self.state='stopped';self.active=None
        elif kind=='result':
            if self.state!='stopped': raise ValueError('result before stop')
            result=p['result']
            if result['success'] and (not self.success_before_stop or self.terminal is not None):
                raise ValueError('success after failed/nonstable path')
            if result['recoveries']!=self.recoveries: raise ValueError('recovery count differs from path')

    def failure_cause(self,result,actions,rejected,trial_start,version):
        if result['success']: return True,[]
        category=result['terminal_failure'];payload=self.terminal_payload or {}
        proof=False
        if category=='stop_transport_failure': proof=True # separately checked stop failure event
        elif category=='invalid_observation': proof=rejected>0
        elif category=='actuator_failure': proof=any(r['state']=='failed' for r in actions.values())
        elif category=='motion_not_verified': proof=any(r.get('motion_verified') is False for r in actions.values())
        elif category in ('carried_object_lost','carried_object_unobserved'): proof=self.blocked==category
        elif category in ('trial_timeout','action_timeout'):
            d=payload.get('deadline')
            scope='trial' if category=='trial_timeout' else 'action'
            if d is not None:
                if not isinstance(d,dict) or set(d)!={'scope','backend_elapsed_s','wall_elapsed_s','action_id'}:
                    raise ValueError('invalid deadline evidence fields')
                if d['scope']!=scope or not finite(d['backend_elapsed_s']) or not finite(d['wall_elapsed_s']) or min(d['backend_elapsed_s'],d['wall_elapsed_s'])<0:
                    raise ValueError('invalid deadline values')
                if scope=='trial':
                    budget=self.config.trial_timeout_s;wall_budget=self.config.wall_timeout_s
                    measured=payload['_time_s']-trial_start
                    if d['action_id'] is not None: raise ValueError('trial deadline names an action')
                else:
                    record=actions.get(d['action_id'])
                    if record is None or record['state'] not in ('pending','stopped'): raise ValueError('deadline has no active unfinished action request')
                    budget=wall_budget=self.config.action_timeout_s
                    measured=payload['_time_s']-record['start']
                if abs(measured-d['backend_elapsed_s'])>1e-6 or not (d['backend_elapsed_s']>=budget or d['wall_elapsed_s']>=wall_budget):
                    raise ValueError('deadline evidence does not exceed matching budget')
                proof=True
            elif category=='trial_timeout': proof=bool(self.last_sample or rejected) and result['elapsed_s']>=self.config.trial_timeout_s
            else: proof=any(payload['_time_s']-r['start']>=self.config.action_timeout_s for r in actions.values())
        elif category=='grasp_target_unverified':
            proof=self.terminal_state=='grasp_ready' and self.last_sample is not None and (self.last_sample['point'] is None or distance(self.last_sample['point'],self.source['position_m'])>self.config.grasp_distance_m)
        elif category=='uncertain_grasp_state':
            proof=self.terminal_state=='lift_done' and self.failure==('grasp_not_verified',True) and not self.on_table(self.evidence([self.last_sample],{}) if not self.robust else self.last_bundle)
        elif self.failure==(category,True): proof=True
        return proof,[] if proof else ['terminal reason is declared but not reconstructible from saved sensor/action/deadline facts']
