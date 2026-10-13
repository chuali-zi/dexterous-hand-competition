"""Read-only observation/planning lifecycle; no SDK execution bridge exists."""
from dataclasses import asdict
from ..closed_loop.contracts import ObservationGate
from ..closed_loop.geometry import finite
from .perception import VisionConfig,observe_many
from .tracking import Tracker
from .planning import PlanConfig,plan


class PlanningSession:
    def __init__(self,cameras,producer,config=PlanConfig(),vision=VisionConfig()):
        if any(vision.workspace[2*i]<config.workspace[2*i] or vision.workspace[2*i+1]>config.workspace[2*i+1] for i in range(3)):
            raise ValueError('vision workspace must stay within planning workspace')
        self.config=config;self.vision=vision;self.cameras=tuple(cameras);self.producer=producer
        self.gate=ObservationGate(cameras,producer,config.max_observation_age_s,config.max_sensor_skew_s,tip_bounds=config.workspace)
        self.tracker=Tracker(config.association_radius_m,config.max_observation_gap_s,config.hand_clearance_m)
        self.started_s=None;self.last_received_s=-1.;self.terminated=False;self.observations=0
    def observe(self,raw,received_s):
        if self.terminated:raise ValueError('planning session irreversibly terminated')
        try:
            if not finite(received_s) or received_s<0 or received_s<self.last_received_s:raise ValueError('receipt clock')
            if self.started_s is None:self.started_s=received_s
            if received_s-self.started_s>=self.config.session_budget_s-1e-9:raise ValueError('planning session deadline')
            observation=self.gate.accept(raw,received_s)
            found=observe_many(observation,self.cameras,self.vision)
            tracks,issues,retired=self.tracker.update(found.blocks,observation.sensor_time_s,observation.tip)
            decision=plan(tracks,self.config)
        except Exception:
            self.terminated=True;raise
        self.observations+=1;self.last_received_s=received_s
        return observation,{'tracks':[asdict(t) for t in tracks],'perception_issues':found.issues,
            'tracking_issues':issues,'retired_local_ids':retired,'decision':decision}
    def stop(self):
        if self.terminated:raise ValueError('planning session already terminated')
        self.terminated=True
    def execute_candidate(self,*args,**kwargs):
        raise NotImplementedError('multi-block robot execution/official SDK mapping is not implemented')
