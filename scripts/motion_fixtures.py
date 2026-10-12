"""Evaluator-only motion generators and full-window truth checks.

Truth stays in this offline fixture/evaluator module. It never enters observe()
packets and is read for evaluation only after the controller has ended.
"""
import math
from hand_competition.closed_loop.synthetic import SyntheticBackend
from hand_competition.closed_loop.geometry import distance


class AlternatingPlacement(SyntheticBackend):
    amplitude=.020
    def _advance(self):
        super()._advance()
        if self._released_point is not None and not self._held and self._tip[2]>=.199 and self._pending is None:
            sign=1 if self._sequence%2 else -1
            self._block=(self._released_point[0]+sign*self.amplitude,*self._released_point[1:])


class Alternating16(AlternatingPlacement):
    amplitude=.016


class PeriodicPlacement(SyntheticBackend):
    amplitude=.020
    period_s=1.6
    def _advance(self):
        super()._advance()
        if self._released_point is not None and not self._held and self._tip[2]>=.199 and self._pending is None:
            phase=2*math.pi*(self._time-self._released_at)/self.period_s
            self._block=(self._released_point[0]+self.amplitude*math.sin(phase),*self._released_point[1:])


class SlowDriftPlacement(SyntheticBackend):
    """6mm/s monotone drift, 18mm over every three-second interval."""
    amplitude=.020
    speed_m_s=.006
    def _advance(self):
        super()._advance()
        if self._released_point is not None and not self._held and self._tip[2]>=.199 and self._pending is None:
            offset=self.speed_m_s*(self._time-self._released_at)
            self._block=(self._released_point[0]+offset,*self._released_point[1:])


class TruthHistory:
    """Wrap a fixture so final post-dynamics truth is recorded at each tick."""
    def __init__(self,*args,**kwargs):
        self.evaluation_history=[]
        super().__init__(*args,**kwargs)
    def _advance(self):
        super()._advance()
        self.evaluation_history.append({'time_s':self._time,'position_m':self._block,
          'tip_m':self._tip,'held':self._held,'released_at_s':self._released_at})


def evaluated_backend(cls,*args,**kwargs):
    wrapped=type('Evaluated'+cls.__name__,(TruthHistory,cls),{})
    return wrapped(*args,**kwargs)


def trajectory_diameter(points):
    return max((distance(a,b) for i,a in enumerate(points) for b in points[:i]),default=0.)


def evaluate_window(history,result,events,config):
    """Independent sampled-truth audit of the *entire* claimed stable window.

    Checks generator ticks, not unobserved continuous physical motion. No window
    means no successful certificate, so false_accept is false, not a proof that
    the object is physically stable. Save truth separately for reproducibility.
    """
    stable=[e for e in events if e['kind']=='stability_verified']
    withdrawn=[h for h in history if h['released_at_s'] is not None and not h['held']
               and distance(h['position_m'],h['tip_m'])>=config.hand_separation_m]
    report={'scope':'evaluator_only_synthetic_sampled_truth','physical_contact_verified':False,
      'post_release_withdrawn_samples':len(withdrawn),
      'post_release_withdrawn_diameter_m':trajectory_diameter([h['position_m'] for h in withdrawn]),
      'successful_window_checked':False,'false_accept_under_evaluator_checks':False,
      'window':None,'violation_categories':[]}
    if not result['success']:return report
    violations=[]
    if len(stable)!=1:violations.append('missing_or_multiple_success_windows')
    if stable:
        p=stable[-1]['payload'];start=p['sensor_start_s'];end=p['sensor_end_s']
        samples=[h for h in history if start-1e-6<=h['time_s']<=end+1e-6]
        points=[h['position_m'] for h in samples]
        complete=bool(samples) and abs(samples[0]['time_s']-start)<=1e-6 and abs(samples[-1]['time_s']-end)<=1e-6
        gaps=[b['time_s']-a['time_s'] for a,b in zip(samples,samples[1:])]
        if not complete or (gaps and max(gaps)>config.max_observation_gap_s+1e-9):violations.append('truth_window_coverage')
        if end-start<config.settle_s-1e-9:violations.append('duration')
        diameter=trajectory_diameter(points)
        if diameter>config.stability_m+1e-9:violations.append('whole_window_motion')
        max_goal=max((distance(v,config.target) for v in points),default=None)
        if max_goal is None or max_goal>config.goal_tolerance_m+1e-9:violations.append('whole_window_goal')
        max_height=max((abs(v[2]-config.table_center_height_m) for v in points),default=None)
        if max_height is None or max_height>config.table_height_tolerance_m+1e-9:violations.append('whole_window_height')
        min_clearance=min((distance(h['position_m'],h['tip_m']) for h in samples),default=None)
        if min_clearance is None or min_clearance<config.hand_separation_m-1e-9:violations.append('whole_window_clearance')
        if any(h['held'] for h in samples):violations.append('still_held')
        if any(not(config.build_xy[0]<=v[0]<=config.build_xy[1] and config.build_xy[2]<=v[1]<=config.build_xy[3]) for v in points):
            violations.append('whole_window_build_region')
        report['window']={'sensor_start_s':start,'sensor_end_s':end,'samples':len(samples),
          'coverage_complete':complete,'diameter_m':diameter,'max_goal_error_m':max_goal,
          'max_height_error_m':max_height,'min_hand_clearance_m':min_clearance,
          'max_sample_gap_s':max(gaps,default=0.)}
        report['successful_window_checked']=True
    report['violation_categories']=violations
    report['false_accept_under_evaluator_checks']=bool(violations)
    return report
