"""One-block feedback controller consuming the internal sensor/action contract."""
from dataclasses import asdict
import time
from .contracts import Action, ObservationGate, ObservationError, guard_move, inside_xy
from .geometry import distance
from .vision import estimate_single_block, VisionError


class TrialFailure(RuntimeError):
    def __init__(self, category, detail):
        super().__init__(detail); self.category = category


class Controller:
    def __init__(self, backend, config, trace):
        self.backend = backend; self.config = config; self.trace = trace
        self.gate = ObservationGate(backend.cameras, backend.producer)
        self.trace.producer = backend.producer
        self.observation = None; self.visual = None; self.failures = []
        self.recoveries = 0; self.started_in_supply = False; self.phase = 'initial'
        self.stability_start = None; self.stability_verified_s = 0.0
        self.start_s = backend.now(); self.wall_start_s = time.monotonic(); self.action_count = 0

    def deadline(self):
        if (self.backend.now()-self.start_s >= self.config.trial_timeout_s or
                time.monotonic()-self.wall_start_s >= self.config.wall_timeout_s):
            raise TrialFailure('trial_timeout', 'experiment deadline reached; no blind retries')

    def sense(self, require_visible=True):
        self.deadline()
        try: raw = self.backend.observe()
        except Exception as error:
            raise TrialFailure('sensor_transport_failure', type(error).__name__) from error
        try: self.observation = self.gate.accept(raw, self.backend.now())
        except ObservationError as error:
            # Do not log forbidden raw fields, values or potentially secret data.
            self.trace.event('observation_rejected', self.backend.now(), error=str(error))
            raise TrialFailure('invalid_observation', str(error)) from error
        try:
            self.visual = estimate_single_block(self.observation, self.backend.cameras)
            self.trace.observation(self.observation, self.visual)
            return self.visual
        except VisionError as error:
            self.visual = None
            self.trace.observation(self.observation, error=str(error))
            if require_visible: return None
            return None

    def visible(self):
        while True:
            estimate = self.sense()
            if estimate is not None: return estimate

    def command(self, phase, kind, target=None):
        self.phase = phase; self.deadline()
        if kind == 'move':
            try: guard_move(self.observation.tip, target, self.config)
            except ValueError as error: raise TrialFailure('unsafe_waypoint', str(error)) from error
        action = Action(kind, target)
        try: action_id = self.backend.submit(action)
        except Exception as error:
            raise TrialFailure('action_transport_failure', type(error).__name__) from error
        self.action_count += 1
        started = self.backend.now()
        self.trace.event('action_submitted', started, phase=phase, action_id=action_id,
                         action=asdict(action), grasp_success_inferred=False)
        while True:
            self.sense(require_visible=False)
            try: status = self.backend.status(action_id)
            except Exception as error:
                raise TrialFailure('action_transport_failure', type(error).__name__) from error
            self.trace.event('action_status', self.backend.now(), action_id=action_id,
                             state=status.state, detail=status.detail)
            if status.state == 'failed': raise TrialFailure('actuator_failure', status.detail)
            if status.state == 'complete': return
            if status.state != 'pending': raise TrialFailure('actuator_failure', 'unknown action status')
            if self.backend.now()-started >= self.config.action_timeout_s:
                raise TrialFailure('action_timeout', f'{phase}: command did not complete')

    def recover(self, category, detail):
        self.failures.append({'category': category, 'phase': self.phase, 'detail': detail})
        self.trace.event('failure_detected', self.backend.now(), category=category, phase=self.phase, detail=detail)
        if not self.config.recovery_enabled: raise TrialFailure(category, detail)
        self.recoveries += 1
        self.trace.event('recovery_started', self.backend.now(), category=category,
                         recovery_number=self.recoveries)
        # Each retry still obeys the global deadline and fresh visual observation.

    def place_held_block(self):
        self.command('target_transit', 'move', (self.config.target[0], self.config.target[1], self.config.safe_height_m))
        self.command('target_descent', 'move', self.config.target)
        self.command('release', 'release')
        self.command('retreat', 'move', (self.config.target[0], self.config.target[1], self.config.safe_height_m))

    def run(self):
        success = False; terminal = None; stop_request_succeeded = False
        try:
            first = self.visible()
            self.started_in_supply = inside_xy(first.position_m, self.config.supply_xy)
            if not self.started_in_supply:
                raise TrialFailure('invalid_initial_location', 'observed block is not in configured supply zone')
            while True:
                self.deadline(); block = self.visible().position_m
                # Raise before lateral travel; these waypoints do NOT certify a
                # real SDK's joint interpolated path or actual collision model.
                self.command('raise_for_approach', 'move', (self.observation.tip[0], self.observation.tip[1], self.config.safe_height_m))
                self.command('above_grasp', 'move', (block[0], block[1], self.config.safe_height_m))
                self.command('grasp_descent', 'move', (block[0], block[1], self.config.table_center_height_m))
                self.command('close', 'grasp')
                self.command('lift', 'move', (block[0], block[1], self.config.safe_height_m))
                lifted = self.visible().position_m
                if lifted[2]-block[2] < self.config.lift_min_m or distance(lifted, self.observation.tip) > 0.03:
                    self.recover('grasp_not_verified', 'RGB stereo did not confirm object lifting with the hand')
                    # Opening is allowed in this toy recovery only after observing
                    # the block on the table; uncertain airborne state is stopped.
                    if abs(lifted[2]-self.config.table_center_height_m) > 0.03:
                        raise TrialFailure('uncertain_grasp_state', 'cannot safely recover an unknown held state')
                    self.command('reset_empty_grasp', 'release')
                    continue
                self.trace.event('grasp_verified', self.backend.now(), position_m=lifted)
                self.place_held_block()
                while True:
                    point = self.visible().position_m
                    if point[2] > self.config.table_center_height_m+0.04:
                        self.recover('release_not_verified', 'object followed retreat; command completion was not release success')
                        self.place_held_block()
                        continue
                    if not inside_xy(point, self.config.build_xy) or distance(point, self.config.target) > self.config.goal_tolerance_m:
                        self.recover('placement_drift', 'released object moved outside experiment target criterion')
                        break  # re-observe and re-grasp; never use an old pose
                    self.phase = 'settling'; anchor = point; start = self.backend.now()
                    self.trace.event('settle_started', start, anchor_m=anchor)
                    unsettled = False
                    while self.backend.now()-start < self.config.settle_s:
                        estimate = self.sense()
                        if estimate is None:
                            self.trace.event('settle_reset', self.backend.now(), reason='visibility_lost')
                            unsettled = True; break
                        current = estimate.position_m
                        if distance(current, anchor) > self.config.stability_m or distance(current, self.config.target) > self.config.goal_tolerance_m:
                            self.trace.event('settle_reset', self.backend.now(), reason='observed_motion')
                            self.recover('placement_drift', 'RGB position changed during stability window')
                            unsettled = True; break
                    if unsettled: break  # restart grasp/place based on new image
                    self.stability_start = start
                    self.stability_verified_s = round(self.backend.now()-start, 6)
                    self.trace.event('stability_verified', self.backend.now(), duration_s=self.stability_verified_s,
                                     position_m=self.visual.position_m)
                    success = True; break
                if success: break
        except TrialFailure as error:
            terminal = error.category
            failure = {'category': error.category, 'phase': self.phase, 'detail': str(error)}
            if not self.failures or self.failures[-1] != failure:
                self.failures.append(failure)
            self.trace.event('trial_stopped', self.backend.now(), category=error.category, detail=str(error))
        finally:
            try:
                self.backend.stop(); stop_request_succeeded = True
                self.trace.event('stop_request_succeeded', self.backend.now())
            except Exception as error:
                success = False; terminal = 'stop_transport_failure'
                self.failures.append({'category': terminal, 'phase': self.phase, 'detail': type(error).__name__})
                self.trace.event('stop_request_failed', self.backend.now(), error_type=type(error).__name__)
        result = {
            'feedback_level': self.backend.environment['feedback_level'],
            'official_platform': False, 'official_score': None,
            'task': 'single_block_pick_place', 'success': success,
            'elapsed_s': round(self.backend.now()-self.start_s, 6),
            'terminal_failure': terminal, 'failures': self.failures,
            'stop_request_succeeded': stop_request_succeeded,
            'physical_stop_verified': False,
            'recoveries': self.recoveries, 'action_count': self.action_count,
            'stability_verified_s': self.stability_verified_s,
            'started_in_configured_supply': self.started_in_supply,
            'final_visual_position_m': self.visual.position_m if self.visual else None,
        }
        self.trace.finish(result)
        return result
