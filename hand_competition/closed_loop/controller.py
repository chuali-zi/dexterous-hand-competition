"""One-block controller; decisions use the internal RGB/robot-feedback contract."""
from dataclasses import asdict
import time
from .contracts import (Action, ActionStatus, ObservationGate, ObservationError, guard_move,
                        inside_xy, placement_valid, finite, PHASE_ACTIONS)
from .geometry import distance, sub, add
from .vision import estimate_single_block, observe_single_block_pixels, VisionError
from .robust import Sample, estimate_bundle, BundleError, bounded_placement, bounded_stability


class TrialFailure(RuntimeError):
    def __init__(self, category, detail):
        super().__init__(detail)
        self.category = category


class Controller:
    def __init__(self, backend, config, trace):
        self.backend = backend; self.config = config; self.trace = trace
        self.gate = ObservationGate(backend.cameras, backend.producer,
                                    config.max_observation_age_s, config.max_sensor_skew_s, tip_bounds=config.workspace)
        self.trace.producer = backend.producer
        self.observation = None; self.visual = None; self.failures = []
        self.pixels = None; self.sample = None; self.bundle = None; self.bundle_id = None
        self.held_offset = None; self.held_color = None; self.bundle_reset_count = 0
        self.recoveries = 0; self.started_in_supply = False; self.phase = 'initial'
        self.stability_verified_s = 0.0; self.start_s = 0.0; self.last_time = 0.0
        self.wall_start_s = time.monotonic(); self.action_count = 0; self.action_ids = set()
        self.holding_confirmed = False; self.release_requested = False; self.finished = False

    def clock(self):
        try: now = self.backend.now()
        except Exception as error: raise TrialFailure('backend_clock_failure', type(error).__name__) from error
        if not finite(now) or now < self.last_time:
            raise TrialFailure('backend_clock_failure', 'non-finite/reset backend clock')
        self.last_time = now
        return now

    def deadline(self):
        if (self.clock()-self.start_s >= self.config.trial_timeout_s or
                time.monotonic()-self.wall_start_s >= self.config.wall_timeout_s):
            raise TrialFailure('trial_timeout', 'experiment deadline reached')

    def emit(self, kind, **payload):
        try: self.trace.event(kind, self.last_time, **payload)
        except (OSError, ValueError) as error:
            raise TrialFailure('artifact_io_failure', type(error).__name__) from error

    def sense(self):
        self.deadline()
        try: raw = self.backend.observe()
        except Exception as error: raise TrialFailure('sensor_transport_failure', type(error).__name__) from error
        self.deadline()  # an observe call returning late cannot complete a trial
        try: self.observation = self.gate.accept(raw, self.last_time)
        except ObservationError as error:
            self.emit('observation_rejected', error=str(error))
            raise TrialFailure('invalid_observation', str(error)) from error
        self.pixels = None
        try:
            self.pixels = observe_single_block_pixels(self.observation)
            self.visual = estimate_single_block(self.observation, self.backend.cameras, self.config.stereo_residual_m, self.pixels)
            vision_error = None
        except VisionError as error:
            self.visual = None; vision_error = str(error)
            if 'missing/ambiguous' in str(error) or 'color correspondence' in str(error): self.pixels = None
        self.sample = None
        if self.pixels is not None:
            self.sample = Sample(self.trace.observations, self.observation.sensor_time_s, self.observation.tip,
                                 self.pixels.camera_ids, self.pixels.pixels, self.pixels.color)
        try: self.trace.observation(self.observation, self.visual, error=vision_error)
        except (OSError, ValueError) as error: raise TrialFailure('artifact_io_failure', type(error).__name__) from error
        return self.visual

    def visible(self):
        while True:
            estimate = self.sense()
            if estimate is not None: return estimate

    def command(self, phase, kind, target=None):
        if self.finished: raise TrialFailure('phase_precondition_failure', 'trial already finalized')
        self.phase = phase; self.deadline()
        if PHASE_ACTIONS.get(phase) != kind:
            raise TrialFailure('action_protocol_failure', 'phase/action mismatch')
        if phase in ('target_transit', 'target_descent', 'release') and not self.holding_confirmed:
            raise TrialFailure('phase_precondition_failure', 'no visually confirmed held object')
        if phase == 'release' and distance(self.observation.tip, self.config.target) > self.config.motion_completion_m:
            raise TrialFailure('phase_precondition_failure', 'release requested away from measured placement target')
        if phase == 'retreat' and not self.release_requested:
            raise TrialFailure('phase_precondition_failure', 'retreat requested without a release attempt')
        grasp_point = self.bundle.position_m if self.config.robust_observations and self.bundle else self.visual.position_m if self.visual else None
        grasp_radius = self.bundle.radius_m+self.config.robust_absolute_floor_m if self.config.robust_observations and self.bundle else 0
        if phase == 'close' and (grasp_point is None or distance(grasp_point, self.observation.tip)+grasp_radius > self.config.grasp_distance_m):
            raise TrialFailure('phase_precondition_failure', 'closing without a fresh visual grasp target')
        if kind == 'move':
            try: guard_move(self.observation.tip, target, self.config)
            except ValueError as error: raise TrialFailure('unsafe_waypoint', str(error)) from error
        action = Action(kind, target)
        requested = self.clock(); wall_requested = time.monotonic()
        try: action_id = self.backend.submit(action)
        except Exception as error: raise TrialFailure('action_transport_failure', type(error).__name__) from error
        if not isinstance(action_id, str) or not action_id.strip() or action_id in self.action_ids:
            raise TrialFailure('action_protocol_failure', 'invalid/reused action id')
        self.action_ids.add(action_id); self.action_count += 1
        self.clock()
        started = requested
        self.emit('action_submitted', phase=phase, action_id=action_id, action=asdict(action),
                  requested_s=requested, grasp_success_inferred=False,
                  **({'estimate_id': self.bundle_id} if self.config.robust_observations else {}))
        self.deadline()
        if self.last_time-started >= self.config.action_timeout_s or time.monotonic()-wall_requested >= self.config.action_timeout_s:
            raise TrialFailure('action_timeout', f'{phase}: submit exceeded deadline')
        while True:
            previous_sensor = self.observation.sensor_time_s
            self.sense()
            # Timeout is checked before accepting a late "complete" response.
            if (self.last_time-started >= self.config.action_timeout_s
                    or time.monotonic()-wall_requested >= self.config.action_timeout_s):
                raise TrialFailure('action_timeout', f'{phase}: command exceeded deadline')
            try: status = self.backend.status(action_id)
            except Exception as error: raise TrialFailure('action_transport_failure', type(error).__name__) from error
            self.deadline()
            if (self.last_time-started >= self.config.action_timeout_s
                    or time.monotonic()-wall_requested >= self.config.action_timeout_s):
                raise TrialFailure('action_timeout', f'{phase}: status exceeded deadline')
            if (not isinstance(status, ActionStatus) or status.state not in ('pending', 'complete', 'failed')
                    or not isinstance(status.detail, str)):
                raise TrialFailure('action_protocol_failure', 'invalid action status')
            self.emit('action_status', action_id=action_id, state=status.state, detail=status.detail)
            if status.state == 'failed': raise TrialFailure('actuator_failure', status.detail)
            if phase in ('target_transit', 'target_descent'):
                if self.config.robust_observations:
                    if self.pixels is None or self.observation.sensor_time_s-previous_sensor > self.config.max_observation_gap_s+1e-9:
                        raise TrialFailure('carried_object_unobserved', 'lost RGB evidence/coverage while carrying')
                    if self.held_offset is None or self.pixels.color != self.held_color:
                        raise TrialFailure('carried_object_lost', 'object identity/held model missing')
                    predicted = add(self.observation.tip, self.held_offset)
                    for name, pixel in zip(self.pixels.camera_ids, self.pixels.pixels):
                        camera = next(c for c in self.backend.cameras if c.name == name)
                        expected = camera.project(predicted)
                        if sum((pixel[i]-expected[i])**2 for i in (0, 1))**0.5 > self.config.robust_tracking_pixel_px:
                            raise TrialFailure('carried_object_lost', 'RGB innovation inconsistent with observed held offset')
                else:
                    if self.visual is None:
                        raise TrialFailure('carried_object_unobserved', 'lost RGB evidence while carrying')
                    if distance(self.visual.position_m, self.observation.tip) > self.config.grasp_distance_m:
                        raise TrialFailure('carried_object_lost', 'RGB object no longer tracks measured hand')
            if status.state == 'complete':
                if kind == 'move' and distance(self.observation.tip, target) > self.config.motion_completion_m:
                    raise TrialFailure('motion_not_verified', 'reported completion without measured target arrival')
                if phase == 'release':
                    self.release_requested = True; self.holding_confirmed = False
                elif phase == 'close':
                    self.release_requested = False
                return

    def recover(self, category, detail):
        self.failures.append({'category': category, 'phase': self.phase, 'detail': detail})
        self.emit('failure_detected', category=category, phase=self.phase, detail=detail)
        if not self.config.recovery_enabled: raise TrialFailure(category, detail)
        self.recoveries += 1
        self.emit('recovery_started', category=category, recovery_number=self.recoveries)

    def place_held_block(self):
        target = self.config.target
        self.command('target_transit', 'move', (target[0], target[1], self.config.safe_height_m))
        self.command('target_descent', 'move', target)
        self.command('release', 'release')
        self.command('retreat', 'move', (target[0], target[1], self.config.safe_height_m))

    def settle(self, initial):
        self.phase = 'settling'
        anchor = initial.position_m; start = self.observation.sensor_time_s; previous = start
        self.emit('settle_started', anchor_m=anchor, sensor_start_s=start)
        while self.observation.sensor_time_s-start < self.config.settle_s:
            estimate = self.sense(); stamp = self.observation.sensor_time_s
            if stamp-previous > self.config.max_observation_gap_s+1e-9:
                self.emit('settle_reset', reason='sensor_gap')
                return 'restart_window'
            previous = stamp
            if estimate is None:
                self.emit('settle_reset', reason='visibility_lost')
                return 'restart_window'
            point = estimate.position_m
            if not placement_valid(point, self.observation.tip, self.config):
                self.emit('settle_reset', reason='placement_or_hand_clearance')
                self.recover('placement_drift', 'placement/support-height/hand-clearance criterion failed')
                return 'regrasp'
            if distance(point, anchor) > self.config.stability_m:
                self.emit('settle_reset', reason='observed_motion')
                self.recover('placement_drift', 'RGB position changed during stability window')
                return 'regrasp'
        self.deadline()
        self.stability_verified_s = round(self.observation.sensor_time_s-start, 6)
        self.emit('stability_verified', duration_s=self.stability_verified_s, sensor_start_s=start,
                  sensor_end_s=self.observation.sensor_time_s, position_m=self.visual.position_m)
        return 'stable'

    def execute_raw(self):
        first = self.visible()
        self.started_in_supply = inside_xy(first.position_m, self.config.supply_xy)
        if not self.started_in_supply:
            raise TrialFailure('invalid_initial_location', 'block not in configured supply zone')
        while True:
            block = self.visible().position_m
            self.command('raise_for_approach', 'move', (*self.observation.tip[:2], self.config.safe_height_m))
            self.command('above_grasp', 'move', (*block[:2], self.config.safe_height_m))
            current = self.visible().position_m
            if distance(current, block) > self.config.motion_completion_m:
                self.recover('target_moved', 'RGB target moved during approach')
                continue
            self.command('grasp_descent', 'move', (*block[:2], self.config.table_center_height_m))
            if self.visual is None or distance(self.visual.position_m, block) > self.config.grasp_distance_m:
                raise TrialFailure('grasp_target_unverified', 'target lost/moved before closing')
            self.command('close', 'grasp')
            self.command('lift', 'move', (*block[:2], self.config.safe_height_m))
            lifted = self.visible().position_m
            if lifted[2]-block[2] < self.config.lift_min_m or distance(lifted, self.observation.tip) > self.config.grasp_distance_m:
                self.recover('grasp_not_verified', 'RGB did not confirm object lifting with hand')
                if abs(lifted[2]-self.config.table_center_height_m) > self.config.table_height_tolerance_m:
                    raise TrialFailure('uncertain_grasp_state', 'unknown airborne state cannot be safely recovered')
                self.command('reset_empty_grasp', 'release')
                continue
            self.holding_confirmed = True
            self.emit('grasp_verified', position_m=lifted, tip_m=self.observation.tip)
            self.place_held_block()
            while True:
                estimate = self.visible(); point = estimate.position_m
                if abs(point[2]-self.config.table_center_height_m) > self.config.table_height_tolerance_m:
                    if distance(point, self.observation.tip) <= self.config.grasp_distance_m:
                        self.holding_confirmed = True
                        self.recover('release_not_verified', 'object followed retreat')
                        self.place_held_block(); continue
                    raise TrialFailure('uncertain_release_state', 'object height does not confirm supported placement')
                if not placement_valid(point, self.observation.tip, self.config):
                    self.recover('placement_drift', 'placement/clearance criterion failed')
                    break
                settled = self.settle(estimate)
                if settled == 'stable': return True
                if settled == 'regrasp': break
                # A gap/brief occlusion invalidates the window, not the placed block.
                # Observe again at the parked hand before deciding to re-grasp.

    def acquire(self, predicate=None, precision=False):
        samples = []
        while True:
            self.sense()
            if self.sample is None:
                samples = []; self.bundle_reset_count += 1
                self.emit('bundle_reset', reason='missing_pixels')
                continue
            if samples and self.sample.sensor_time_s-samples[-1].sensor_time_s > self.config.max_observation_gap_s+1e-9:
                samples = []; self.bundle_reset_count += 1
                self.emit('bundle_reset', reason='sensor_gap')
            samples.append(self.sample)
            if len(samples) > self.config.robust_max_samples:
                samples = samples[-self.config.robust_max_samples:]
                if precision:
                    self.bundle_reset_count += 1
                    self.emit('bundle_reset', reason='rolling_window_trim')
            if len(samples) < self.config.robust_min_samples: continue
            try: estimated = estimate_bundle(samples, self.backend.cameras, self.config)
            except BundleError as error:
                if str(error) in ('nonstationary_pixels', 'robot_moved', 'identity_change', 'temporal_change'):
                    samples = []; self.bundle_reset_count += 1
                    self.emit('bundle_reset', reason=str(error))
                continue
            budget = self.config.robust_max_uncertainty_m if precision else self.config.grasp_distance_m/2
            if estimated.radius_m > budget: continue
            if predicate is not None and not predicate(estimated): continue
            self.bundle = estimated; self.bundle_id = self.trace.count
            self.emit('robust_observation', estimate_id=self.bundle_id, sample_indices=estimated.sample_indices,
                      position_m=estimated.position_m, uncertainty_xyz_m=estimated.uncertainty_xyz_m,
                      color=estimated.color, sensor_start_s=estimated.sensor_start_s,
                      sensor_end_s=estimated.sensor_end_s, mean_pixels=estimated.mean_pixels,
                      effective_samples=estimated.effective_samples)
            return estimated

    def held_candidate(self, estimated):
        return distance(estimated.position_m, self.observation.tip)+estimated.radius_m+self.config.robust_absolute_floor_m <= self.config.grasp_distance_m

    def confirm_held(self):
        estimated = self.acquire(lambda value: self.held_candidate(value) and value.color == self.held_color)
        self.holding_confirmed = True; self.held_color = estimated.color
        self.held_offset = sub(estimated.position_m, self.observation.tip)
        return estimated

    def place_robust(self):
        target = self.config.target
        self.command('target_transit', 'move', (target[0], target[1], self.config.safe_height_m))
        self.holding_confirmed = False; self.confirm_held()
        self.command('target_descent', 'move', target)
        self.holding_confirmed = False; self.confirm_held()
        self.command('release', 'release')
        self.command('retreat', 'move', (target[0], target[1], self.config.safe_height_m))

    def execute_robust(self):
        first = self.acquire()
        self.started_in_supply = inside_xy(first.position_m, self.config.supply_xy)
        if not self.started_in_supply: raise TrialFailure('invalid_initial_location', 'block not in supply region')
        block = first
        while True:
            self.command('raise_for_approach', 'move', (*self.observation.tip[:2], self.config.safe_height_m))
            self.command('above_grasp', 'move', (*block.position_m[:2], self.config.safe_height_m))
            current = self.acquire()
            if distance(current.position_m, block.position_m)-current.radius_m-block.radius_m > self.config.motion_completion_m:
                self.recover('target_moved', 'confirmed RGB target change during approach'); block = current; continue
            self.command('grasp_descent', 'move', (*block.position_m[:2], self.config.table_center_height_m))
            before = self.acquire(lambda value: self.held_candidate(value) and
                                  abs(value.position_m[2]-self.config.table_center_height_m)+value.uncertainty_xyz_m[2] <= self.config.table_height_tolerance_m)
            self.command('close', 'grasp')
            self.command('lift', 'move', (*block.position_m[:2], self.config.safe_height_m))
            lifted = self.acquire()
            lift_margin = lifted.uncertainty_xyz_m[2]+before.uncertainty_xyz_m[2]
            if lifted.position_m[2]-before.position_m[2]-lift_margin < self.config.lift_min_m or not self.held_candidate(lifted):
                self.recover('grasp_not_verified', 'bundle did not confirm lift and hand proximity')
                if abs(lifted.position_m[2]-self.config.table_center_height_m)+lifted.uncertainty_xyz_m[2] > self.config.table_height_tolerance_m:
                    raise TrialFailure('uncertain_grasp_state', 'unknown airborne state')
                self.command('reset_empty_grasp', 'release'); block = lifted; continue
            self.holding_confirmed = True; self.held_color = lifted.color
            self.held_offset = sub(lifted.position_m, self.observation.tip)
            self.emit('grasp_verified', position_m=lifted.position_m, tip_m=self.observation.tip, estimate_id=self.bundle_id)
            self.place_robust()
            anchor = None; start = None; anchor_id = None
            while True:
                resets = self.bundle_reset_count
                estimated = self.acquire(precision=True)
                point = estimated.position_m
                if distance(point, self.observation.tip)+estimated.radius_m <= self.config.grasp_distance_m:
                    self.holding_confirmed = True; self.held_color = estimated.color
                    self.held_offset = sub(point, self.observation.tip)
                    self.recover('release_not_verified', 'object remains near withdrawn hand'); self.place_robust(); anchor = None; continue
                # Act on confirmed displacement, not on a single noisy frame.
                if distance(point, self.config.target)-estimated.radius_m-self.config.robust_absolute_floor_m > self.config.goal_tolerance_m:
                    self.recover('placement_drift', 'confident displacement outside target'); block = estimated; break
                if not bounded_placement(estimated, self.observation.tip, self.config):
                    anchor = None; self.emit('settle_reset', reason='uncertain_placement'); continue
                if anchor is None or resets != self.bundle_reset_count:
                    anchor = estimated; start = estimated.sensor_start_s; anchor_id = self.bundle_id
                    self.phase = 'settling'
                    self.emit('settle_started', anchor_m=point, sensor_start_s=start, estimate_id=anchor_id)
                    continue
                if not bounded_stability(anchor, estimated, self.config):
                    self.emit('settle_reset', reason='bundle_motion_or_uncertainty'); anchor = None; continue
                if estimated.sensor_end_s-start >= self.config.settle_s:
                    self.deadline(); self.stability_verified_s = round(estimated.sensor_end_s-start, 6)
                    self.emit('stability_verified', duration_s=self.stability_verified_s, sensor_start_s=start,
                              sensor_end_s=estimated.sensor_end_s, position_m=point,
                              estimate_id=self.bundle_id, anchor_estimate_id=anchor_id)
                    return True

    def run(self):
        if self.finished: raise ValueError('controller can only run once')
        success = False; terminal = None; stop_request_succeeded = False
        try:
            self.start_s = self.clock()
            self.emit('trial_started', start_s=self.start_s)
            success = self.execute_robust() if self.config.robust_observations else self.execute_raw()
        except Exception as error:
            terminal = error.category if isinstance(error, TrialFailure) else 'controller_internal_error'
            detail = str(error) if isinstance(error, TrialFailure) else type(error).__name__
            failure = {'category': terminal, 'phase': self.phase, 'detail': detail}
            if not self.failures or self.failures[-1] != failure: self.failures.append(failure)
            try: self.emit('trial_stopped', category=terminal, detail=detail)
            except TrialFailure: pass
        finally:
            try:
                self.backend.stop(); stop_request_succeeded = True
            except Exception as error:
                success = False; terminal = 'stop_transport_failure'
                self.failures.append({'category': terminal, 'phase': self.phase, 'detail': type(error).__name__})
                try: self.emit('stop_request_failed', error_type=type(error).__name__)
                except TrialFailure: pass
            else:
                try: self.emit('stop_request_succeeded')
                except TrialFailure:
                    success = False; terminal = 'artifact_io_failure'
        result = {
            'feedback_level': self.backend.environment['feedback_level'], 'official_platform': False, 'official_score': None,
            'task': 'single_block_pick_place', 'success': success, 'elapsed_s': round(self.last_time-self.start_s, 6),
            'terminal_failure': terminal, 'failures': self.failures, 'stop_request_succeeded': stop_request_succeeded,
            'physical_stop_verified': False, 'recoveries': self.recoveries, 'action_count': self.action_count,
            'stability_verified_s': self.stability_verified_s, 'started_in_configured_supply': self.started_in_supply,
            'final_visual_position_m': self.visual.position_m if self.visual else None,
            **({'final_robust_estimate_id': self.bundle_id} if self.config.robust_observations else {}),
            'trace_complete': terminal != 'artifact_io_failure',
        }
        try: self.trace.finish(result, time_s=self.last_time)
        except (OSError, ValueError):
            result.update(success=False, trace_complete=False, terminal_failure='artifact_io_failure')
        self.finished = True
        return result
