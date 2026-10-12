"""Synthetic RGB kinematic scene, NOT official simulation or contact physics.

Hidden scene state is used only to render images and execute toy dynamics.
The controller receives RGB and robot tip feedback through observe(), never
object pose/held-state or fault labels. It does not import this module.
"""
import math
import random
from dataclasses import dataclass, asdict, replace
from .contracts import ActionStatus
from .geometry import distance, look_at, finite

SCENARIOS = (
    'nominal', 'empty_grasp_once', 'slip_once', 'occluded_start',
    'release_stuck_once', 'drift_after_release', 'move_timeout',
    'motion_failed', 'stale_after_close', 'pose_leak', 'camera_skew',
    'occluded_always', 'duplicate_after_close', 'continuous_drift',
    'slip_during_transit', 'false_move_complete', 'sparse_frames',
)


@dataclass(frozen=True)
class SensorNoise:
    pixel_jitter_px: int = 0
    calibration_shift_m: float = 0.0

    def __post_init__(self):
        if type(self.pixel_jitter_px) is not int or not 0 <= self.pixel_jitter_px <= 8:
            raise ValueError('pixel jitter must be an integer from 0 to 8')
        if not finite(self.calibration_shift_m) or abs(self.calibration_shift_m) > 0.05:
            raise ValueError('synthetic calibration shift must be finite and <=50mm')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value)-{'pixel_jitter_px', 'calibration_shift_m'}:
            raise ValueError('unknown sensor-noise configuration')
        return cls(**value)


class SyntheticBackend:
    producer = 'synthetic_rgb_renderer_v1'
    cameras = (
        look_at('camera_left', (-0.38, -0.50, 0.65), (0, 0.40, 0.05)),
        look_at('camera_right', (0.38, -0.50, 0.65), (0, 0.40, 0.05)),
    )

    def __init__(self, scenario='nominal', seed=0, noise=None):
        self.noise = noise or SensorNoise()
        self._render_cameras = type(self).cameras
        self.cameras = tuple(replace(c, origin=(c.origin[0]+self.noise.calibration_shift_m, *c.origin[1:]))
                             for c in self._render_cameras)
        if scenario not in SCENARIOS: raise ValueError('unknown synthetic scenario')
        self.environment = {
            'feedback_level': 'synthetic_rgb_kinematic',
            'backend_version': '2', 'official_platform': False,
            'contact_physics': False, 'robot_model': 'point_tip_not_LBot',
            'scene': scenario, 'seed': seed,
            'camera_calibration': 'synthetic_pinhole_not_platform_calibration',
            'sensor_noise': asdict(self.noise),
        }
        self._scenario = scenario; self._rng = random.Random(seed)
        self._pixel_rng = random.Random(seed+100000)
        self._time = 0.0; self._sequence = 0
        self._tip = (0.0, 0.36, 0.23)
        self._block = (-0.16+self._rng.uniform(-0.015, 0.015),
                       0.42+self._rng.uniform(-0.02, 0.02), 0.025)
        self._held = False; self._slip_used = False; self._close_count = 0
        self._open_count = 0; self._moves = 0; self._drift_used = False
        self._released_at = None; self._released_point = None
        self._actions = {}; self._pending = None; self.stop_count = 0

    def now(self): return self._time

    def submit(self, action):
        if self._pending is not None: raise RuntimeError('cannot overlap toy backend actions')
        if action.kind not in ('move', 'grasp', 'release'): raise ValueError('unsupported internal action')
        action_id = f'a{len(self._actions)+1:04}'
        self._actions[action_id] = {'action': action, 'start': self._time,
                                    'status': ActionStatus('pending')}
        self._pending = action_id
        if action.kind == 'move': self._moves += 1
        return action_id

    def status(self, action_id): return self._actions[action_id]['status']

    def stop(self):
        self.stop_count += 1
        if self._pending:
            self._actions[self._pending]['status'] = ActionStatus('failed', 'stopped')
            self._pending = None

    def _advance(self):
        self._time = round(self._time+(0.5 if self._scenario == 'sparse_frames' else 0.1), 6); self._sequence += 1
        if self._pending:
            record = self._actions[self._pending]; action = record['action']
            if self._scenario == 'move_timeout' and self._moves == 1: return
            if self._time-record['start'] >= 0.3:
                if self._scenario == 'motion_failed' and self._moves == 1:
                    record['status'] = ActionStatus('failed', 'synthetic actuator failure')
                else:
                    if action.kind == 'move':
                        if not (self._scenario == 'false_move_complete' and self._moves == 1):
                            self._tip = tuple(action.target_m)
                        if self._held:
                            if (not self._slip_used and ((self._scenario == 'slip_once' and self._tip[2] > 0.12)
                                    or (self._scenario == 'slip_during_transit' and self._moves == 5))):
                                self._slip_used = True; self._held = False
                                self._block = (self._tip[0], self._tip[1], 0.025)
                            else: self._block = self._tip
                    elif action.kind == 'grasp':
                        self._close_count += 1
                        forced_empty = self._scenario == 'empty_grasp_once' and self._close_count == 1
                        self._held = not forced_empty and distance(self._tip, self._block) < 0.030
                        if self._held: self._block = self._tip
                    elif action.kind == 'release':
                        self._open_count += 1
                        stuck = self._scenario == 'release_stuck_once' and self._open_count == 1
                        if self._held and not stuck:
                            self._held = False
                            self._block = (self._tip[0], self._tip[1], 0.025)
                            self._released_at = self._time; self._released_point = self._block
                    # Command completion deliberately says nothing about held state.
                    record['status'] = ActionStatus('complete')
                self._pending = None
        if not self._held and self._released_at is not None:
            if self._scenario == 'drift_after_release' and not self._drift_used and self._time-self._released_at > 1.2:
                self._block = (self._block[0]+0.04, self._block[1], 0.025)
                self._drift_used = True
            elif self._scenario == 'continuous_drift':
                self._block = (self._released_point[0]+0.035*math.sin(8*(self._time-self._released_at)),
                               self._released_point[1], 0.025)

    def _render(self, camera, hidden):
        pixels = bytearray([35, 35, 35]) * (camera.width*camera.height)
        def rectangle(point, half_width, half_height, color):
            u, v = camera.project(point)
            # Exact RGB pixels; no hidden segmentation/pose accompanies them.
            jitter = self.noise.pixel_jitter_px
            cx = round(u)+self._pixel_rng.randint(-jitter, jitter)
            cy = round(v)+self._pixel_rng.randint(-jitter, jitter)
            for y in range(max(0, cy-half_height), min(camera.height, cy+half_height+1)):
                for x in range(max(0, cx-half_width), min(camera.width, cx+half_width+1)):
                    index = 3*(y*camera.width+x); pixels[index:index+3] = bytes(color)
        rectangle(self._tip, 2, 2, (180, 180, 180))
        if not hidden: rectangle(self._block, 5, 4, (230, 30, 30))
        return bytes(pixels)

    def observe(self):
        self._advance()
        hidden = self._scenario == 'occluded_always' or (self._scenario == 'occluded_start' and self._sequence <= 8)
        stale = self._scenario == 'stale_after_close' and self._close_count > 0
        duplicate = self._scenario == 'duplicate_after_close' and self._close_count > 0
        frames = []
        for i, camera in enumerate(self.cameras):
            stamp = self._time-0.5 if stale else self._time
            if self._scenario == 'camera_skew' and i == 1: stamp -= 0.1
            frames.append({'camera_id': camera.name, 'frame_id': camera.name+'_optical',
                           'sequence': 1 if duplicate else self._sequence,
                           'captured_s': stamp, 'width': camera.width, 'height': camera.height,
                           'encoding': 'rgb8', 'rgb': self._render(self._render_cameras[i], hidden)})
        raw = {'schema_version': 2, 'producer': self.producer, 'calibration_id': 'synthetic-pinhole-v1',
               'time_s': self._time, 'frames': frames,
               'robot': {'frame_id': 'base_link', 'tip_position_m': self._tip, 'producer': self.producer,
                         'captured_s': self._time, 'sequence': self._sequence}}
        if self._scenario == 'pose_leak': raw['ground_truth_object_pose'] = self._block
        return raw
