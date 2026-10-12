"""Internal adapter/data contracts; these names are not official SDK APIs."""
from dataclasses import dataclass, fields
from typing import Protocol
import math
from .geometry import Camera, finite, vector, distance


class ObservationError(ValueError):
    pass


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ObservationError(f'{label}: unexpected/missing fields; control input rejected')


@dataclass(frozen=True)
class Frame:
    camera_id: str
    frame_id: str
    sequence: int
    captured_s: float
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True)
class Observation:
    time_s: float
    frames: tuple
    tip: tuple
    calibration_id: str
    robot_captured_s: float | None = None
    robot_sequence: int | None = None
    robot_producer: str | None = None

    @property
    def sensor_time_s(self):
        stamps = [frame.captured_s for frame in self.frames]
        if self.robot_captured_s is not None:
            stamps.append(self.robot_captured_s)
        return min(stamps)


class ObservationGate:
    """RGB, reviewed calibration and timestamped robot feedback only.

    Schema 1 is accepted solely for explicitly requested legacy log replay.
    Schema 2 also validates robot publisher/sequence/time and sensor alignment.
    """
    def __init__(self, cameras, allowed_producer, max_age_s=0.25, max_skew_s=0.03,
                 allowed_robot_producer=None, allow_legacy=False, tip_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)):
        if not isinstance(cameras, (tuple, list)) or not cameras or not all(isinstance(c, Camera) for c in cameras):
            raise ValueError('expected nonempty reviewed camera list')
        if len({c.name for c in cameras}) != len(cameras) or len({c.calibration_id for c in cameras}) != 1:
            raise ValueError('duplicate camera names or inconsistent calibration group')
        if not isinstance(allowed_producer, str) or not allowed_producer.strip():
            raise ValueError('invalid approved producer')
        robot_producer = allowed_producer if allowed_robot_producer is None else allowed_robot_producer
        if not isinstance(robot_producer, str) or not robot_producer.strip():
            raise ValueError('invalid approved robot producer')
        if not finite(max_age_s) or max_age_s <= 0 or not finite(max_skew_s) or not 0 <= max_skew_s <= max_age_s:
            raise ValueError('invalid observation age/skew thresholds')
        if type(allow_legacy) is not bool:
            raise ValueError('allow_legacy must be boolean')
        self.tip_bounds = vector(tip_bounds, 6, 'tip bounds')
        if any(self.tip_bounds[i] >= self.tip_bounds[i+1] for i in (0, 2, 4)):
            raise ValueError('invalid tip bounds')
        self.cameras = {c.name: c for c in cameras}
        self.producer = allowed_producer; self.robot_producer = robot_producer
        self.max_age_s = max_age_s; self.max_skew_s = max_skew_s; self.allow_legacy = allow_legacy
        self.last_sequence = {}; self.last_capture = {}; self.last_time = -1.0
        self.last_robot_sequence = -1; self.last_robot_capture = -1.0

    def accept(self, raw, now_s):
        exact_keys(raw, {'schema_version', 'producer', 'calibration_id', 'time_s', 'frames', 'robot'}, 'observation')
        version = raw['schema_version']
        if type(version) is not int or version not in ((1, 2) if self.allow_legacy else (2,)):
            raise ObservationError('unsupported sensor schema')
        if raw['producer'] != self.producer or not isinstance(raw['producer'], str):
            raise ObservationError('unapproved observation producer')
        stamp = raw['time_s']
        if not finite(now_s) or now_s < 0 or not finite(stamp) or not 0 <= stamp <= now_s or stamp <= self.last_time:
            raise ObservationError('clock reset/non-finite/future time')
        if now_s-stamp > self.max_age_s:
            raise ObservationError('stale observation time')
        if not isinstance(raw['frames'], list) or len(raw['frames']) != len(self.cameras):
            raise ObservationError('expected exactly the configured RGB cameras')
        accepted = []; sequences = {}; captures = {}
        for value in raw['frames']:
            exact_keys(value, {'camera_id', 'frame_id', 'sequence', 'captured_s', 'width', 'height', 'encoding', 'rgb'}, 'frame')
            if not isinstance(value['camera_id'], str):
                raise ObservationError('camera id must be text')
            camera = self.cameras.get(value['camera_id'])
            if camera is None or camera.name in sequences:
                raise ObservationError('unknown/duplicate camera')
            if raw['calibration_id'] != camera.calibration_id or value['frame_id'] != camera.name+'_optical':
                raise ObservationError('calibration/frame mismatch')
            sequence = value['sequence']; capture = value['captured_s']
            if type(sequence) is not int or sequence < 0 or sequence <= self.last_sequence.get(camera.name, -1):
                raise ObservationError('repeated/out-of-order camera sequence')
            if (not finite(capture) or not 0 <= capture <= stamp or now_s-capture > self.max_age_s
                    or capture <= self.last_capture.get(camera.name, -1.0)):
                raise ObservationError('stale/future camera timestamp')
            if (type(value['width']) is not int or type(value['height']) is not int
                    or value['width'] != camera.width or value['height'] != camera.height or value['encoding'] != 'rgb8'):
                raise ObservationError('image dimensions/encoding mismatch')
            pixels = value['rgb']
            if not isinstance(pixels, bytes) or len(pixels) != camera.width*camera.height*3:
                raise ObservationError('invalid RGB payload')
            accepted.append(Frame(camera.name, value['frame_id'], sequence, capture, camera.width, camera.height, pixels))
            sequences[camera.name] = sequence; captures[camera.name] = capture
        robot_keys = {'frame_id', 'tip_position_m'}
        if version == 2:
            robot_keys |= {'producer', 'captured_s', 'sequence'}
        exact_keys(raw['robot'], robot_keys, 'robot feedback')
        robot = raw['robot']
        try: tip = vector(robot['tip_position_m'], 3, 'robot tip')
        except ValueError as error: raise ObservationError(str(error)) from error
        if any(not self.tip_bounds[2*i] <= tip[i] <= self.tip_bounds[2*i+1] for i in range(3)):
            raise ObservationError('measured tip outside configured bounds')
        if robot['frame_id'] != 'base_link':
            raise ObservationError('invalid robot feedback frame')
        robot_capture = robot_sequence = robot_source = None
        all_captures = list(captures.values())
        if version == 2:
            robot_capture = robot['captured_s']; robot_sequence = robot['sequence']; robot_source = robot['producer']
            if not isinstance(robot_source, str) or robot_source != self.robot_producer:
                raise ObservationError('unapproved robot producer')
            if type(robot_sequence) is not int or robot_sequence < 0 or robot_sequence <= self.last_robot_sequence:
                raise ObservationError('repeated/out-of-order robot sequence')
            if (not finite(robot_capture) or not 0 <= robot_capture <= stamp or now_s-robot_capture > self.max_age_s
                    or robot_capture <= self.last_robot_capture):
                raise ObservationError('stale/future robot timestamp')
            all_captures.append(robot_capture)
        if max(all_captures)-min(all_captures) > self.max_skew_s:
            raise ObservationError('unsynchronized camera/robot samples')
        # Commit sequence state atomically only after the whole packet passes.
        self.last_time = stamp; self.last_sequence = sequences; self.last_capture = captures
        if version == 2:
            self.last_robot_sequence = robot_sequence; self.last_robot_capture = robot_capture
        return Observation(stamp, tuple(accepted), tip, raw['calibration_id'], robot_capture, robot_sequence, robot_source)


PHASE_ACTIONS = {
    'raise_for_approach': 'move', 'above_grasp': 'move', 'grasp_descent': 'move',
    'close': 'grasp', 'lift': 'move', 'reset_empty_grasp': 'release',
    'target_transit': 'move', 'target_descent': 'move', 'release': 'release', 'retreat': 'move',
}


@dataclass(frozen=True)
class Action:
    kind: str
    target_m: tuple | None = None


@dataclass(frozen=True)
class ActionStatus:
    state: str
    detail: str = ''


class Backend(Protocol):
    """Calls must return within adapter deadlines; SDK mapping remains unimplemented."""
    cameras: tuple[Camera, ...]
    producer: str
    environment: dict
    def now(self) -> float: ...
    def observe(self) -> dict: ...
    def submit(self, action: Action) -> str: ...
    def status(self, action_id: str) -> ActionStatus: ...
    def stop(self) -> None: ...


@dataclass(frozen=True)
class Config:
    target: tuple = (0.16, 0.42, 0.025)
    safe_height_m: float = 0.20
    transit_min_height_m: float = 0.15
    action_timeout_s: float = 1.8
    trial_timeout_s: float = 30.0
    wall_timeout_s: float = 60.0
    settle_s: float = 3.0
    stability_m: float = 0.015
    goal_tolerance_m: float = 0.025
    table_center_height_m: float = 0.025
    table_height_tolerance_m: float = 0.012
    lift_min_m: float = 0.06
    grasp_distance_m: float = 0.03
    hand_separation_m: float = 0.08
    motion_completion_m: float = 0.015
    max_observation_age_s: float = 0.25
    max_sensor_skew_s: float = 0.03
    max_observation_gap_s: float = 0.25
    stereo_residual_m: float = 0.02
    recovery_enabled: bool = True
    robust_observations: bool = False
    robust_min_samples: int = 9
    robust_max_samples: int = 49
    robust_max_uncertainty_m: float = 0.0075
    robust_sigma_factor: float = 2.0
    robust_pixel_span_px: float = 6.0
    robust_stationary_tip_m: float = 0.003
    robust_absolute_floor_m: float = 0.002
    robust_difference_floor_m: float = 0.001
    robust_tracking_pixel_px: float = 4.0
    robust_change_sigma: float = 4.0
    robust_change_floor_px: float = 0.5
    supply_xy: tuple = (-0.30, -0.05, 0.20, 0.60)
    build_xy: tuple = (0.05, 0.30, 0.20, 0.60)
    workspace: tuple = (-0.35, 0.35, 0.10, 0.65, 0.005, 0.30)

    def __post_init__(self):
        for name, size in (('target', 3), ('supply_xy', 4), ('build_xy', 4), ('workspace', 6)):
            object.__setattr__(self, name, vector(getattr(self, name), size, name))
        for item in fields(self):
            if item.name not in ('target', 'supply_xy', 'build_xy', 'workspace', 'recovery_enabled', 'robust_observations', 'robust_min_samples', 'robust_max_samples'):
                value = getattr(self, item.name)
                if not finite(value) or (value < 0 if item.name == 'max_sensor_skew_s' else value <= 0):
                    raise ValueError(f'{item.name}: expected finite positive value')
        if type(self.recovery_enabled) is not bool or type(self.robust_observations) is not bool:
            raise ValueError('recovery_enabled/robust_observations must be boolean')
        if (type(self.robust_min_samples) is not int or type(self.robust_max_samples) is not int
                or not 3 <= self.robust_min_samples <= self.robust_max_samples <= 99):
            raise ValueError('invalid robust sample counts')
        if self.robust_max_uncertainty_m >= self.stability_m or self.robust_difference_floor_m >= self.stability_m:
            raise ValueError('uncertainty budget cannot exceed stability criterion')
        for zone in (self.supply_xy, self.build_xy):
            if zone[0] >= zone[1] or zone[2] >= zone[3]: raise ValueError('invalid region ordering')
        if any(self.workspace[i] >= self.workspace[i+1] for i in (0, 2, 4)):
            raise ValueError('invalid workspace ordering')
        for zone in (self.supply_xy, self.build_xy):
            if not (self.workspace[0] <= zone[0] < zone[1] <= self.workspace[1]
                    and self.workspace[2] <= zone[2] < zone[3] <= self.workspace[3]):
                raise ValueError('region outside workspace')
        if not inside_xy(self.target, self.build_xy) or not within_workspace(self.target, self):
            raise ValueError('target outside build region/workspace')
        if abs(self.target[2]-self.table_center_height_m) > 1e-9:
            raise ValueError('single-block table placement requires target/table center heights to match')
        if not (self.target[2] < self.transit_min_height_m <= self.safe_height_m <= self.workspace[5]):
            raise ValueError('unsafe transit/safe height')
        if self.settle_s < 3 or self.max_sensor_skew_s > self.max_observation_age_s:
            raise ValueError('invalid stability/sensor timing bounds')
        if self.max_observation_gap_s > min(self.settle_s, self.max_observation_age_s):
            raise ValueError('observation gap exceeds stability window')
        if (self.stability_m > self.goal_tolerance_m or self.stability_m >= self.lift_min_m/2
                or self.table_height_tolerance_m >= self.lift_min_m/2
                or self.motion_completion_m >= self.lift_min_m/2):
            raise ValueError('visual/feedback tolerances cannot mask a lift or placement failure')
        if self.hand_separation_m <= self.grasp_distance_m or self.safe_height_m-self.target[2] <= self.hand_separation_m:
            raise ValueError('hand clearance incompatible with grasp/transit configuration')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value)-{item.name for item in fields(cls)}:
            raise ValueError('config: unknown fields or wrong structure')
        return cls(**value)


def inside_xy(point, zone):
    return zone[0] <= point[0] <= zone[1] and zone[2] <= point[1] <= zone[3]


def within_workspace(point, config):
    return all(config.workspace[2*i] <= point[i] <= config.workspace[2*i+1] for i in range(3))


def placement_valid(point, tip, config):
    return (inside_xy(point, config.build_xy) and distance(point, config.target) <= config.goal_tolerance_m
            and abs(point[2]-config.table_center_height_m) <= config.table_height_tolerance_m
            and distance(point, tip) >= config.hand_separation_m)


def guard_move(current, target, config):
    current = vector(current, 3, 'measured tip'); target = vector(target, 3, 'motion target')
    if not within_workspace(current, config) or not within_workspace(target, config):
        raise ValueError('measured tip/target outside experiment workspace')
    horizontal = math.hypot(target[0]-current[0], target[1]-current[1])
    if horizontal > 1e-6 and min(current[2], target[2]) < config.transit_min_height_m:
        raise ValueError('horizontal transit below configured safe height')
