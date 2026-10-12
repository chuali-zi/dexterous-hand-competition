"""Our adapter contract. None of these names claim to be official SDK APIs."""
from dataclasses import dataclass
import math
from typing import Protocol
from .geometry import Camera


class ObservationError(ValueError):
    pass


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ObservationError(f'{label}: unexpected/missing fields; control input rejected')


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


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


class ObservationGate:
    """Only RGB + explicitly configured camera calibration + robot tip feedback.

    No scene entity pose, object TF, depth, force, segmentation or hidden labels.
    Future official permission and actual topic provenance require separate review.
    """
    def __init__(self, cameras, allowed_producer, max_age_s=0.25, max_skew_s=0.03):
        self.cameras = {c.name: c for c in cameras}
        self.producer = allowed_producer
        self.max_age_s = max_age_s; self.max_skew_s = max_skew_s
        self.last_sequence = {}; self.last_capture = {}; self.last_time = -1.0

    def accept(self, raw, now_s):
        exact_keys(raw, {'schema_version', 'producer', 'calibration_id', 'time_s', 'frames', 'robot'}, 'observation')
        if raw['schema_version'] != 1 or type(raw['schema_version']) is not int:
            raise ObservationError('unsupported sensor schema')
        if not isinstance(raw['producer'], str) or raw['producer'] != self.producer:
            raise ObservationError('unapproved observation producer')
        if not finite(now_s) or not finite(raw['time_s']) or raw['time_s'] <= self.last_time:
            raise ObservationError('clock reset/non-finite time')
        if abs(raw['time_s']-now_s) > self.max_age_s:
            raise ObservationError('stale observation time')
        if not isinstance(raw['frames'], list) or len(raw['frames']) != len(self.cameras):
            raise ObservationError('expected exactly the configured RGB cameras')
        frames = []; next_sequences = {}; next_captures = {}
        for value in raw['frames']:
            exact_keys(value, {'camera_id', 'frame_id', 'sequence', 'captured_s', 'width', 'height', 'encoding', 'rgb'}, 'frame')
            if not isinstance(value['camera_id'], str):
                raise ObservationError('camera id must be text')
            camera = self.cameras.get(value['camera_id'])
            if camera is None or camera.name in next_sequences:
                raise ObservationError('unknown/duplicate camera')
            if raw['calibration_id'] != camera.calibration_id or value['frame_id'] != camera.name+'_optical':
                raise ObservationError('calibration/frame mismatch')
            sequence = value['sequence']
            if type(sequence) is not int or sequence <= self.last_sequence.get(camera.name, -1):
                raise ObservationError('repeated/out-of-order camera sequence')
            stamp = value['captured_s']
            if (not finite(stamp) or stamp > now_s or now_s-stamp > self.max_age_s
                    or stamp <= self.last_capture.get(camera.name, -1.0)):
                raise ObservationError('stale/future camera timestamp')
            if (type(value['width']) is not int or type(value['height']) is not int
                    or value['width'] != camera.width or value['height'] != camera.height or value['encoding'] != 'rgb8'):
                raise ObservationError('image dimensions/encoding mismatch')
            pixels = value['rgb']
            if not isinstance(pixels, bytes) or len(pixels) != camera.width*camera.height*3:
                raise ObservationError('invalid RGB payload')
            frames.append(Frame(camera.name, value['frame_id'], sequence, stamp, camera.width, camera.height, pixels))
            next_sequences[camera.name] = sequence; next_captures[camera.name] = stamp
        if max(f.captured_s for f in frames)-min(f.captured_s for f in frames) > self.max_skew_s:
            raise ObservationError('unsynchronized cameras')
        exact_keys(raw['robot'], {'frame_id', 'tip_position_m'}, 'robot feedback')
        tip = raw['robot']['tip_position_m']
        if raw['robot']['frame_id'] != 'base_link' or not isinstance(tip, (tuple, list)) or len(tip) != 3 or not all(finite(x) for x in tip):
            raise ObservationError('invalid robot feedback/frame')
        self.last_time = raw['time_s']; self.last_sequence = next_sequences; self.last_capture = next_captures
        return Observation(raw['time_s'], tuple(frames), tuple(tip), raw['calibration_id'])


@dataclass(frozen=True)
class Action:
    kind: str
    target_m: tuple | None = None


@dataclass(frozen=True)
class ActionStatus:
    state: str  # pending, complete, failed; complete is NOT grasp success
    detail: str = ''


class Backend(Protocol):
    """Internal boundary to implement only after reviewing the actual platform."""
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
    goal_tolerance_m: float = 0.025  # experiment criterion, NOT official judging tolerance
    table_center_height_m: float = 0.025
    lift_min_m: float = 0.06
    recovery_enabled: bool = True
    supply_xy: tuple = (-0.30, -0.05, 0.20, 0.60)
    build_xy: tuple = (0.05, 0.30, 0.20, 0.60)

    def __post_init__(self):
        numbers = (*self.target, self.safe_height_m, self.transit_min_height_m,
                   self.action_timeout_s, self.trial_timeout_s, self.wall_timeout_s, self.settle_s,
                   self.stability_m, self.goal_tolerance_m, self.table_center_height_m,
                   self.lift_min_m, *self.supply_xy, *self.build_xy)
        if not all(finite(v) for v in numbers): raise ValueError('non-finite experiment configuration')
        if self.settle_s < 3 or self.action_timeout_s <= 0 or self.trial_timeout_s <= 0 or self.wall_timeout_s <= 0:
            raise ValueError('invalid time budget/stability duration')
        if self.stability_m <= 0 or self.goal_tolerance_m <= 0 or self.lift_min_m <= 0:
            raise ValueError('invalid visual thresholds')
        if self.safe_height_m < self.transit_min_height_m or self.target[2] >= self.transit_min_height_m:
            raise ValueError('unsafe waypoint height configuration')
        if type(self.recovery_enabled) is not bool:
            raise ValueError('recovery_enabled must be boolean')
        for zone in (self.supply_xy, self.build_xy):
            if len(zone) != 4 or zone[0] >= zone[1] or zone[2] >= zone[3]: raise ValueError('invalid zone')
        if not inside_xy(self.target, self.build_xy): raise ValueError('target outside configured build zone')


def inside_xy(point, zone):
    return zone[0] <= point[0] <= zone[1] and zone[2] <= point[1] <= zone[3]


def guard_move(current, target, config):
    if len(target) != 3 or not all(finite(x) for x in target): raise ValueError('invalid motion target')
    if not (-0.35 <= target[0] <= 0.35 and 0.10 <= target[1] <= 0.65 and 0.005 <= target[2] <= 0.30):
        raise ValueError('target outside experiment workspace')
    horizontal = math.hypot(target[0]-current[0], target[1]-current[1])
    if horizontal > 0.01 and min(current[2], target[2]) < config.transit_min_height_m:
        raise ValueError('horizontal transit below configured safe height')
