#!/usr/bin/env python3
"""Replay RGB and causal hand feedback locally (requires Pillow); no robot commands."""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path

from PIL import Image

from rabokit_hand import HAND_JOINTS


def stamp_ns(stamp):
    if isinstance(stamp, dict):
        stamp = [stamp['sec'], stamp['nanosec']]
    sec, nano = stamp
    if not isinstance(sec, int) or not isinstance(nano, int) or not 0 <= nano < 10**9:
        raise ValueError('Expected integer ROS seconds/nanoseconds')
    return sec * 10**9 + nano


def vector(sample, field):
    values = sample.get(field, {})
    return [float(values[name]) if isinstance(values.get(name), (int, float)) and
            math.isfinite(values[name]) else None for name in HAND_JOINTS]


def rgb_observation(folder, stamp):
    """Check locally available capture metadata/PNG, leaving missing pixels explicit."""
    result = {'image': str(folder / 'frame.png'), 'available': False}
    try:
        metadata = json.loads((folder / 'camera.json').read_text(encoding='utf-8'))
        if stamp_ns(metadata['stamp']) != stamp:
            raise ValueError('RGB metadata stamp differs from the paired frame')
        if metadata['encoding'] != 'rgb8' or not metadata['frame_id']:
            raise ValueError('Expected rgb8 with a camera frame')
        k = metadata['k']
        if len(k) != 9 or not all(math.isfinite(v) for v in k) or k[0] <= 0 or k[4] <= 0:
            raise ValueError('Expected finite calibrated camera intrinsics')
        with Image.open(folder / 'frame.png') as image:
            image.load()
            width, height = image.size
            if image.format != 'PNG' or image.mode != 'RGB':
                raise ValueError('Expected a decodable RGB PNG')
        if (width, height) != (metadata['width'], metadata['height']) or not width or not height:
            raise ValueError('PNG dimensions differ from camera metadata')
        result.update(available=True, width=width, height=height, frame_id=metadata['frame_id'],
                      k=metadata['k'])
    except (OSError, ValueError, KeyError) as error:
        result['reason'] = str(error)
    return result


def replay(trace, rgb_root, history_length=5, history_period=.08, max_age=.04):
    if not isinstance(history_length, int) or not 1 <= history_length <= 100:
        raise ValueError('History length must be in [1,100]')
    if any(not math.isfinite(v) or v <= 0 for v in (history_period, max_age)):
        raise ValueError('History period and maximum feedback age must be positive and finite')
    samples = sorted(trace['joint_samples'], key=lambda row: stamp_ns(row['stamp']))
    stamps = [stamp_ns(row['stamp']) for row in samples]
    if len(set(stamps)) != len(stamps):
        raise ValueError('Ambiguous duplicate JointState stamps')
    rgb = trace['rgb_trace']
    if rgb['cameras'] != ['base_cam', 'base_cam2']:
        raise ValueError('Expected the captured fixed stereo camera pair')
    period_ns, max_age_ns = round(history_period * 10**9), round(max_age * 10**9)
    rows = []
    for frame in rgb['frames']:
        stamp = stamp_ns(frame['stamp'])
        history, reasons = [], []
        for offset in range(history_length - 1, -1, -1):
            requested = stamp - offset * period_ns
            index = bisect_right(stamps, requested) - 1
            if index < 0 or requested - stamps[index] > max_age_ns:
                reasons.append('Insufficient recent causal hand history')
                history.append(None)
                continue
            sample = samples[index]
            fields = {key: vector(sample, key) for key in ('position', 'velocity', 'effort')}
            if any(value is None for values in fields.values() for value in values):
                reasons.append('Missing or nonfinite hand feedback channel')
            history.append({'requested_stamp_ns': requested, 'sample_stamp_ns': stamps[index],
                            'age_seconds': (requested - stamps[index]) / 10**9, **fields})
        folder = Path(frame['directory'])
        if folder.is_absolute() or len(folder.parts) != 1 or folder.name in ('.', '..'):
            raise ValueError('Expected one captured RGB frame directory name')
        cameras = {name: rgb_observation(Path(rgb_root) / folder / name, stamp) for name in rgb['cameras']}
        if not all(view['available'] for view in cameras.values()):
            reasons.append('Local RGB pixels/metadata unavailable or inconsistent')
        rows.append({'stamp_ns': stamp, 'relative_command_seconds': frame['relative_command_seconds'],
                     'cameras': cameras, 'history': history, 'usable_observation': not reasons,
                     'issues': sorted(set(reasons))})
    return {'joint_names': list(HAND_JOINTS), 'history_length': history_length,
            'history_period_seconds': history_period, 'max_feedback_age_seconds': max_age,
            'position_units': 'radian', 'velocity_units': 'radian/second',
            'effort_units': 'uncalibrated reported effort',
            'command_error_available': False,
            'scope': 'Captured observations only; no policy, action labels, contact identity or grasp-success labels. '
                     'Stage endpoints are not the issued per-step command or an internal servo reference.',
            'trace_command_complete': trace.get('command_complete', False),
            'frame_count': len(rows), 'usable_observation_count': sum(row['usable_observation'] for row in rows),
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, help='Saved *-command.json or static hand-feedback.json containing JointState and RGB metadata')
    parser.add_argument('--rgb-root', type=Path, help='Local *-rgb folder, or parent of the camera folder for static hand-feedback.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rgb_root = args.rgb_root or (args.trace.parent.parent if args.trace.name == 'hand-feedback.json' else
                                args.trace.with_name(args.trace.name.removesuffix('-command.json') + '-rgb'))
    try:
        report = replay(json.loads(args.trace.read_text(encoding='utf-8')), rgb_root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Replayed {report["frame_count"]} frame pairs; '
          f'{report["usable_observation_count"]} have local RGB and complete causal feedback history')


if __name__ == '__main__':
    main()
