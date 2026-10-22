#!/usr/bin/env python3
"""Compare local hand replay with retained feedback; never connects to the platform.

Fit an algebraic reported-effort law only for joints whose published targets stay
constant. Coefficients are not calibrated physical torques or contact forces.
Trajectory alignment uses publication TF references, not servo application times.
Optional RGB comparison also requires Pillow (requirements-dynamics.txt).
"""
import argparse
from bisect import bisect_right
import itertools
import json
from pathlib import Path

import numpy as np

from rabokit_hand import HAND_JOINTS


def seconds(sample):
    sec, nano = sample['stamp']
    return sec + nano * 1e-9


def effort_design(traces):
    rows, efforts, counts = [], [], {}
    for stage, trace in traces.items():
        commands = trace['issued_commands']
        times = [c['robot_tf_reference_stamp_sec'] for c in commands]
        if not times or not np.isfinite(times).all() or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Expected ordered finite publication TF references')
        held = {}
        for name in HAND_JOINTS:
            values = np.array([c['targets_rad'][name] for c in commands])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite published target')
            if np.ptp(values) <= 1e-12:
                held[name] = float(values[0])
        stage_counts = {n: 0 for n in held}
        for sample in trace['joint_samples']:
            # Avoid the immediate onset; unchanged publication still cannot prove application time.
            if seconds(sample) < times[0] + .1:
                continue
            for name, target in held.items():
                row = [target - sample['position'][name], -sample['velocity'][name]]
                effort = sample['effort'][name]
                if not np.isfinite([*row, effort]).all():
                    raise ValueError('Nonfinite retained feedback')
                rows.append(row)
                efforts.append(effort)
                stage_counts[name] += 1
        counts[stage] = stage_counts
    return np.array(rows).reshape(-1, 2), np.array(efforts), counts


def fit_effort(traces):
    design, efforts, counts = effort_design(traces)
    coefficients, _, rank, singular = np.linalg.lstsq(design, efforts, rcond=None)
    if rank != 2:
        raise ValueError('Both position-error and velocity terms must be identifiable')
    residual = design @ coefficients - efforts
    return {'law': 'reported_effort = kp * (constant_published_target - position) - kv * velocity',
            'kp': float(coefficients[0]), 'kv': float(coefficients[1]), 'samples': len(efforts),
            'rmse': float(np.sqrt(np.mean(residual ** 2))), 'maximum_absolute_residual': float(np.max(np.abs(residual))),
            'design_condition_number': float(singular[0] / singular[-1]), 'joint_samples_by_stage': counts,
            'physical_torque_units_verified': False}


def compare_trajectory(traces, proxy, max_age=.025, origin=None):
    """Use the preceding proxy sample; do not bridge gaps or synthetic lift periods."""
    if origin is None:
        origin = seconds(next(iter(traces.values()))['joint_samples'][0])
    rows = proxy['rows']
    times = [row['seconds'] for row in rows]
    if (not times or not np.isfinite(times).all() or any(b <= a for a, b in zip(times, times[1:]))
            or not np.isfinite(max_age) or max_age <= 0):
        raise ValueError('Expected ordered finite proxy times and positive maximum age')
    result = {}
    for stage, trace in traces.items():
        errors, ages = [], []
        for sample in trace['joint_samples']:
            time = seconds(sample) - origin
            index = bisect_right(times, time) - 1
            if index < 0 or time - times[index] > max_age or rows[index]['tool_rise_m'] != 0:
                continue
            row = rows[index]
            delta = [row['hand_position_rad'][n] - sample['position'][n] for n in HAND_JOINTS]
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite compared position')
            errors.append(delta)
            ages.append(time - times[index])
        if not errors:
            result[stage] = {'matched': 0, 'unmatched': len(trace['joint_samples'])}
            continue
        array = np.array(errors)
        result[stage] = {'matched': len(errors), 'unmatched': len(trace['joint_samples']) - len(errors),
                         'maximum_proxy_age_seconds': float(max(ages)),
                         'rmse_rad': float(np.sqrt(np.mean(array ** 2))),
                         'joint_rmse_rad': {n: float(v) for n, v in zip(HAND_JOINTS, np.sqrt(np.mean(array ** 2, axis=0)))},
                         'maximum_absolute_error_rad': float(np.max(np.abs(array)))}
    return result


def cube_pixels(row, sdk, camera):
    """Project the proxy's full oriented 50 mm cube, including roll and pitch."""
    from scipy.spatial.transform import Rotation
    center = np.array(row['local_target_position'])
    q = np.array(row['local_target_quaternion_wxyz'])
    corners = np.array(list(itertools.product((-.025, .025), repeat=3)))
    points = corners @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T + center
    sr, sp = sdk
    cr, cp, k = camera
    optical = ((points @ sr.T + sp) - cp) @ cr
    if not np.isfinite(optical).all() or np.any(optical[:, 2] <= 0):
        raise ValueError('Nonfinite cube or cube behind camera')
    uv = optical @ k.T
    return uv[:, :2] / uv[:, 2, None]


def projection_constraints(uv, observation):
    """Visible green must be inside the silhouette; visible wood must be outside."""
    from scipy.spatial import ConvexHull
    green, wood = observation
    result = {'green_samples': len(green), 'wood_samples': len(wood), 'available': False}
    if min(len(green), len(wood)) < 3:
        return result
    eq = ConvexHull(uv).equations
    gd, wd = (-np.max(points @ eq[:, :2].T + eq[:, 2], axis=1) for points in (green, wood))
    result.update(available=True, maximum_green_miss_px=float(max(0, -gd.min())),
                  maximum_wood_intrusion_px=float(max(0, wd.max())),
                  squared_cost=float(np.mean(np.maximum(0, .5 - gd) ** 2)
                                     + np.mean(np.maximum(0, .5 + wd) ** 2)))
    return result


def compare_rgb(trial, proxy, stages):
    from PIL import Image
    from analyze_occluded_brick import visible_pixels
    from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT, world_transform
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    frames = read(trial / 'contact/robot-frames.json')['frames']
    sr, lp = world_transform(frames, LEFT_ROOT)
    _, rp = world_transform(frames, RIGHT_ROOT)
    sdk = np.array(sr), (np.array(lp) + rp) / 2
    source = read(trial / 'grasp.json')['source_surface']['colored_surface_point_sdk_m']
    baseline = {'local_target_position': np.array(source) - [0, 0, .025],
                'local_target_quaternion_wxyz': [1., 0., 0., 0.]}
    cameras, references, rois = {}, {}, {}
    for name in ['base_cam', 'base_cam2', 'base_cam3']:
        info = read(trial / 'contact/camera' / name / 'camera.json')
        cr, cp = world_transform(frames, info['frame_id'])
        cameras[name] = np.array(cr), np.array(cp), np.array(info['k']).reshape(3, 3)
        references[name] = info
        uv = cube_pixels(baseline, sdk, cameras[name])
        lo, hi = np.floor(uv.min(axis=0) - 20).astype(int), np.ceil(uv.max(axis=0) + 20).astype(int)
        rois[name] = [max(0, int(lo[0])), max(0, int(lo[1])),
                      min(info['width'], int(hi[0])), min(info['height'], int(hi[1]))]
    times = [row['seconds'] for row in proxy['rows']]
    result = []
    for stage in stages:
        metadata = {n: read(trial / stage / 'camera' / n / 'camera.json') for n in cameras}
        stamps = [seconds({'stamp': [m['stamp']['sec'], m['stamp']['nanosec']]}) for m in metadata.values()]
        if len(set(stamps)) != 1:
            raise ValueError('Expected synchronized RGB')
        relative = stamps[0] - proxy['reference_origin_stamp_sec']
        index = bisect_right(times, relative) - 1
        item = {'stage': stage, 'stamp_sec': stamps[0], 'available': False}
        if (index < 0 or relative - times[index] > .025
                or relative >= proxy['synthetic_lift_start_seconds']):
            item['reason'] = 'No preceding stationary proxy sample within 25 ms'
            result.append(item)
            continue
        row = proxy['rows'][index]
        item.update(available=True, proxy_age_seconds=relative - times[index], cameras={},
                    local_target_position=row['local_target_position'],
                    local_target_quaternion_wxyz=row['local_target_quaternion_wxyz'])
        for name, camera in cameras.items():
            info, ref = metadata[name], references[name]
            if any(info[k] != ref[k] for k in ['frame_id', 'k', 'width', 'height']):
                raise ValueError('Camera identity or intrinsics changed')
            with Image.open(trial / stage / 'camera' / name / 'frame.png') as image:
                image.load()
                if image.mode != 'RGB' or image.size != (ref['width'], ref['height']):
                    raise ValueError('Expected raw RGB with reference dimensions')
                observation = visible_pixels([np.array(image)], [rois[name]], 160)[0]
            uv = cube_pixels(row, sdk, camera)
            item['cameras'][name] = {'projected_vertices': uv.tolist(),
                                    **projection_constraints(uv, observation)}
        result.append(item)
    return {'scope': 'Saved RGB silhouette constraints on the full proxy pose; no native object truth, '
                     'occluded-surface reconstruction or physical grasp verification.',
            'rois': rois, 'green_threshold': 160, 'rows': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--proxy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rgb', action='store_true', help='Compare the stationary proxy cube with saved three-camera RGB')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new file')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    proxy = read(args.proxy)
    if Path(proxy['trial']).resolve() != args.trial.resolve():
        parser.error('Proxy and retained trial paths differ')
    stages = ['empty_closed'] if proxy['parameters']['empty'] else ['thumb_ready', 'thumb_opposed', 'closed']
    traces = {stage: read(args.trial / (stage + '-command.json')) for stage in stages}
    fit_traces = {stage: read(args.trial / (stage + '-command.json'))
                  for stage in ['thumb_ready', 'thumb_opposed', 'closed']}
    result = {'scope': __doc__, 'trial': str(args.trial), 'proxy': str(args.proxy),
              'effort_fit': fit_effort(fit_traces), 'command_window_positions': compare_trajectory(traces, proxy),
              'platform_physics_calibrated': False, 'physical_grasp_verified': False}
    if not proxy['parameters']['empty']:
        static = {stage: {'joint_samples': read(args.trial / stage / 'robot-frames.json')['hand_joint_states']}
                  for stage in stages}
        result['static_window_positions'] = compare_trajectory(
            static, proxy, origin=seconds(traces[stages[0]]['joint_samples'][0]))
    if args.rgb:
        if proxy['parameters']['empty']:
            parser.error('RGB cube comparison requires a contact replay')
        result['rgb_constraints'] = compare_rgb(args.trial, proxy, stages)
        rgb_feedback = {stage: read(args.trial / stage / 'camera/hand-feedback.json') for stage in stages}
        result['rgb_window_positions'] = compare_trajectory(
            rgb_feedback, proxy, origin=proxy['reference_origin_stamp_sec'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output),
                      'effort_fit': {k: result['effort_fit'][k] for k in ['kp', 'kv', 'samples', 'rmse']},
                      'position_rmse_rad': {s: v.get('rmse_rad') for s, v in result['command_window_positions'].items()}}))


if __name__ == '__main__':
    main()
