#!/usr/bin/env python3
"""Compare local hand replay with retained feedback; never connects to the platform.

Fit an algebraic reported-effort law only for joints whose published targets stay
constant. Coefficients are not calibrated physical torques or contact forces.
Trajectory alignment uses publication TF references, not servo application times.
"""
import argparse
from bisect import bisect_right
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--proxy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output),
                      'effort_fit': {k: result['effort_fit'][k] for k in ['kp', 'kv', 'samples', 'rmse']},
                      'position_rmse_rad': {s: v.get('rmse_rad') for s, v in result['command_window_positions'].items()}}))


if __name__ == '__main__':
    main()
