"""Numerical SDK joint route for a fixed-shape table slide, without ROS or motion.

Reuses the recorded empty approach to its high waypoint, then checks the new
connection, table-slide route and high return. Whole-arm collision remains
unverified; no robot object, ROS initialization or movement is used.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rabokit_joint_path import centered_inverse, pose_path
from try_table_slide import command_offset


def check_empty_segments(segments, hand_targets, mount, trace):
    """Static local hand/table/brick checks for new empty travel only."""
    sys.path.insert(0, str(Path('runs/mujoco-runtime').resolve()))
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(Path('runs/bimanual-contact-68/local-calibrated-contact.xml').resolve()))
    data = mujoco.MjData(model)
    left = int(model.jnt_qposadr[model.joint('hand_root_free').id])
    right = int(model.jnt_qposadr[model.joint('right_hand_root_free').id])
    data.qpos[right + 2] += .5
    target_body = trace['candidate'].get('target_body', 'target')
    target = int(model.jnt_qposadr[model.body(target_body).jntadr[0]])
    for name, value in hand_targets.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    checks = []
    for name, samples, forward in segments:
        if name.startswith('return'):
            data.qpos[target:target + 3] = trace['rows'][-1]['target_position']
            data.qpos[target + 3:target + 7] = trace['rows'][-1]['target_quaternion_wxyz']
        collisions, roots = [], []
        for joints in samples:
            root = forward(joints) @ mount
            roots.append(root[:3, 3])
            data.qpos[left:left + 3] = root[:3, 3]
            data.qpos[left + 3:left + 7] = Rotation.from_matrix(root[:3, :3]).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                pair = [model.geom(int(g)).name for g in contact.geom]
                if any(n.startswith('lh_') for n in pair):
                    collisions.append(dict(pair=pair, distance_m=float(contact.dist)))
        checks.append(dict(phase=name, samples=len(roots), clear=not collisions,
                           hand_root_minimum=np.min(roots, axis=0).tolist(),
                           hand_root_maximum=np.max(roots, axis=0).tolist(),
                           contacts=collisions[:20], contact_sample_count=len(collisions)))
    return dict(clear=all(c['clear'] for c in checks), target_body=target_body, checks=checks,
                scope='Local hand collision only, fixed parked right hand; no full-arm collision claim')


def interpolated_joints(points):
    samples = [points[0]]
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a), np.asarray(b)
        count = max(1, math.ceil(float(np.max(np.abs(b - a))) / .005))
        samples.extend((a + (b - a) * t / count).tolist() for t in range(1, count + 1))
    return samples


def run(trace_path, output, sdk_cache):
    sys.path.insert(0, str(sdk_cache))
    from _robolib_core.arms import _lbot_kinematics as sdk
    trace = json.loads(trace_path.read_text())
    candidate = trace['candidate']
    withdraw_height = candidate.get('withdraw_height_m', .08)
    push_sign = candidate.get('push_sign', -1)
    push_distance = candidate.get('push_distance_m', .12)
    release_order = candidate.get('release_order', 'retreat_then_lift')
    if release_order not in ('retreat_then_lift', 'lift_then_retreat'):
        raise ValueError('Unsupported release order')
    lift_first = release_order == 'lift_then_retreat'
    def offset_at(seconds):
        return command_offset(seconds, withdraw_height, push_sign, push_distance, lift_first)
    old = json.loads(Path('runs/arm-feasible-74/opposition/asymmetric-contact-candidate.json').read_text())
    limits = np.asarray(json.loads(Path('runs/two-contact-73/reachability/endpoint-results.json').read_text())['sides']['left']['intersection_limits'])
    mount = np.asarray(json.loads(Path('configs/bimanual-contact.json').read_text())['tool_to_hand']['left'])
    inv_mount = np.linalg.inv(mount)
    root = np.eye(4)
    root[:3, 3] = candidate['hand_root_position']
    root[:3, :3] = Rotation.from_quat(np.asarray(candidate['hand_root_quaternion_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    forward = lambda q: np.asarray(sdk.forward_kinematics('left', list(q)))
    stats = {}
    inverse = lambda target, q: centered_inverse(target, q, forward, limits, stats)
    started = time.monotonic()
    contact_seed = candidate.get('contact_joints_sdk', old['arm_joints_sdk']['left'])
    q = inverse(root @ inv_mount, contact_seed)
    if q is None:
        result = {'continuous': False, 'failure': 'No nearby contact-pose IK from retained left-arm seed'}
    else:
        contact_q = list(q)

        def segment(joints, xyz):
            target = root.copy()
            target[:3, 3] = xyz
            actual = forward(joints)
            return pose_path(dict(joints=list(joints), pose=[actual[:3, 3].tolist(), [0, 0, 0]]),
                             target @ inv_mount, inverse, forward, spacing=.005, refine_once=True)

        reverse = []
        for seconds in [3., .5]:
            offset, phase = offset_at(seconds)
            path = segment(q, root[:3, 3] + offset)
            reverse.append(dict(seconds=seconds, phase=phase, **path))
            if not path['continuous']:
                break
            q = path['points'][-1]['joints']
        stages, all_points = [], []
        complete = all(p['continuous'] for p in reverse)
        staging_q = list(q) if complete else None
        if complete:
            all_points.append(dict(seconds=0., phase='staging', joints=list(q)))
            previous_seconds = .5
            release_stages = ([(10., 'withdraw'), (11., 'retreat')] if lift_first else
                              [(9., 'retreat'), (11., 'withdraw')])
            for seconds, phase in [(3., 'descend'), (4., 'approach'), (8., 'push')] + release_stages:
                offset, _ = offset_at(seconds)
                origin = (forward(q) @ mount)[:3, 3]
                end = root[:3, 3] + offset
                path = segment(q, end)
                stages.append(dict(start_seconds=previous_seconds, end_seconds=seconds, phase=phase, **path))
                for point in path['points']:
                    actual_root = forward(point['joints']) @ mount
                    denominator = float((end - origin) @ (end - origin))
                    fraction = float(np.clip((actual_root[:3, 3] - origin) @ (end - origin) / denominator, 0, 1)) if denominator else 1.
                    all_points.append(dict(seconds=previous_seconds + fraction * (seconds - previous_seconds),
                                           phase=phase, joints=point['joints'],
                                           hand_root_position=actual_root[:3, 3].tolist(),
                                           hand_root_quaternion_wxyz=Rotation.from_matrix(actual_root[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()))
                if not path['continuous']:
                    complete = False
                    break
                q = path['points'][-1]['joints']
                previous_seconds = seconds
        values = [p['joints'] for p in all_points]
        segment_points = [p for s in stages for p in s['points']]
        result = dict(continuous=complete, side='left', contact_joints_sdk=contact_q,
                      staging_joints_sdk=staging_q, joint_limits_sdk=limits.tolist(), hand_targets=candidate['hand_targets'],
                      reverse_staging=reverse, stages=stages, points=all_points,
                      minimum_joint_margin_rad=float(min(np.minimum(np.array(v) - limits[:, 0], limits[:, 1] - np.array(v)).min() for v in values)) if values else None,
                      maximum_joint_step_rad=max((p['maximum_joint_step_rad'] for p in segment_points), default=None),
                      maximum_sampled_position_error_m=max((p['sampled_position_error_m'] for p in segment_points), default=None),
                      maximum_sampled_orientation_error_rad=max((p['sampled_orientation_error_rad'] for p in segment_points), default=None), stats=stats)
        if complete:
            history = json.loads(Path('configs/bimanual-empty-approach.json').read_text())
            prefix = history['left']['points'][:86]
            previous = np.asarray(prefix[-1])
            goal = np.asarray(staging_q)
            count = max(1, math.ceil(float(np.max(np.abs(goal - previous))) / .1))
            bridge = [(previous + (goal - previous) * i / count).tolist()
                      for i in range(1, count + 1)]
            result['neutral'] = history['left']['neutral']
            result['lead_in'] = dict(
                history_source='configs/bimanual-empty-approach.json',
                history_last_index=85, history_native_empty_execution=history['native_empty_execution'],
                set_hand_targets_after_history=True, history_points=prefix, connection_points=bridge,
                points=[dict(phase='historical_empty_approach', joints=p) for p in prefix] +
                       [dict(phase='high_connection', joints=p) for p in bridge],
                maximum_connection_joint_step_rad=float(np.max(np.abs(goal - previous))) / count)
            result['return_stages'] = []
            staging_offset, _ = offset_at(.5)
            above_source_offset = staging_offset.copy()
            above_source_offset[2] = withdraw_height
            for phase, offset in [('return_above_source', above_source_offset),
                                  ('return_to_staging', staging_offset)]:
                path = segment(q, root[:3, 3] + np.array(offset))
                result['return_stages'].append(dict(phase=phase, **path))
                if not path['continuous']:
                    break
                q = path['points'][-1]['joints']
            result['return_continuous'] = all(p['continuous'] for p in result['return_stages'])
            # The same high Cartesian pose can land on a different redundant-arm
            # branch; return through a checked joint connection, not a joint jump.
            previous = np.asarray(q)
            goal = np.asarray(prefix[-1])
            count = max(1, math.ceil(float(np.max(np.abs(goal - previous))) / .1))
            result['return_high_connection'] = [(previous + (goal - previous) * i / count).tolist()
                                                for i in range(1, count + 1)]
            result['return_historical_points'] = list(reversed(prefix[:-1])) + [result['neutral']]
            margins = [(float(np.minimum(np.asarray(p['joints']) - limits[:, 0],
                                         limits[:, 1] - np.asarray(p['joints'])).min()), p)
                       for p in all_points]
            margin, point = min(margins, key=lambda item: item[0])
            result['minimum_margin_location'] = dict(phase=point['phase'], seconds=point['seconds'], margin_rad=margin)
            returned = [all_points[-1]['joints']] + [p['joints'] for s in result['return_stages'] for p in s['points']]
            result['new_empty_travel_collision'] = check_empty_segments([
                ('high_connection', interpolated_joints([prefix[-1]] + bridge), forward),
                ('return_above_source_and_staging', interpolated_joints(returned), forward),
                ('return_high_connection', interpolated_joints([returned[-1]] + result['return_high_connection']), forward),
            ], candidate['hand_targets'], mount, trace)
            full = ([result['neutral']] + prefix + bridge + values + returned +
                    result['return_high_connection'] + result['return_historical_points'])
            result['minimum_complete_route_margin_rad'] = float(min(np.minimum(np.asarray(p) - limits[:, 0], limits[:, 1] - np.asarray(p)).min() for p in full))
            result['maximum_complete_route_joint_step_rad'] = float(np.max(np.abs(np.diff(np.asarray(full), axis=0))))
            result['numerical_empty_travel_pass'] = bool(result['return_continuous'] and result['new_empty_travel_collision']['clear'])
    result.update(source_trace=str(trace_path), candidate=candidate,
                  trajectory_parameters=dict(push_sign=push_sign, push_distance_m=push_distance,
                                             withdraw_height_m=withdraw_height,
                                             release_order=release_order,
                                             target_body=candidate.get('target_body', 'target')),
                  wall_seconds=time.monotonic() - started,
                  scope=__doc__, motion_commanded=False, native_to_staging_checked=False,
                  historical_to_staging_numerically_checked=bool(result.get('numerical_empty_travel_pass')),
                  whole_arm_collision_checked=False, idle_right_raise_is_native_pose=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=Path('runs/simple-transfer-76/slide/center-two-low-withdraw.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/simple-transfer-76/route/center-two-low-withdraw.json'))
    parser.add_argument('--sdk-cache', type=Path, default=Path.home() / '.rabo_robolib/core-cp312.zip')
    args = parser.parse_args()
    result = run(args.trace, args.output, args.sdk_cache)
    print(json.dumps({k: v for k, v in result.items() if k not in ['stages', 'points', 'reverse_staging', 'stats', 'lead_in', 'return_stages', 'return_high_connection', 'return_historical_points']}, indent=2))
    raise SystemExit(0 if result['continuous'] and result.get('numerical_empty_travel_pass') and
                          result['minimum_complete_route_margin_rad'] >= 0 and
                          result['maximum_complete_route_joint_step_rad'] <= .15 else 1)
