#!/usr/bin/env python3
"""Observe an empty hand at a high pose, with a +20 degree tilt or lateral peeks."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from rabokit_calibration_probe import hand_command, measured, sample
from rabokit_grasp import observe
from rabokit_hover import move_joint_pose
from rabokit_joint_path import pose_path, seeded_approach_plan, translation_path
from rabokit_pickup_vision import neighboring_layout


def inspection_plan(start, surface, reference, inverse, forward, peek_y=False):
    """Reuse the high entry, then inspect at fixed height without approaching contact."""
    import numpy as np

    point = surface['colored_surface_point_sdk_m']
    route, checks = seeded_approach_plan(start, point, surface['half_pixel_perturbation_radius_m'],
                                         .04, reference, 'approach', inverse, forward)
    aligned = next(i for i, event in enumerate(route) if event['name'] == 'aligned')
    route = [event for event in route[:aligned + 1] if 'joints' in event]
    for event in route:
        event['capture'] = False

    def extend(name, target=None):
        joints = route[-1]['joints']
        current = {'joints': joints, 'pose': [forward(joints)[:3, 3].tolist(), [0.] * 3]}
        path = (translation_path(current, [point[0] - .08, point[1], point[2] + .25], inverse, forward)
                if target is None else pose_path(current, target, inverse, forward))
        checks.append({'stage': name, **{k: v for k, v in path.items() if k != 'points'},
                       'point_count': len(path['points'])})
        if not path['continuous']:
            raise ValueError(f'{name}: {path.get("reason")} at step {path.get("failed_step")}')
        for i, item in enumerate(path['points']):
            route.append({'name': name if i == len(path['points']) - 1 else f'{name}_{i:03}',
                          'joints': item['joints'], 'capture': False})

    extend('high')
    high_index = len(route) - 1
    high = np.asarray(forward(route[-1]['joints'])).copy()
    if peek_y:
        for name, offset in (('peek_plus04', .04), ('peek_minus04', -.04), ('peek_minus08', -.08)):
            target = high.copy()
            target[1, 3] += offset
            extend(name, target)
    else:
        target = high.copy()
        c, s = math.cos(math.radians(20)), math.sin(math.radians(20))
        target[:3, :3] = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]]) @ target[:3, :3]
        extend('tilted', target)
    return route, high_index, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-record', type=Path, required=True)
    parser.add_argument('--peek-y', action='store_true',
                        help='Replace the tilt with fixed-height Y offsets +4, -4 and -8 cm')
    parser.add_argument('--motion', action='store_true', help='Execute empty-hand observation and recorded return')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run only in the Rabokit cloud simulation terminal')
    if args.output.exists():
        parser.error('--output must be a new directory')
    args.output.mkdir(parents=True)
    initial = observe(args.output / 'initial')
    green = [item for item in initial['matches'] if item['color'] == 'green']
    if initial['issues'] or len(green) != 1:
        raise RuntimeError('Expected a unique current RGB green surface before planning')
    from rabo_robolib import LbotDualArm
    from _robolib_core.arms import _lbot_kinematics as k

    robot = LbotDualArm(mode='sim')
    report = {'motion_requested': args.motion, 'physical_grasp_verified': False,
              'official_score': None, 'completed': [], 'returned': []}
    def save():
        (args.output / 'inspection.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        time.sleep(1)
        arm = robot.left_arm
        start = measured(arm)
        forward = lambda q: k.forward_kinematics('left', q)
        route, high_index, checks = inspection_plan(start, green[0],
            json.loads(args.seed_record.read_text(encoding='utf-8')),
            lambda t, q: k.inverse_kinematics('left', t, seed_real=q), forward, args.peek_y)
        if any(not low <= q <= high for event in route for q, (low, high)
               in zip(event['joints'], k.REAL_JOINT_LIMITS_LEFT)):
            raise ValueError('Inspection route exceeds SDK joint limits')
        report.update(start=start, source_surface=green[0], route=route, path_checks=checks, peek_y=args.peek_y,
                      scope='SDK kinematics and sampled tool paths; no whole-body collision proof')
        save()
        print(json.dumps({'route_points': len(route), 'path_checks': checks}), flush=True)
        if args.motion:
            if not arm.enable() or not robot.left_hand.set_force([1.] * 6):
                raise RuntimeError('Arm enable or hand force-limit setup failed')
            hand_command('open')
            time.sleep(1)
            for i, event in enumerate(route):
                report['active_stage'] = event['name']
                save()
                state = move_joint_pose(arm, event['joints'], forward)
                report['completed'].append({'name': event['name'], 'measured': state})
                save()
                if i == high_index:
                    sample(arm, args.output, 'high_open', stereo=True, side_camera=True)
                    hand_command('close_without_pinky')
                    time.sleep(1)
                    sample(arm, args.output, 'high_closed', stereo=True, side_camera=True)
                if args.peek_y and event['name'] in ('peek_plus04', 'peek_minus04', 'peek_minus08'):
                    sample(arm, args.output, event['name'] + '_closed', stereo=True, side_camera=True)
            if not args.peek_y:
                sample(arm, args.output, 'tilted_closed', stereo=True, side_camera=True)
            hand_command('open')
            time.sleep(1)
            sample(arm, args.output, 'peek_minus08_open' if args.peek_y else 'tilted_open',
                   stereo=True, side_camera=True)
            targets = list(reversed(report['completed'][:-1])) + [{'name': 'start', 'measured': start}]
            for event in targets:
                report['active_stage'] = event['name'] + '_return'
                save()
                state = move_joint_pose(arm, event['measured']['joints'], forward)
                report['returned'].append({'name': event['name'], 'measured': state})
                save()
            report['return_completed'] = True
            save()
            sample(arm, args.output, 'restored', stereo=True, side_camera=True)
            views = [observe(args.output / label, require_complete=False) for label in ('final', 'settled')]
            report['neighbor_observations_after_return'] = [neighboring_layout(initial, view) for view in views]
            save()
            if not all(item['consistent'] for item in report['neighbor_observations_after_return']):
                raise RuntimeError('Post-return RGB arrangement differs from the initial observation')
        print('PASS: inspection plan' + (' and measured empty-hand return' if args.motion else ' (no motion)'), flush=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        try:
            report['last_measured'] = measured(robot.left_arm)
        except RuntimeError:
            pass
        save()
        raise
    finally:
        try:
            if args.motion:
                robot.stop()
        finally:
            robot.shutdown()


if __name__ == '__main__':
    main()
