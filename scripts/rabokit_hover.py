#!/usr/bin/env python3
"""Plan or test a left-hand hover above the RGB stereo green brick, then return."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from rabokit_calibration_probe import measured, sample


def move_pose(arm, point, orientation):
    if not arm.move_to(*point, *orientation, blocking=False):
        raise RuntimeError('SDK rejected the waypoint')
    deadline = time.monotonic() + 40
    previous = None
    settled_since = None
    while time.monotonic() < deadline:
        state = measured(arm)
        error = math.dist(state['pose'][0], point)
        angle_error = max(abs((a - b + math.pi) % (2 * math.pi) - math.pi)
                          for a, b in zip(state['pose'][1], orientation))
        quiet = previous is not None and max(abs(a - b) for a, b in
                                            zip(state['joints'], previous['joints'])) < .0005
        if error < .003 and angle_error < .01 and quiet:
            if settled_since is None:
                settled_since = time.monotonic()
            elif time.monotonic() - settled_since >= .5:
                return state
        else:
            settled_since = None
        previous = state
        time.sleep(.1)
    raise TimeoutError('Measured tool pose did not reach the waypoint')


def return_joints(arm, joints):
    if not arm.move_joints(joints, speed=10, blocking=False):
        raise RuntimeError('SDK rejected the return command')
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if max(abs(a - b) for a, b in zip(measured(arm)['joints'], joints)) < .005:
            return
        time.sleep(.1)
    raise TimeoutError('Joint feedback did not return to the start')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--motion', action='store_true', help='Execute the checked hover and return in simulation')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run in the Rabokit cloud simulation terminal')
    if args.output.exists():
        parser.error('--output must be a new directory')
    args.output.mkdir(parents=True)
    vision = args.output / 'vision'
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('rabokit_stereo.py')),
                             '--output', str(vision)], capture_output=True, text=True, timeout=25)
    (args.output / 'vision.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    result.check_returncode()
    observation = json.loads((vision / 'stereo.json').read_text(encoding='utf-8'))
    green = [item for item in observation['matches'] if item['color'] == 'green']
    if observation['issues'] or len(green) != 1:
        raise RuntimeError('Expected a unique green surface point from current RGB stereo')
    from rabo_robolib import LbotDualArm

    robot = LbotDualArm(mode='sim')
    try:
        time.sleep(1)
        arm = robot.left_arm
        start = measured(arm)
        point = green[0]['colored_surface_point_sdk_m']
        hover_z = point[2] + .30 + green[0]['half_pixel_perturbation_radius_m']
        waypoints = [('retreat', (-.15, start['pose'][0][1], -.35)),
                     ('outside', (-.20, .60, hover_z)),
                     ('front', (.35, .60, hover_z)),
                     ('over_row', (.35, point[1], hover_z)),
                     ('hover', (point[0], point[1], hover_z))]
        orientation = (0., 0., 0.)
        checks = []
        previous = start['pose'][0]
        for label, target in waypoints:
            for fraction in (.5, 1):
                intermediate = [a + fraction * (b - a) for a, b in zip(previous, target)]
                checks.append({'stage': label, 'fraction': fraction, 'point': intermediate,
                               'result': arm.pose_check(*intermediate, *orientation)})
            previous = target
        report = {'motion_requested': args.motion, 'start': start, 'surface': green[0],
                  'waypoints': waypoints,
                  'orientation': orientation, 'reachability_checks': checks,
                  'path_note': 'SDK joint interpolation; sampled IK is not collision checking',
                  'completed_stages': [], 'physical_grasp_verified': False, 'official_score': None}

        def save():
            (args.output / 'hover.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

        save()
        print(json.dumps(report), flush=True)
        if not all(item['result'][0] for item in checks):
            raise RuntimeError('SDK reachability check rejected a waypoint')
        if args.motion:
            if not arm.enable():
                raise RuntimeError('Left arm enable failed')
            path = waypoints + [(label + '_return', target) for label, target in reversed(waypoints[:-1])]
            for label, target in path:
                state = move_pose(arm, target, orientation)
                report['completed_stages'].append({'stage': label, 'measured': state})
                save()
                print(json.dumps({'completed': label, 'measured': state}), flush=True)
                if label == 'hover':
                    sample(arm, args.output, label)
            return_joints(arm, start['joints'])
            sample(arm, args.output, 'restored')
            report['completed_stages'].append({'stage': 'restored', 'measured': measured(arm)})
            save()
        print('PASS: hover reachability' + (' and measured motion/return' if args.motion else ' (no motion)'))
    finally:
        try:
            if args.motion:
                robot.stop()
        finally:
            robot.shutdown()


if __name__ == '__main__':
    main()
