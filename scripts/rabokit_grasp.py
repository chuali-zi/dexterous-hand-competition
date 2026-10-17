#!/usr/bin/env python3
"""Plan or test one RGB-guided green-brick pickup and placement in simulation."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from rabokit_calibration_probe import hand_command, measured, sample
from rabokit_hover import hover_waypoints, move_pose, return_joints

TOP_GRASP_RPY = (math.pi / 2, 0., math.pi)
ZERO_RPY = (0., 0., 0.)


def make_plan(start, point, uncertainty, stage, clearance=.03):
    height = point[2] + .30 + uncertainty
    transit = hover_waypoints(start, point, height)
    destination = (point[0], point[1] - .12, point[2])
    events = []

    def move(name, position, orientation, capture=False):
        events.append({'name': name, 'position': position, 'orientation': orientation, 'capture': capture})

    def hand(name, pose):
        events.append({'name': name, 'hand': pose, 'capture': True})

    hand('opened', 'open')
    for name, position in transit:
        move(name, position, ZERO_RPY)
    move('oriented', transit[-1][1], TOP_GRASP_RPY, True)
    grasp_xy = (point[0], point[1] + .08)
    if stage in ('approach', 'grasp', 'plan'):
        move('aligned', (*grasp_xy, height), TOP_GRASP_RPY)
        move('approached', (*grasp_xy, point[2] + .12), TOP_GRASP_RPY, True)
    if stage in ('grasp', 'plan'):
        move('contact', (*grasp_xy, point[2] + clearance), TOP_GRASP_RPY, True)
        hand('closed', 'close')
        move('lifted', (*grasp_xy, point[2] + .20), TOP_GRASP_RPY, True)
        placed_xy = (destination[0], destination[1] + .08)
        move('transferred', (*placed_xy, point[2] + .20), TOP_GRASP_RPY, True)
        move('lowered', (*placed_xy, point[2] + clearance), TOP_GRASP_RPY, True)
        hand('released', 'open')
        move('withdrawn', (*placed_xy, height), TOP_GRASP_RPY, True)
    if stage in ('approach', 'grasp', 'plan'):
        move('aligned_return', (*grasp_xy, height), TOP_GRASP_RPY)
    move('oriented_return', transit[-1][1], TOP_GRASP_RPY)
    move('vertical_return', transit[-1][1], ZERO_RPY)
    for name, position in reversed(transit[:-1]):
        move(name + '_return', position, ZERO_RPY)
    return events, destination


def observe(output):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('rabokit_stereo.py')),
                             '--output', str(output)], capture_output=True, text=True, timeout=25)
    output.with_suffix('.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    result.check_returncode()
    return json.loads((output / 'stereo.json').read_text(encoding='utf-8'))


def check_plan(arm, start, events):
    previous, previous_rpy = start['pose']
    checks = []
    for event in events:
        if 'position' not in event:
            continue
        target, rpy = event['position'], event['orientation']
        checks.append({'stage': event['name'], 'point': target, 'orientation': rpy,
                       'result': arm.pose_check(*target, *rpy)})
        if tuple(previous_rpy) == tuple(rpy):
            midpoint = [(a + b) / 2 for a, b in zip(previous, target)]
            checks.append({'stage': event['name'] + '_midpoint', 'point': midpoint,
                           'orientation': rpy, 'result': arm.pose_check(*midpoint, *rpy)})
        previous, previous_rpy = target, rpy
    return checks


def relocation(observation, destination, source):
    green = [item for item in observation['matches'] if item['color'] == 'green']
    if observation['issues'] or len(green) != 1:
        return {'relocation_verified': False, 'reason': 'green surface not uniquely observed'}
    point = green[0]['colored_surface_point_sdk_m']
    planar_error = math.dist(point[:2], destination[:2])
    height_error = abs(point[2] - destination[2])
    moved = math.dist(point[:2], source[:2])
    return {'observed_surface': point, 'destination_planar_error_m': planar_error,
            'destination_height_error_m': height_error, 'source_planar_displacement_m': moved,
            'relocation_verified': planar_error < .035 and height_error < .035 and moved > .08}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('plan', 'orient', 'approach', 'grasp'), default='plan')
    parser.add_argument('--tool-clearance', type=float, default=.03,
                        help='Contact tool height above the observed colored surface, metres')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run only in the Rabokit simulation terminal')
    if not math.isfinite(args.tool_clearance) or not 0 < args.tool_clearance <= .10:
        parser.error('--tool-clearance must be in (0, .10]')
    if args.output.exists():
        parser.error('--output must be a new directory')
    args.output.mkdir(parents=True)
    initial = observe(args.output / 'initial')
    green = [item for item in initial['matches'] if item['color'] == 'green']
    if initial['issues'] or len(green) != 1:
        raise RuntimeError('Expected a unique current RGB green surface before motion')
    surface = green[0]
    point = surface['colored_surface_point_sdk_m']
    from rabo_robolib import LbotDualArm

    robot = LbotDualArm(mode='sim')
    motion = args.stage != 'plan'
    try:
        time.sleep(1)
        arm = robot.left_arm
        start = measured(arm)
        events, destination = make_plan(start, point, surface['half_pixel_perturbation_radius_m'],
                                        args.stage, args.tool_clearance)
        if any(math.dist(item['colored_surface_point_sdk_m'][:2], destination[:2]) < .07
               for item in initial['matches'] if item['color'] != 'green'):
            raise RuntimeError('The RGB destination overlaps another observed brick')
        checks = check_plan(arm, start, events)
        report = {'stage': args.stage, 'start': start, 'source_surface': surface,
                  'destination_surface': destination, 'events': events, 'reachability_checks': checks,
                  'tool_clearance_m': args.tool_clearance, 'completed': [], 'physical_grasp_verified': False,
                  'official_score': None, 'path_note': 'SDK joint interpolation; sampled IK is not collision checking'}

        def save():
            (args.output / 'grasp.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

        save()
        print(json.dumps({'stage': args.stage, 'events': events, 'checks': checks}), flush=True)
        if not all(item['result'][0] for item in checks):
            raise RuntimeError('SDK rejected a planned waypoint; no motion started')
        if motion:
            if not arm.enable() or not robot.left_hand.set_force([1.] * 6):
                raise RuntimeError('Arm enable or hand force-limit setup failed')
            for event in events:
                if 'hand' in event:
                    hand_command(event['hand'])
                    time.sleep(1)
                    state = measured(arm)
                else:
                    state = move_pose(arm, event['position'], event['orientation'])
                report['completed'].append({'name': event['name'], 'measured': state})
                save()
                print(json.dumps({'completed': event['name'], 'measured': state}), flush=True)
                if event['capture']:
                    sample(arm, args.output, event['name'], stereo=True)
            return_joints(arm, start['joints'])
            sample(arm, args.output, 'restored', stereo=True)
            report['completed'].append({'name': 'restored', 'measured': measured(arm)})
            report['motion_sequence_completed'] = True
            save()
            if args.stage == 'grasp':
                final = observe(args.output / 'final')
                settled = observe(args.output / 'settled')
                report['final_observation'] = relocation(final, destination, point)
                report['settled_observation'] = relocation(settled, destination, point)
                positions = [report[key].get('observed_surface') for key in
                             ('final_observation', 'settled_observation')]
                report['release_observations_consistent'] = (
                    all(positions) and math.dist(*positions) < .02)
                save()
                if not (report['final_observation']['relocation_verified'] and
                        report['settled_observation']['relocation_verified'] and
                        report['release_observations_consistent']):
                    raise RuntimeError('Motion completed, but RGB pickup/placement relocation was not verified')
                print('PASS: RGB relocation in two separated observations (official score unmeasured)', flush=True)
        print(f'PASS: {args.stage} motion plan' + (' executed with measured return' if motion else ' (no motion)'), flush=True)
    finally:
        try:
            if motion:
                robot.stop()
        finally:
            robot.shutdown()


if __name__ == '__main__':
    main()
