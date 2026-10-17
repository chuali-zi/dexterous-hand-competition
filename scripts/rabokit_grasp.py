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
from rabokit_hover import hover_waypoints, move_joint_pose, move_pose, return_joints
from rabokit_pickup_vision import neighboring_layout, pickup_evidence, saved_scene, target_observation

TOP_GRASP_RPY = (math.pi / 2, 0., -math.pi / 2)
ZERO_RPY = (0., 0., 0.)


def make_plan(start, point, uncertainty, stage, clearance=.03, yaw=-math.pi / 2):
    height = point[2] + .30 + uncertainty
    transit = hover_waypoints(start, point, height)
    destination = (point[0], point[1] - .12, point[2])
    events = []
    orientation = (math.pi / 2, 0., yaw)
    direction = (-math.sin(yaw), math.cos(yaw))

    def move(name, position, orientation, capture=False):
        events.append({'name': name, 'position': position, 'orientation': orientation, 'capture': capture})

    def hand(name, pose):
        events.append({'name': name, 'hand': pose, 'capture': True})

    def reverse(name, original, capture=False):
        source = next(event for event in events if event['name'] == original)
        events.append({'name': name, 'position': source['position'], 'orientation': source['orientation'],
                       'reverse_of': original, 'capture': capture})

    hand('opened', 'open')
    for name, position in transit:
        move(name, position, ZERO_RPY if name in ('retreat', 'outside', 'front') else orientation, name == 'hover')
        if name == 'front':
            move('oriented', position, orientation, True)
    outbound = [event['name'] for event in events if 'position' in event]
    grasp_xy = tuple(value - .08 * axis for value, axis in zip(point[:2], direction))
    if stage in ('approach', 'touch', 'pickup', 'grasp', 'plan'):
        move('aligned', (*grasp_xy, height), orientation)
        move('approached', (*grasp_xy, point[2] + .12), orientation, True)
    if stage in ('touch', 'pickup', 'grasp', 'plan'):
        move('contact', (*grasp_xy, point[2] + clearance), orientation, True)
    if stage in ('pickup', 'grasp', 'plan'):
        hand('closed', 'close')
        move('lifted', (*grasp_xy, point[2] + .20), orientation, True)
        move('followed', (*grasp_xy, point[2] + .25), orientation, True)
    if stage in ('grasp', 'plan'):
        placed_xy = tuple(value - .08 * axis for value, axis in zip(destination[:2], direction))
        move('transferred', (*placed_xy, point[2] + .25), orientation, True)
        events[-1]['requires_verified_pickup'] = True
        move('lowered', (*placed_xy, point[2] + clearance), orientation, True)
        hand('released', 'open')
        reverse('withdrawn', 'transferred', True)
        reverse('source_high_return', 'followed')
    elif stage == 'pickup':
        reverse('lifted_return', 'lifted')
        reverse('source_release_position', 'contact')
        hand('released', 'open')
    if stage in ('touch', 'pickup'):
        reverse('approached_return', 'approached', True)
    if stage in ('approach', 'touch', 'pickup', 'grasp', 'plan'):
        reverse('aligned_return', 'aligned')
    for name in reversed(outbound):
        reverse(name + '_return', name)
    return events, destination


def observe(output, require_complete=True):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('rabokit_stereo.py')),
                             '--output', str(output)], capture_output=True, text=True, timeout=25)
    output.with_suffix('.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if require_complete or not (output / 'stereo.json').exists():
        result.check_returncode()
    observation = json.loads((output / 'stereo.json').read_text(encoding='utf-8'))
    observation['capture_process_exit'] = result.returncode
    if result.returncode and not observation['issues']:
        observation['issues'].append(f'Capture process exited {result.returncode}')
    return observation


def check_plan(arm, start, events):
    previous, previous_rpy = start['pose']
    checks = []
    for event in events:
        if 'position' not in event or 'reverse_of' in event or 'joints' in event:
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
            'relocation_verified': planar_error < .015 and height_error < .02 and moved > .08}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('plan', 'orient', 'approach', 'touch', 'pickup', 'grasp'), default='plan')
    parser.add_argument('--plan-only', action='store_true', help='Check the selected stage without enabling motion')
    parser.add_argument('--seed-record', type=Path, help='Use the achieved contact joint branch from a prior grasp.json')
    parser.add_argument('--tool-clearance', type=float, default=.03,
                        help='Contact tool height above the observed colored surface, metres')
    parser.add_argument('--yaw-deg', type=float, default=-90,
                        help='Horizontal entry direction; -90 approaches from outside the brick columns')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run only in the Rabokit simulation terminal')
    if not math.isfinite(args.tool_clearance) or not 0 < args.tool_clearance <= .10:
        parser.error('--tool-clearance must be in (0, .10]')
    if not math.isfinite(args.yaw_deg) or not -180 <= args.yaw_deg <= 180:
        parser.error('--yaw-deg must be in [-180, 180]')
    if args.seed_record and (args.stage not in ('approach', 'touch') or args.yaw_deg != -90):
        parser.error('--seed-record currently requires approach/touch and yaw -90')
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
    motion = args.stage != 'plan' and not args.plan_only
    try:
        time.sleep(1)
        arm = robot.left_arm
        start = measured(arm)
        events, destination = make_plan(start, point, surface['half_pixel_perturbation_radius_m'],
                                        args.stage, args.tool_clearance, math.radians(args.yaw_deg))
        seeded_checks = None
        if args.seed_record:
            from rabokit_joint_path import seeded_approach_plan
            from _robolib_core.arms import _lbot_kinematics as k
            forward = lambda joints: k.forward_kinematics('left', joints)
            events, seeded_checks = seeded_approach_plan(start, point, surface['half_pixel_perturbation_radius_m'],
                args.tool_clearance, json.loads(args.seed_record.read_text(encoding='utf-8')), args.stage,
                lambda target, seed: k.inverse_kinematics('left', target, seed_real=seed), forward)
        if any(math.dist(item['colored_surface_point_sdk_m'][:2], destination[:2]) < .07
               for item in initial['matches'] if item['color'] != 'green'):
            raise RuntimeError('The RGB destination overlaps another observed brick')
        checks = check_plan(arm, start, events)
        report = {'stage': args.stage, 'start': start, 'source_surface': surface,
                  'destination_surface': destination, 'events': events, 'reachability_checks': checks,
                  'tool_clearance_m': args.tool_clearance, 'completed': [], 'physical_grasp_verified': False,
                  'entry_yaw_deg': args.yaw_deg,
                  'seeded_path_checks': seeded_checks,
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
            joint_targets = {}
            pending = list(events)
            recovery_plan, _ = make_plan(start, point, surface['half_pixel_perturbation_radius_m'],
                                         'pickup', args.tool_clearance, math.radians(args.yaw_deg))
            recovery_plan = recovery_plan[next(i for i, event in enumerate(recovery_plan)
                                              if event['name'] == 'followed') + 1:]

            def recover(reason):
                report['recovery_reason'] = reason
                return [{**event, 'capture': False} for event in recovery_plan if 'reverse_of' not in event or
                        event['reverse_of'] in joint_targets]

            while pending:
                event = pending.pop(0)
                report['active_stage'] = event['name']
                save()
                if event.get('requires_verified_pickup') and not report.get('pickup_scene_verified', False):
                    pending = recover('No verified pickup before lateral transfer')
                    save()
                    continue
                if 'hand' in event:
                    hand_command(event['hand'])
                    time.sleep(1)
                    state = measured(arm)
                elif 'reverse_of' in event:
                    state = (move_joint_pose(arm, joint_targets[event['reverse_of']], forward)
                             if args.seed_record else return_joints(arm, joint_targets[event['reverse_of']]))
                elif 'joints' in event:
                    state = move_joint_pose(arm, event['joints'], forward, settle=.5 if event['capture'] else .2)
                else:
                    state = move_pose(arm, event['position'], event['orientation'])
                joint_targets[event['name']] = state['joints']
                report['completed'].append({'name': event['name'], 'measured': state})
                save()
                print(json.dumps({'completed': event['name'], 'measured': state}), flush=True)
                if event['capture']:
                    try:
                        sample(arm, args.output, event['name'], stereo=True)
                    except (RuntimeError, subprocess.TimeoutExpired) as error:
                        if args.seed_record:
                            raise RuntimeError(f'{event["name"]}: seeded route camera sampling failed') from error
                        pending = recover(f'{event["name"]}: camera sampling failed: {error}')
                        save()
                        continue
                if event['name'] == 'followed':
                    try:
                        first = target_observation(args.output / 'lifted')
                        second = target_observation(args.output / 'followed')
                        tools = [json.loads((args.output / label / 'sdk.json').read_text(encoding='utf-8'))
                                 ['after']['pose'][0] for label in ('lifted', 'followed')]
                        direction = (-math.sin(math.radians(args.yaw_deg)), math.cos(math.radians(args.yaw_deg)))
                        proof = pickup_evidence(point, first, second, *tools,
                                                 [axis * .08 for axis in direction], args.tool_clearance)
                        neighbors = [neighboring_layout(initial, saved_scene(args.output / label,
                                      colors={'red', 'yellow', 'blue'}), exclude_green=True)
                                     for label in ('lifted', 'followed')]
                        report['pickup_evidence'] = proof
                        report['neighbor_observations_during_pickup'] = neighbors
                        report['physical_grasp_verified'] = proof['physical_grasp_verified']
                        report['pickup_scene_verified'] = proof['physical_grasp_verified'] and all(
                            item['consistent'] for item in neighbors)
                    except (ValueError, KeyError, OSError) as error:
                        report['pickup_scene_verified'] = False
                        report['pickup_evidence'] = {'physical_grasp_verified': False, 'reason': str(error)}
                    save()
                    print(json.dumps({'pickup_evidence': report.get('pickup_evidence'),
                                      'pickup_scene_verified': report['pickup_scene_verified']}), flush=True)
            return_joints(arm, start['joints'])
            report['completed'].append({'name': 'restored', 'measured': measured(arm)})
            report['return_completed'] = True
            report['motion_sequence_completed'] = 'recovery_reason' not in report
            save()
            try:
                sample(arm, args.output, 'restored', stereo=True)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                report['restored_capture_error'] = str(error)
                save()
            if args.stage in ('touch', 'pickup', 'grasp'):
                final = observe(args.output / 'final', require_complete=False)
                settled = observe(args.output / 'settled', require_complete=False)
                report['neighbor_observations_after_return'] = [neighboring_layout(initial, view,
                    exclude_green=args.stage != 'touch') for view in (final, settled)]
                goal = destination if args.stage == 'grasp' and 'recovery_reason' not in report else point
                report['final_observation'] = relocation(final, goal, point)
                report['settled_observation'] = relocation(settled, goal, point)
                positions = [report[key].get('observed_surface') for key in
                             ('final_observation', 'settled_observation')]
                report['release_observations_consistent'] = (
                    all(positions) and math.dist(*positions) < .01)
                save()
                neighbors_ok = all(item['consistent'] for item in report['neighbor_observations_after_return'])
                if args.stage == 'touch':
                    if not neighbors_ok or 'recovery_reason' in report:
                        raise RuntimeError('Low open-hand approach did not preserve the observed brick arrangement')
                else:
                    placed = (all(report[key].get('destination_planar_error_m', math.inf) < .015 and
                                  report[key].get('destination_height_error_m', math.inf) < .02
                                  for key in ('final_observation', 'settled_observation')) and
                              report['release_observations_consistent'])
                    if not (report.get('pickup_scene_verified') and neighbors_ok and placed and
                            'recovery_reason' not in report):
                        raise RuntimeError('RGB did not verify clean pickup and release; measured return completed')
                    print('PASS: RGB pickup and release with consistent neighboring observations', flush=True)
        print(f'PASS: {args.stage} motion plan' + (' executed with measured return' if motion else ' (no motion)'), flush=True)
    except Exception as error:
        if 'report' in locals():
            report['error'] = f'{type(error).__name__}: {error}'
            try:
                report['last_measured'] = measured(arm)
            except RuntimeError:
                pass
            save()
        raise
    finally:
        try:
            if motion:
                robot.stop()
        finally:
            robot.shutdown()


if __name__ == '__main__':
    main()
