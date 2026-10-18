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

from rabokit_calibration_probe import hand_command, initialize_empty_hand, measured, sample
from rabokit_hover import hover_waypoints, move_joint_pose, move_pose, return_joints
from rabokit_pickup_vision import neighboring_layout, pickup_evidence, saved_scene, target_observation

TOP_GRASP_RPY = (math.pi / 2, 0., -math.pi / 2)
ZERO_RPY = (0., 0., 0.)


def pickup_offset(yaw, tool_offset_y=0., tool_offset_x=0.):
    """Expected tool-to-surface XY vector, including a deliberate tool shift."""
    return [-.08 * math.sin(yaw) - tool_offset_x, .08 * math.cos(yaw) - tool_offset_y]


def high_pass_through_plan(events, start, surface_z):
    """Mark only high outbound intermediate joints before hand prepositioning."""
    previous = start['pose'][0]
    outbound = True
    result = []
    for index, event in enumerate(events):
        if 'hand' in event and event['name'] != 'opened':
            outbound = False
        if outbound and 'joints' in event and 'position' in event:
            before_hand = index + 1 < len(events) and 'hand' in events[index + 1]
            if (not event.get('capture') and not before_hand and
                    min(previous[2], event['position'][2]) >= surface_z + .15):
                event = {**event, 'pass_through': True}
            previous = event['position']
        result.append(event)
    return result


def preshape_fingers_plan(events, limited_pinch=False, prepare_thumb_high=False):
    """Prepare fingers high, optionally bend the thumb high, and retain the release shape."""
    if prepare_thumb_high and not limited_pinch:
        raise ValueError('High thumb preparation requires limited pinch')
    names = [event['name'] for event in events]
    if any(names.count(name) != 1 for name in ('aligned', 'contact', 'closed', 'released')) or not (
            names.index('aligned') < names.index('contact') < names.index('closed') < names.index('released')):
        raise ValueError('Preshape requires high alignment before contact and pickup closure')
    preshape = 'pinch_preshape_without_pinky' if limited_pinch else 'preshape_without_pinky'
    high_preshape = 'pinch_high_thumb_ready_without_pinky' if prepare_thumb_high else preshape
    thumb_ready = 'pinch_thumb_ready_without_pinky' if limited_pinch else 'thumb_ready_without_pinky'
    staged = []
    for event in events:
        if event['name'] == 'released':
            event = {**event, 'hand': preshape,
                     'preshape_feedback_key': 'release_preshape_feedback'}
        elif limited_pinch and event['name'] == 'closed':
            event = {**event, 'hand': 'pinch_without_pinky'}
        elif event.get('requires_tabletop_target') and event.get('hand') == 'open':
            event = {**event, 'name': 'empty_pickup_preshape', 'hand': preshape,
                     'capture': True, 'wait_open': False,
                     'preshape_feedback_key': 'empty_return_preshape_feedback'}
        staged.append(event)
        if event['name'] == 'aligned':
            staged.append({'name': 'preshaped', 'hand': high_preshape, 'capture': True,
                           'wait_arrival': limited_pinch,
                           'preshape_feedback_key': 'preshape_feedback'})
        elif event['name'] == 'contact':
            staged.append({'name': 'thumb_ready', 'hand': thumb_ready, 'capture': True,
                           'preshape_feedback_key': 'thumb_ready_feedback'})
    return [{**event, 'hand_ramp_seconds': 2.} if limited_pinch and 'hand' in event else event for event in staged]


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
    parser.add_argument('--keep-pinky-open', action='store_true',
                        help='Close the other fingers while keeping the pinky at its measured open target')
    parser.add_argument('--thumb-yaw', type=float,
                        help='Native thumb yaw endpoint; limited pinch also holds it during preshape and release')
    parser.add_argument('--preshape-fingers', action='store_true',
                        help='Seeded pickup: half-close fingers high, descend, then prepare thumb before closing')
    parser.add_argument('--limited-pinch', action='store_true',
                        help='Preshaped inspection only: preposition thumb yaw high, use 45%% thumb / 75%% finger closure')
    parser.add_argument('--pass-through-high', action='store_true',
                        help='Limited pinch only: advance high intermediate joints without a stop at every point')
    parser.add_argument('--prepare-thumb-high', action='store_true',
                        help='Limited pinch only: bend the thumb to its 45%% ready target before descending')
    parser.add_argument('--side-camera', action='store_true',
                        help='Record synchronized side RGB at static stages for grasp visibility diagnosis')
    parser.add_argument('--inspect-pickup', action='store_true',
                        help='Seeded pickup only: diagnose at Y -8 cm and heights 20/25 cm, then return to source')
    parser.add_argument('--tool-clearance', type=float, default=.03,
                        help='Contact tool height above the observed colored surface, metres')
    parser.add_argument('--tool-offset-y', type=float, default=0.,
                        help='Seeded route only: signed tool Y adjustment, at most 2 cm; RGB source stays unchanged')
    parser.add_argument('--tool-offset-x', type=float, default=0.,
                        help='Seeded route only: signed tool X adjustment, at most 2 cm; RGB source stays unchanged')
    parser.add_argument('--yaw-deg', type=float, default=-90,
                        help='Horizontal entry direction; -90 approaches from outside the brick columns')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run only in the Rabokit simulation terminal')
    if args.preshape_fingers and (not args.seed_record or args.stage != 'pickup' or not args.keep_pinky_open):
        parser.error('--preshape-fingers requires --seed-record, --stage pickup and --keep-pinky-open')
    if args.limited_pinch and not (args.preshape_fingers and args.inspect_pickup):
        parser.error('--limited-pinch requires --preshape-fingers and --inspect-pickup')
    if args.pass_through_high and not args.limited_pinch:
        parser.error('--pass-through-high requires --limited-pinch')
    if args.prepare_thumb_high and not args.limited_pinch:
        parser.error('--prepare-thumb-high requires --limited-pinch')
    if args.thumb_yaw is not None:
        if args.stage not in ('pickup', 'grasp'):
            parser.error('--thumb-yaw requires pickup or grasp')
        from rabokit_hand import targets
        try:
            targets('close', args.thumb_yaw)
        except ValueError as error:
            parser.error(str(error))
    if not math.isfinite(args.tool_clearance) or not 0 < args.tool_clearance <= .10:
        parser.error('--tool-clearance must be in (0, .10]')
    if not math.isfinite(args.tool_offset_y) or abs(args.tool_offset_y) > .02:
        parser.error('--tool-offset-y must be finite and within +/- .02 m')
    if not math.isfinite(args.tool_offset_x) or abs(args.tool_offset_x) > .02:
        parser.error('--tool-offset-x must be finite and within +/- .02 m')
    if (args.tool_offset_y or args.tool_offset_x) and not args.seed_record:
        parser.error('--tool-offset-x/--tool-offset-y require --seed-record')
    if not math.isfinite(args.yaw_deg) or not -180 <= args.yaw_deg <= 180:
        parser.error('--yaw-deg must be in [-180, 180]')
    if args.seed_record and (args.stage not in ('approach', 'touch', 'pickup') or args.yaw_deg != -90):
        parser.error('--seed-record currently requires approach/touch/pickup and yaw -90')
    if args.inspect_pickup and (not args.seed_record or args.stage != 'pickup'):
        parser.error('--inspect-pickup requires --seed-record and --stage pickup')
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
        fk_continuations = []
        if args.seed_record:
            from rabokit_joint_path import continuous_inverse, seeded_approach_plan
            from _robolib_core.arms import _lbot_kinematics as k
            forward = lambda joints: k.forward_kinematics('left', joints)
            sdk_inverse = lambda target, seed: k.inverse_kinematics('left', target, seed_real=seed)
            inverse = lambda target, seed: continuous_inverse(target, seed, sdk_inverse, forward,
                                                              k.REAL_JOINT_LIMITS_LEFT, fk_continuations)
            events, seeded_checks = seeded_approach_plan(start, point, surface['half_pixel_perturbation_radius_m'],
                args.tool_clearance, json.loads(args.seed_record.read_text(encoding='utf-8')), args.stage,
                inverse, forward,
                tool_offset_y=args.tool_offset_y, tool_offset_x=args.tool_offset_x)
            if args.inspect_pickup:
                from rabokit_joint_path import pickup_inspection_plan
                events, inspection_checks = pickup_inspection_plan(events, point, inverse, forward,
                    tool_offset_y=args.tool_offset_y, tool_offset_x=args.tool_offset_x)
                seeded_checks.extend(inspection_checks)
            if any(not low <= value <= high for event in events if 'joints' in event
                   for value, (low, high) in zip(event['joints'], k.REAL_JOINT_LIMITS_LEFT)):
                raise ValueError('Seeded pickup route exceeds SDK joint limits')
        if any(math.dist(item['colored_surface_point_sdk_m'][:2], destination[:2]) < .07
               for item in initial['matches'] if item['color'] != 'green'):
            raise RuntimeError('The RGB destination overlaps another observed brick')
        checks = check_plan(arm, start, events)
        if args.preshape_fingers:
            events = preshape_fingers_plan(events, limited_pinch=args.limited_pinch,
                                          prepare_thumb_high=args.prepare_thumb_high)
        if args.pass_through_high:
            events = high_pass_through_plan(events, start, point[2])
        if args.keep_pinky_open:
            for event in events:
                if event.get('hand') == 'close':
                    event['hand'] = 'close_without_pinky'
        if args.thumb_yaw is not None:
            for event in events:
                if event.get('hand') in ('close', 'close_without_pinky', 'thumb_ready_without_pinky',
                                         'pinch_high_thumb_ready_without_pinky',
                                         'pinch_preshape_without_pinky', 'pinch_thumb_ready_without_pinky',
                                         'pinch_without_pinky'):
                    event['thumb_yaw_rad'] = args.thumb_yaw
        evidence_labels = ('inspection_low', 'inspection_followed') if args.inspect_pickup else ('lifted', 'followed')
        report = {'stage': args.stage, 'start': start, 'source_surface': surface,
                  'destination_surface': destination, 'events': events, 'reachability_checks': checks,
                  'tool_clearance_m': args.tool_clearance, 'completed': [], 'physical_grasp_verified': False,
                  'tool_offset_y_m': args.tool_offset_y,
                  'tool_offset_x_m': args.tool_offset_x,
                  'thumb_yaw_override_rad': args.thumb_yaw,
                  'preshape_fingers_requested': args.preshape_fingers,
                  'limited_pinch_requested': args.limited_pinch,
                  'prepare_thumb_high_requested': args.prepare_thumb_high,
                  'hand_ramp_seconds': 2. if args.limited_pinch else 0.,
                  'pass_through_high_requested': args.pass_through_high,
                  'pass_through_joint_count': sum(bool(e.get('pass_through')) for e in events),
                  'entry_yaw_deg': args.yaw_deg,
                  'seeded_path_checks': seeded_checks,
                  'fk_continuation_fallbacks': fk_continuations,
                  'side_camera_capture': args.side_camera,
                  'inspection_requested': args.inspect_pickup, 'pickup_observation_stages': list(evidence_labels),
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
            if args.seed_record and args.stage == 'pickup':
                report['hand_initialization'] = initialize_empty_hand(arm, args.output / 'hand-initialization',
                                                                      ramp_seconds=report['hand_ramp_seconds'])
                save()
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
                if event.get('requires_tabletop_target') and not report.get('pickup_evidence', {}).get(
                        'target_stationary_on_tabletop', False):
                    report.setdefault('skipped', []).append({'name': event['name'],
                        'reason': 'No positive RGB evidence of a stationary tabletop target'})
                    save()
                    continue
                if event.get('requires_verified_pickup') and not report.get('pickup_scene_verified', False):
                    pending = recover('No verified pickup before lateral transfer')
                    save()
                    continue
                if 'hand' in event:
                    feedback = hand_command(event['hand'], thumb_yaw=event.get('thumb_yaw_rad'),
                                            wait_open=event.get('wait_open', False),
                                            wait_arrival=event.get('wait_arrival', False),
                                            ramp_seconds=event.get('hand_ramp_seconds', report['hand_ramp_seconds']))
                    if not (event.get('wait_open') or event.get('wait_arrival')):
                        time.sleep(1)
                    state = measured(arm)
                    if feedback is not None:
                        state['hand_feedback'] = feedback
                elif 'reverse_of' in event:
                    state = (move_joint_pose(arm, joint_targets[event['reverse_of']], forward)
                             if args.seed_record else return_joints(arm, joint_targets[event['reverse_of']]))
                elif 'joints' in event:
                    state = move_joint_pose(arm, event['joints'], forward, settle=.5 if event['capture'] else .2,
                                            require_settled=not event.get('pass_through', False))
                else:
                    state = move_pose(arm, event['position'], event['orientation'])
                joint_targets[event['name']] = state['joints']
                report['completed'].append({'name': event['name'], 'measured': state})
                save()
                print(json.dumps({'completed': event['name'], 'measured': state}), flush=True)
                if event['capture']:
                    try:
                        sample(arm, args.output, event['name'], stereo=True, side_camera=args.side_camera,
                               hand_trace=args.limited_pinch)
                    except (RuntimeError, subprocess.TimeoutExpired) as error:
                        if args.seed_record:
                            raise RuntimeError(f'{event["name"]}: seeded route camera sampling failed') from error
                        pending = recover(f'{event["name"]}: camera sampling failed: {error}')
                        save()
                        continue
                if event.get('preshape_feedback_key'):
                    from rabokit_hand import preparation_feedback
                    frames = json.loads((args.output / event['name'] / 'robot-frames.json').read_text(encoding='utf-8'))['frames']
                    feedback = preparation_feedback(frames, event['hand'], event.get('thumb_yaw_rad'))
                    report[event['preshape_feedback_key']] = feedback
                    save()
                    if not feedback['complete'] or (feedback['goal_arrival_required'] and
                                                   feedback['maximum_error_rad'] >= .03):
                        raise RuntimeError(f'{event["name"]}: hand preshape did not reach its targets; arm continuation stopped')
                if event['name'] in ('preshaped', 'thumb_ready'):
                    observation_key = event['name'] + '_observation' if event['name'] == 'thumb_ready' else 'preshape_observation'
                    try:
                        report[observation_key] = target_observation(args.output / event['name'])
                        if report[observation_key]['valid']:
                            report[observation_key]['source_displacement_m'] = math.dist(
                                point, report[observation_key]['point'])
                    except (ValueError, KeyError, OSError) as error:
                        report[observation_key] = {'valid': False, 'reason': str(error)}
                    save()
                if event['name'] == evidence_labels[1]:
                    try:
                        first = target_observation(args.output / evidence_labels[0])
                        second = target_observation(args.output / evidence_labels[1])
                        tools = [json.loads((args.output / label / 'sdk.json').read_text(encoding='utf-8'))
                                 ['after']['pose'][0] for label in evidence_labels]
                        proof = pickup_evidence(point, first, second, *tools,
                                                 pickup_offset(math.radians(args.yaw_deg), args.tool_offset_y,
                                                               args.tool_offset_x),
                                                 args.tool_clearance)
                        neighbors = [neighboring_layout(initial, saved_scene(args.output / label,
                                      colors={'red', 'yellow', 'blue'}), exclude_green=True)
                                     for label in evidence_labels]
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
                    if args.inspect_pickup and not report['pickup_scene_verified']:
                        report['inspection_stopped_early'] = True
                        save()
                        raise RuntimeError('RGB did not verify clean pickup; inspection trial stopped before low return')
            return_joints(arm, start['joints'])
            report['completed'].append({'name': 'restored', 'measured': measured(arm)})
            report['return_completed'] = True
            save()
            if args.preshape_fingers:
                report['active_stage'] = 'retreat_open'
                save()
                report['retreat_open_feedback'] = hand_command('open', wait_open=True,
                                                               ramp_seconds=report['hand_ramp_seconds'])
                report['active_stage'] = 'restored'
            report['motion_sequence_completed'] = 'recovery_reason' not in report
            save()
            try:
                sample(arm, args.output, 'restored', stereo=True, side_camera=args.side_camera)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                report['restored_capture_error'] = str(error)
                save()
            if args.stage in ('approach', 'touch', 'pickup', 'grasp'):
                final = observe(args.output / 'final', require_complete=False)
                settled = observe(args.output / 'settled', require_complete=False)
                report['neighbor_observations_after_return'] = [neighboring_layout(initial, view,
                    exclude_green=args.stage not in ('approach', 'touch')) for view in (final, settled)]
                goal = destination if args.stage == 'grasp' and 'recovery_reason' not in report else point
                report['final_observation'] = relocation(final, goal, point)
                report['settled_observation'] = relocation(settled, goal, point)
                positions = [report[key].get('observed_surface') for key in
                             ('final_observation', 'settled_observation')]
                report['post_return_observations_consistent'] = (
                    all(positions) and math.dist(*positions) < .01)
                save()
                neighbors_ok = all(item['consistent'] for item in report['neighbor_observations_after_return'])
                if args.stage in ('approach', 'touch'):
                    if not neighbors_ok or 'recovery_reason' in report:
                        raise RuntimeError('Low open-hand approach did not preserve the observed brick arrangement')
                else:
                    placed = (all(report[key].get('destination_planar_error_m', math.inf) < .015 and
                                  report[key].get('destination_height_error_m', math.inf) < .02
                                  for key in ('final_observation', 'settled_observation')) and
                              report['post_return_observations_consistent'])
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
