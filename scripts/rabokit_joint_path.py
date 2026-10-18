#!/usr/bin/env python3
"""Check a seeded translation between recorded robot states; never send commands."""
import argparse
import json
import math
from pathlib import Path


def interpolate_rotation(start, goal, fraction):
    """Interpolate on SO(3), including the 180-degree case."""
    import numpy as np

    delta = start.T @ goal
    angle = math.acos(max(-1., min(1., float((np.trace(delta) - 1) / 2))))
    if angle < 1e-8:
        return start.copy()
    if abs(math.sin(angle)) > 1e-6:
        axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                         delta[1, 0] - delta[0, 1]]) / (2 * math.sin(angle))
    else:
        _, vectors = np.linalg.eigh((delta + delta.T + 2 * np.eye(3)) / 4)
        axis = vectors[:, -1]
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    theta = angle * fraction
    return start @ (np.eye(3) + math.sin(theta) * skew + (1 - math.cos(theta)) * (skew @ skew))


def continuous_inverse(target, seed, inverse, forward, limits, fallbacks=None):
    """Keep accepted SDK solutions; try bounded local FK solving after a branch jump."""
    import numpy as np

    seed = np.asarray(seed, dtype=float)
    solved = inverse(target, seed.tolist())
    valid = solved is not None and len(solved) == len(seed) and np.all(np.isfinite(solved))
    jump = float(np.max(np.abs(np.asarray(solved) - seed))) if valid else None
    if valid and jump <= .15 and all(lo <= value <= hi for value, (lo, hi) in zip(solved, limits)):
        return list(solved)
    lower = np.maximum(seed - .1499, [lo for lo, hi in limits])
    upper = np.minimum(seed + .1499, [hi for lo, hi in limits])
    if np.any(lower > upper):
        return None
    target = np.asarray(target, dtype=float)

    def residual(joints):
        actual = np.asarray(forward(joints.tolist()), dtype=float)
        delta = target[:3, :3].T @ actual[:3, :3]
        spin = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                         delta[1, 0] - delta[0, 1]]) / 2
        angle = math.acos(max(-1., min(1., float((np.trace(delta) - 1) / 2))))
        position = actual[:3, 3] - target[:3, 3]
        return np.r_[position / .003, spin / .01], float(np.linalg.norm(position)), angle

    joints = np.clip(seed, lower, upper)
    for _ in range(60):
        error, position_error, angle_error = residual(joints)
        if position_error < 1e-5 and angle_error < 1e-4:
            if fallbacks is not None:
                fallbacks.append({'sdk_joint_step_rad': jump,
                                  'local_joint_step_rad': float(np.max(np.abs(joints - seed))),
                                  'position_error_m': position_error, 'orientation_error_rad': angle_error})
            return joints.tolist()
        jacobian = np.column_stack([(residual(joints + np.eye(len(seed))[axis] * 1e-5)[0] - error) / 1e-5
                                    for axis in range(len(seed))])
        step = -np.linalg.lstsq(jacobian, error, rcond=1e-6)[0]
        step *= min(1., .03 / max(float(np.max(np.abs(step))), 1e-12))
        for scale in (1., .5, .25, .125):
            candidate = np.clip(joints + scale * step, lower, upper)
            if np.linalg.norm(residual(candidate)[0]) < np.linalg.norm(error):
                joints = candidate
                break
        else:
            return None
    return None


def pose_path(start, goal, inverse, forward, spacing=.01, joint_step_limit=.15):
    """Plan translation and rotation with seeded IK; never send controller commands."""
    import numpy as np

    joints = list(start['joints'])
    transform = np.asarray(forward(joints), dtype=float)
    origin = transform[:3, 3].copy()
    if math.dist(origin, start['pose'][0]) > .003:
        raise ValueError('SDK FK and recorded position disagree')
    goal = np.asarray(goal, dtype=float)
    angle = math.acos(max(-1., min(1., float((np.trace(transform[:3, :3].T @ goal[:3, :3]) - 1) / 2))))
    count = max(1, math.ceil(float(np.linalg.norm(goal[:3, 3] - origin)) / spacing), math.ceil(angle / .05))
    points = []
    for index in range(1, count + 1):
        target = transform.copy()
        target[:3, 3] = origin + (goal[:3, 3] - origin) * index / count
        target[:3, :3] = interpolate_rotation(transform[:3, :3], goal[:3, :3], index / count)
        solved = inverse(target, joints)
        if solved is None:
            return {'continuous': False, 'failed_step': index, 'reason': 'No seeded IK solution', 'points': points}
        solved = [float(value) for value in solved]
        if len(solved) != len(joints) or not all(math.isfinite(value) for value in solved):
            return {'continuous': False, 'failed_step': index, 'reason': 'Invalid IK joints', 'points': points}
        jump = max(abs(a - b) for a, b in zip(joints, solved))
        if jump > joint_step_limit:
            return {'continuous': False, 'failed_step': index, 'maximum_joint_step_rad': jump,
                    'reason': 'Seeded IK changed branch', 'points': points}
        # The controller interpolates joints; check that sampled tool poses also follow this segment.
        previous = np.asarray(forward(joints))
        previous_xyz = previous[:3, 3]
        maximum_error = 0.
        maximum_angle = 0.
        for fraction in (.25, .5, .75, 1.):
            interpolated = [a + fraction * (b - a) for a, b in zip(joints, solved)]
            actual = np.asarray(forward(interpolated))
            expected = previous_xyz + fraction * (target[:3, 3] - previous_xyz)
            maximum_error = max(maximum_error, float(np.linalg.norm(actual[:3, 3] - expected)))
            expected_rotation = interpolate_rotation(previous[:3, :3], target[:3, :3], fraction)
            cosine = float((np.trace(expected_rotation.T @ actual[:3, :3]) - 1) / 2)
            maximum_angle = max(maximum_angle, math.acos(max(-1., min(1., cosine))))
        if maximum_error > .003 or maximum_angle > .01:
            return {'continuous': False, 'failed_step': index, 'reason': 'Joint interpolation leaves tool segment',
                    'maximum_position_error_m': maximum_error, 'maximum_orientation_error_rad': maximum_angle,
                    'points': points}
        points.append({'joints': solved, 'target_xyz': target[:3, 3].tolist(),
                       'target_rotation': target[:3, :3].tolist(),
                       'maximum_joint_step_rad': jump, 'sampled_position_error_m': maximum_error,
                       'sampled_orientation_error_rad': maximum_angle})
        joints = solved
    result = {'continuous': True, 'points': points,
              'scope': 'Seeded IK and sampled tool interpolation only; no robot-body collision clearance'}
    if spacing < .01:
        result['points'] = _coarsen_samples(start, points, forward, joint_step_limit)
        result['control_point_reduction'] = {'original_count': len(points),
                                             'retained_count': len(result['points'])}
    return result


def _coarsen_samples(start, points, forward, joint_step_limit):
    """Merge neighboring fine samples only after the ordinary segment checks pass."""
    import numpy as np

    retained, previous, index = [], start, 0
    previous_target = np.asarray(start['pose'][0])
    while index < len(points):
        selected, combined = index, None
        if index + 1 < len(points):
            joints = points[index + 1]['joints']
            target = np.asarray(forward(joints))
            current = np.asarray(forward(previous['joints']))
            angle = math.acos(max(-1., min(1., float((np.trace(current[:3, :3].T @ target[:3, :3]) - 1) / 2))))
            if (np.linalg.norm(np.asarray(points[index + 1]['target_xyz']) - previous_target) <= .01 + 1e-9
                    and angle <= .05 and
                    max(abs(a - b) for a, b in zip(joints, previous['joints'])) <= joint_step_limit):
                checked = pose_path(previous, target, lambda t, s: joints, forward,
                                    spacing=.01, joint_step_limit=joint_step_limit)
                if checked['continuous']:
                    selected, combined = index + 1, checked['points'][-1]
        point = points[selected] if combined is None else combined
        retained.append(point)
        previous_target = np.asarray(points[selected]['target_xyz'])
        previous = {'joints': point['joints'],
                    'pose': [np.asarray(forward(point['joints']))[:3, 3].tolist(), [0.] * 3]}
        index = selected + 1
    return retained


def translation_path(start, goal, inverse, forward, spacing=.01, joint_step_limit=.15):
    """Keep the recorded tool rotation and reject discontinuous IK solutions."""
    import numpy as np

    target = np.asarray(forward(start['joints']), dtype=float).copy()
    target[:3, 3] = goal
    return pose_path(start, target, inverse, forward, spacing, joint_step_limit)


def seeded_approach_plan(start, point, uncertainty, clearance, reference, stage, inverse, forward,
                         tool_offset_y=0., tool_offset_x=0.):
    """Keep accepted routes; retry a rejected route once with 5 mm translations."""
    target_point = [point[0] + tool_offset_x, point[1] + tool_offset_y, point[2]]
    arguments = (start, target_point, uncertainty, clearance, reference, stage, inverse, forward)
    try:
        return _seeded_approach_plan(*arguments, spacing=.01)
    except ValueError as coarse_error:
        try:
            events, checks = _seeded_approach_plan(*arguments, spacing=.005)
        except ValueError as fine_error:
            raise ValueError(f'10 mm route rejected: {coarse_error}; '
                             f'5 mm route rejected: {fine_error}') from fine_error
        checks.insert(0, {'stage': 'translation_refinement', 'spacing_m': .005,
                          'coarse_plan_rejection': str(coarse_error)})
        return events, checks


def _seeded_approach_plan(start, point, uncertainty, clearance, reference, stage, inverse, forward, spacing):
    """Reverse-plan from an achieved low branch, then connect behind the table."""
    import numpy as np

    if stage not in ('approach', 'touch', 'pickup'):
        raise ValueError('Seeded route supports approach, touch and source-return pickup stages')
    reference_joints = next(event['measured']['joints'] for event in reference['completed']
                            if event['name'] == 'contact')
    target = np.asarray(forward(reference_joints)).copy()
    target[:3, 3] = [point[0] - .08, point[1], point[2] + clearance]
    low = inverse(target, reference_joints)
    if low is None or max(abs(a - b) for a, b in zip(low, reference_joints)) > .15:
        raise ValueError('Fresh RGB low target does not preserve the recorded joint branch')
    if float(np.linalg.norm(np.asarray(forward(low))[:3, 3] - target[:3, 3])) > .003:
        raise ValueError('Seeded low joint target does not reach the RGB tool target')

    def node(name, joints, capture=False):
        transform = np.asarray(forward(joints))
        rotation = transform[:3, :3]
        rpy = [math.atan2(rotation[2, 1], rotation[2, 2]),
               math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2])),
               math.atan2(rotation[1, 0], rotation[0, 0])]
        return {'name': name, 'joints': [float(value) for value in joints],
                'position': transform[:3, 3].tolist(), 'orientation': rpy, 'capture': capture}

    backward = [node('contact', low, True)]
    checks = []
    height = point[2] + .30 + uncertainty
    destinations = [('approached', [point[0] - .08, point[1], point[2] + .12], False),
                    ('aligned', [point[0] - .08, point[1], height], False),
                    ('over_row', [.35, point[1], height], False),
                    ('front', [.35, .60, height], False),
                    ('vertical_front', [.35, .60, height], True),
                    ('outside', [-.20, .60, height], True)]
    joints = low
    for name, xyz, vertical in destinations:
        current = node('current', joints)
        target = np.asarray(forward(joints)).copy()
        target[:3, 3] = xyz
        if vertical:
            target[:3, :3] = np.eye(3)
        path = pose_path({'joints': joints, 'pose': [current['position'], current['orientation']]},
                         target, inverse, forward, spacing=spacing)
        checks.append({'stage': name, **{key: value for key, value in path.items() if key != 'points'},
                       'point_count': len(path['points'])})
        if not path['continuous']:
            raise ValueError(f'{name}: {path.get("reason")} at step {path.get("failed_step")}; '
                             f'joint step {path.get("maximum_joint_step_rad")} rad')
        for index, sample in enumerate(path['points']):
            last = index == len(path['points']) - 1
            backward.append(node(name if last else f'{name}_{index:03}', sample['joints'], last))
        joints = path['points'][-1]['joints']

    # This connection stays behind the observed bricks; it does not claim whole-body collision clearance.
    count = max(1, math.ceil(max(abs(a - b) for a, b in zip(start['joints'], joints)) / .10))
    bridge = [node(f'bridge_{index:03}', [a + (b - a) * index / count
                   for a, b in zip(start['joints'], joints)]) for index in range(1, count + 1)]
    bridge_xyz = np.asarray([forward([a + (b - a) * t for a, b in zip(start['joints'], joints)])[:3, 3]
                             for t in np.linspace(0, 1, 201)])
    if float(bridge_xyz[:, 0].max()) > .05:
        raise ValueError('Initial joint connection leaves the rear tool region')
    checks.append({'stage': 'initial_joint_connection', 'continuous': True,
                   'tool_minimum': bridge_xyz.min(axis=0).tolist(),
                   'tool_maximum': bridge_xyz.max(axis=0).tolist(), 'point_count': len(bridge)})
    outgoing = bridge
    for event in reversed(backward[:-1]):
        outgoing.append(event)
        if event['name'] == ('approached' if stage == 'approach' else 'contact'):
            break
    returning = [{**event, 'name': event['name'] + '_return', 'reverse_of': event['name'], 'capture': False}
                 for event in reversed(outgoing[:-1])]
    pickup = []
    if stage == 'pickup':
        raised = []
        joints = outgoing[-1]['joints']
        for name, height_offset in (('lifted', .20), ('followed', .25)):
            current = node('current', joints)
            target = np.asarray(forward(joints)).copy()
            target[:3, 3] = [point[0] - .08, point[1], point[2] + height_offset]
            path = pose_path({'joints': joints, 'pose': [current['position'], current['orientation']]},
                             target, inverse, forward, spacing=spacing)
            checks.append({'stage': name, **{key: value for key, value in path.items() if key != 'points'},
                           'point_count': len(path['points'])})
            if not path['continuous']:
                raise ValueError(f'{name}: {path.get("reason")} at step {path.get("failed_step")}')
            for index, sample in enumerate(path['points']):
                last = index == len(path['points']) - 1
                raised.append(node(name if last else f'{name}_{index:03}', sample['joints'], last))
            joints = path['points'][-1]['joints']
        lowered = [{**event, 'name': event['name'] + '_lower_return', 'reverse_of': event['name'], 'capture': False}
                   for event in reversed(raised[:-1])]
        pickup = [{'name': 'closed', 'hand': 'close', 'capture': True}, *raised, *lowered,
                  {**outgoing[-1], 'name': 'source_release_position', 'reverse_of': 'contact', 'capture': False},
                  {'name': 'released', 'hand': 'open', 'capture': True}]
    return [{'name': 'opened', 'hand': 'open', 'capture': True}, *outgoing, *pickup, *returning], checks


def pickup_inspection_plan(events, point, inverse, forward, tool_offset_y=0., tool_offset_x=0.):
    """Insert an explicit high diagnostic peek, then retrace before source release."""
    import numpy as np

    point = [point[0] + tool_offset_x, point[1] + tool_offset_y, point[2]]
    names = [event['name'] for event in events]
    if 'followed' not in names or 'source_release_position' not in names or 'transferred' in names:
        raise ValueError('Inspection requires a source-return pickup route')
    index = names.index('followed')
    joints = events[index]['joints']
    extra, checks = [], []
    for name, height in (('inspection_high', .25), ('inspection_low', .20)):
        current = np.asarray(forward(joints))
        target = current.copy()
        target[:3, 3] = [point[0] - .08, point[1] - .08, point[2] + height]
        path = pose_path({'joints': joints, 'pose': [current[:3, 3].tolist(), [0.] * 3]},
                         target, inverse, forward)
        checks.append({'stage': name, **{key: value for key, value in path.items() if key != 'points'},
                       'point_count': len(path['points'])})
        if not path['continuous']:
            raise ValueError(f'{name}: {path.get("reason")} at step {path.get("failed_step")}')
        for i, sample in enumerate(path['points']):
            last = i == len(path['points']) - 1
            extra.append({'name': name if last else f'{name}_{i:03}', 'joints': sample['joints'],
                          'capture': last and name != 'inspection_high'})
        joints = path['points'][-1]['joints']
        if name == 'inspection_high':
            horizontal = list(extra)
    # Reuse the checked descent instead of asking IK to choose an upward branch.
    upward = list(reversed(extra[len(horizontal) - 1:-1]))
    for i, event in enumerate(upward):
        last = i == len(upward) - 1
        extra.append({'name': 'inspection_followed' if last else f'inspection_followed_{i:03}',
                      'joints': event['joints'], 'reverse_of': event['name'], 'capture': last})
    checks.append({'stage': 'inspection_followed', 'continuous': True, 'point_count': len(upward),
                   'scope': 'Reverse of checked descent; uses actually measured joints at execution'})
    returning = [{**event, 'name': event['name'] + '_inspection_return',
                  'reverse_of': event['name'], 'capture': False} for event in reversed(horizontal[:-1])]
    returning.append({'name': 'inspection_source_return', 'reverse_of': 'followed', 'capture': False})
    returning.append({'name': 'empty_pickup_open', 'hand': 'open', 'capture': False,
                      'requires_tabletop_target': True, 'wait_open': True})
    return events[:index + 1] + extra + returning + events[index + 1:], checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path, help='grasp.json with actual completed robot states')
    parser.add_argument('--start', required=True, help='Name of the recorded starting stage')
    parser.add_argument('--goal', required=True, help='Name of the stage whose XYZ is the target')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new file')
    # These private interfaces were inspected in the installed SDK 3.0.0; no controller is constructed.
    import rabo_robolib  # noqa: F401 - loads the SDK implementation
    from _robolib_core.arms import _lbot_kinematics as k

    states = {event['name']: event['measured'] for event in json.loads(
        args.record.read_text(encoding='utf-8'))['completed']}
    result = translation_path(states[args.start], states[args.goal]['pose'][0],
                              lambda target, seed: k.inverse_kinematics('left', target, seed_real=seed),
                              lambda joints: k.forward_kinematics('left', joints))
    result.update(start=args.start, goal=args.goal, motion_sent=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'points'}), flush=True)
    raise SystemExit(0 if result['continuous'] else 1)


if __name__ == '__main__':
    main()
