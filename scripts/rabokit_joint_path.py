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
    return {'continuous': True, 'points': points,
            'scope': 'Seeded IK and sampled tool interpolation only; no robot-body collision clearance'}


def translation_path(start, goal, inverse, forward, spacing=.01, joint_step_limit=.15):
    """Keep the recorded tool rotation and reject discontinuous IK solutions."""
    import numpy as np

    target = np.asarray(forward(start['joints']), dtype=float).copy()
    target[:3, 3] = goal
    return pose_path(start, target, inverse, forward, spacing, joint_step_limit)


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
