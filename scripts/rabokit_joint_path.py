#!/usr/bin/env python3
"""Check a seeded translation between recorded robot states; never send commands."""
import argparse
import json
import math
from pathlib import Path


def translation_path(start, goal, inverse, forward, spacing=.01, joint_step_limit=.15):
    """Keep the recorded tool rotation and reject discontinuous IK solutions."""
    import numpy as np

    joints = list(start['joints'])
    transform = np.asarray(forward(joints), dtype=float)
    origin = transform[:3, 3].copy()
    if math.dist(origin, start['pose'][0]) > .003:
        raise ValueError('SDK FK and recorded position disagree')
    goal = np.asarray(goal, dtype=float)
    count = max(1, math.ceil(float(np.linalg.norm(goal - origin)) / spacing))
    points = []
    for index in range(1, count + 1):
        target = transform.copy()
        target[:3, 3] = origin + (goal - origin) * index / count
        solved = inverse(target, joints)
        if solved is None:
            return {'continuous': False, 'failed_step': index, 'reason': 'No seeded IK solution', 'points': points}
        solved = [float(value) for value in solved]
        jump = max(abs(a - b) for a, b in zip(joints, solved))
        if jump > joint_step_limit:
            return {'continuous': False, 'failed_step': index, 'maximum_joint_step_rad': jump,
                    'reason': 'Seeded IK changed branch', 'points': points}
        # The controller interpolates joints; check that sampled tool poses also follow this segment.
        previous_xyz = np.asarray(forward(joints))[:3, 3]
        maximum_error = 0.
        maximum_angle = 0.
        for fraction in (.25, .5, .75, 1.):
            interpolated = [a + fraction * (b - a) for a, b in zip(joints, solved)]
            actual = np.asarray(forward(interpolated))
            expected = previous_xyz + fraction * (target[:3, 3] - previous_xyz)
            maximum_error = max(maximum_error, float(np.linalg.norm(actual[:3, 3] - expected)))
            cosine = float((np.trace(target[:3, :3].T @ actual[:3, :3]) - 1) / 2)
            maximum_angle = max(maximum_angle, math.acos(max(-1., min(1., cosine))))
        if maximum_error > .003 or maximum_angle > .01:
            return {'continuous': False, 'failed_step': index, 'reason': 'Joint interpolation leaves tool segment',
                    'maximum_position_error_m': maximum_error, 'maximum_orientation_error_rad': maximum_angle,
                    'points': points}
        points.append({'joints': solved, 'target_xyz': target[:3, 3].tolist(),
                       'maximum_joint_step_rad': jump, 'sampled_position_error_m': maximum_error,
                       'sampled_orientation_error_rad': maximum_angle})
        joints = solved
    return {'continuous': True, 'points': points,
            'scope': 'Seeded IK and sampled tool interpolation only; no robot-body collision clearance'}


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
