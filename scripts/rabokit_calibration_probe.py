#!/usr/bin/env python3
"""Record SDK pose, robot frames and RGB; optionally move the wrist or empty hand."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def measured(arm):
    joints, pose = arm.get_joint_angles(), arm.get_pose()
    if (joints is None or len(joints) != 7 or pose is None or len(pose) != 2
            or any(len(part) != 3 for part in pose)
            or not all(math.isfinite(v) for v in (*joints, *pose[0], *pose[1]))):
        raise RuntimeError('Expected measured joint angles and a finite SDK pose')
    return {'joints': list(joints), 'pose': [list(part) for part in pose]}


def move(arm, goal):
    if not arm.move_joints(goal, speed=10, blocking=False):
        raise RuntimeError('Wrist movement was rejected')
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        state = measured(arm)
        if max(abs(a - b) for a, b in zip(state['joints'], goal)) < 0.005:
            return
        time.sleep(0.1)
    raise TimeoutError('Wrist joint feedback did not reach its target')


def sample(arm, root, label):
    folder = root / label
    folder.mkdir()
    before = measured(arm)
    scripts = Path(__file__).resolve().parent
    for command in (
        [str(scripts / 'rabokit_robot_frames.py'), '--seconds', '2',
         '--output', str(folder / 'robot-frames.json')],
        [str(scripts / 'rabokit_camera.py'), '--seconds', '1',
         '--topic', 'base/base_cam/image_raw', '--output', str(folder / 'camera')],
    ):
        result = subprocess.run([sys.executable, *command], capture_output=True, text=True, timeout=15)
        (folder / (Path(command[0]).stem + '.log')).write_text(
            result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{label}: {Path(command[0]).name} exited {result.returncode}')
    after = measured(arm)
    drift = max(abs(a - b) for a, b in zip(before['joints'], after['joints']))
    if drift > 0.005:
        raise RuntimeError(f'{label}: arm drifted during static calibration sampling')
    record = {'label': label, 'before': before, 'after': after, 'max_joint_drift_rad': drift}
    (folder / 'sdk.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--motion', action='store_true', help='Left J7 +0.08 rad, sample, then restore')
    actions.add_argument('--hand-motion', action='store_true', help='Open, close and reopen the empty left hand')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run in the Rabokit cloud simulation terminal')
    if args.output.exists():
        parser.error('--output must be a new directory')
    from rabo_robolib import LbotDualArm

    args.output.mkdir(parents=True)
    robot = LbotDualArm(mode='sim')
    try:
        time.sleep(1)
        arm = robot.left_arm
        start = measured(arm)['joints']
        records = [sample(arm, args.output, 'start')]
        if args.motion:
            if not arm.enable():
                raise RuntimeError('Left arm enable failed')
            target = start.copy()
            target[6] += 0.08
            move(arm, target)
            records.append(sample(arm, args.output, 'rolled'))
            move(arm, start)
            records.append(sample(arm, args.output, 'restored'))
        if args.hand_motion:
            for label, action in (('opened', robot.left_hand.open), ('closed', robot.left_hand.close),
                                  ('reopened', robot.left_hand.open)):
                if not action():
                    raise RuntimeError(f'Left hand {label} command rejected')
                # SDK hand blocking is an estimate; preserve actual sim link frames for comparison.
                time.sleep(1)
                records.append(sample(arm, args.output, label))
        report = {'motion_requested': args.motion or args.hand_motion,
                  'motion_kind': 'wrist' if args.motion else 'empty_hand' if args.hand_motion else None,
                  'samples': records,
                  'extrinsics_validated': False, 'official_score': None}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('PASS: static calibration samples' + (' and wrist motion/return' if args.motion else
                                                  ' and empty hand command sequence' if args.hand_motion else ''))
    finally:
        try:
            if args.hand_motion:
                robot.left_hand.open()
            if args.motion:
                robot.stop()
        finally:
            robot.shutdown()


if __name__ == '__main__':
    main()
