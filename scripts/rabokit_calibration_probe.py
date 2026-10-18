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


def sample(arm, root, label, stereo=False, side_camera=False, hand_trace=False):
    folder = root / label
    folder.mkdir()
    before = measured(arm)
    scripts = Path(__file__).resolve().parent
    camera_command = ([str(scripts / 'rabokit_stereo.py'), '--capture-only', '--seconds', '6' if side_camera else '1',
                       '--output', str(folder / 'camera')] if stereo else
                      [str(scripts / 'rabokit_camera.py'), '--seconds', '1',
                       '--topic', 'base/base_cam/image_raw', '--output', str(folder / 'camera')])
    if side_camera:
        if not stereo:
            raise ValueError('Side-view sampling requires the synchronized stereo capture')
        camera_command.append('--side-camera')
    frame_command = [str(scripts / 'rabokit_robot_frames.py'), '--seconds', '2',
                     '--output', str(folder / 'robot-frames.json')]
    if hand_trace:
        frame_command.append('--hand-trace')
    for command in (frame_command, camera_command):
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
            if not robot.left_hand.set_force([1.] * 6):
                raise RuntimeError('Simulation hand force-limit setup was not acknowledged')
            print(json.dumps({'hand_force_limit_normalized': [1.] * 6}), flush=True)
            for label, pose in (('opened', 'open'), ('closed', 'close'), ('reopened', 'open')):
                hand_command(pose)
                time.sleep(1)
                record = sample(arm, args.output, label)
                record['hand_command_published'] = True
                record['hand_arrival_verified'] = False
                records.append(record)
        report = {'motion_requested': args.motion or args.hand_motion,
                  'motion_kind': 'wrist' if args.motion else 'empty_hand' if args.hand_motion else None,
                  'samples': records,
                  'extrinsics_validated': False, 'official_score': None}
        if args.hand_motion:
            from rabokit_calibration_analysis import compare_hand_frames

            views = [json.loads((args.output / label / 'robot-frames.json').read_text(encoding='utf-8'))['frames']
                     for label in ('opened', 'closed', 'reopened')]
            report['hand_geometry'] = compare_hand_frames(*views)
            report['hand_driver'] = 'native Float64 targets measured from SDK 3.0.0 commands'
            report['hand_force_limit_normalized'] = [1.] * 6
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if args.hand_motion:
            print(json.dumps(report['hand_geometry']), flush=True)
            if not report['hand_geometry']['sequence_verified']:
                raise RuntimeError('Measured hand links did not close and return within 5 mm')
        print('PASS: static calibration samples' + (' and wrist motion/return' if args.motion else
                                                  ' and empty hand command sequence' if args.hand_motion else ''))
    finally:
        try:
            if args.hand_motion:
                hand_command('open')
                time.sleep(1)
            if args.motion:
                robot.stop()
        finally:
            robot.shutdown()


def hand_command(pose, thumb_yaw=None, wait_open=False, wait_arrival=False, ramp_seconds=0.):
    command = [sys.executable, str(Path(__file__).with_name('rabokit_hand.py')), '--pose', pose]
    if thumb_yaw is not None:
        command.extend(('--thumb-yaw', str(thumb_yaw)))
    if ramp_seconds:
        command.extend(('--ramp-seconds', str(ramp_seconds)))
    if wait_open:
        command.append('--wait-open')
    if wait_arrival:
        command.append('--wait-arrival')
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=25 if ramp_seconds else 22 if wait_open or wait_arrival else 10)
    print(result.stdout + result.stderr, end='', flush=True)
    result.check_returncode()
    if wait_open or wait_arrival:
        feedback = json.loads(result.stdout.strip().splitlines()[-1])
        if not feedback.get('hand_arrival_verified'):
            raise RuntimeError('Hand subprocess did not verify arrival')
        return feedback


def initialize_empty_hand(arm, output, ramp_seconds=0.):
    """Initialize at the tested rear pose; an optional ramp opens directly."""
    from rabokit_hand import hand_feedback, targets

    initial = measured(arm)
    if any(abs(q) > .01 for q in initial['joints']):
        raise ValueError('Empty-hand initialization requires the tested initial arm pose')
    output = Path(output)
    output.mkdir()
    report = {'initial': initial, 'arm_motion_commanded': False, 'initialized': False}

    def capture_frames(label):
        frame_file = output / (label + '-frames.json')
        subprocess.run([sys.executable, str(Path(__file__).with_name('rabokit_robot_frames.py')),
                        '--seconds', '1', '--output', str(frame_file)],
                       capture_output=True, text=True, check=True, timeout=8)
        return json.loads(frame_file.read_text(encoding='utf-8'))['frames']

    try:
        if ramp_seconds:
            report['kind'] = 'direct_ramped_open'
            report['opened'] = hand_command('open', wait_open=True, ramp_seconds=ramp_seconds)
        else:
            report['initial_open_error'] = hand_feedback(capture_frames('initial'), targets('open'))
            try:
                hand_command('close', thumb_yaw=.62)
                time.sleep(1)
                report['closed'] = hand_feedback(capture_frames('closed'), targets('close', .62))
                if not report['closed']['complete'] or report['closed']['maximum_error_rad'] >= .03:
                    raise RuntimeError('Empty-hand closure was not verified; no arm motion started')
            finally:
                report['opened'] = hand_command('open', wait_open=True)
        report['final'] = measured(arm)
        report['arm_joint_drift_rad'] = max(abs(a-b) for a,b in
            zip(initial['joints'], report['final']['joints']))
        if report['arm_joint_drift_rad'] > .005:
            raise RuntimeError('Arm drifted during empty-hand initialization')
        report['initialized'] = True
        return report
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (output / 'initialization.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
