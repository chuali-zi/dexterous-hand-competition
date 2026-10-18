#!/usr/bin/env python3
"""Observe empty-hand goal holding at the tested initial arm pose; --motion is explicit."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from rabokit_calibration_probe import hand_command, initialize_empty_hand, measured
from rabokit_hand import hand_feedback, targets


def run_motion(robot, output, seconds=20, prepared_thumb=False, force_limit=1., sdk_close=False, thumb_yaw=None, ramp_seconds=0.):
    arm = robot.left_arm
    before = measured(arm)
    if any(abs(value) > .01 for value in before['joints']):
        raise ValueError('Hand hold diagnostic requires the tested initial arm pose')
    if not math.isfinite(force_limit) or not 0 < force_limit <= 1:
        raise ValueError('Normalized force limit must be in (0, 1]')
    if not math.isfinite(ramp_seconds) or not 0 <= ramp_seconds < 1:
        raise ValueError('Ramp seconds must be in [0,1)')
    if sdk_close and (thumb_yaw is not None or ramp_seconds):
        raise ValueError('Thumb yaw override and ramp require native commands')
    pose = 'close' if sdk_close else 'pinch_thumb_ready_without_pinky' if prepared_thumb else 'pinch_preshape_without_pinky'
    goals = targets(pose, thumb_yaw)
    report = {'before': before, 'arm_motion_commanded': False, 'official_score': None,
              'pose': pose, 'native_targets_rad': goals, 'probe_force_limit_normalized': force_limit,
              'initialization_force_limit_normalized': 1.,
              'ramp_seconds': ramp_seconds,
              'command_driver': 'sdk_close' if sdk_close else 'native_float64'}
    trace = None
    ready_line = ''
    started = time.monotonic()
    try:
        if not arm.enable() or not robot.left_hand.set_force([1.] * 6):
            raise RuntimeError('Arm enable or hand force-limit setup failed')
        report['initialization'] = initialize_empty_hand(arm, output / 'hand-initialization')
        trace = subprocess.Popen([sys.executable, str(Path(__file__).with_name('rabokit_robot_frames.py')),
            '--hand-trace', '--seconds', str(seconds), '--output', str(output / 'trace.json')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready_line = trace.stdout.readline()
        if ready_line.strip() != 'HAND_TRACE_READY':
            raise RuntimeError('Hand TF recorder was not ready; preshape not commanded')
        if force_limit != 1.:
            report['force_limit_command_monotonic'] = time.monotonic()
            report['force_limit_request_accepted'] = robot.left_hand.set_force([force_limit] * 6)
            if not report['force_limit_request_accepted']:
                raise RuntimeError('SDK rejected the probe force-limit request')
        report['preshape_command_wall_seconds'] = time.monotonic() - started
        report['preshape_command_monotonic'] = time.monotonic()
        try:
            if sdk_close:
                report['sdk_close_request_accepted'] = robot.left_hand.close(blocking=False)
                if not report['sdk_close_request_accepted']:
                    raise RuntimeError('SDK close request was rejected')
            elif prepared_thumb:
                hand_command(pose, thumb_yaw=thumb_yaw, ramp_seconds=ramp_seconds)
            else:
                report['preshape_arrival'] = hand_command(pose, thumb_yaw=thumb_yaw, wait_arrival=True, ramp_seconds=ramp_seconds)
        except (RuntimeError, subprocess.SubprocessError) as error:
            report['preshape_error'] = f'{type(error).__name__}: {error}'
        report['command_return_monotonic'] = time.monotonic()
        if not sdk_close:
            report['preshape_process_exit_wall_seconds'] = time.monotonic() - started
            report['preshape_process_exit_monotonic'] = report['command_return_monotonic']
        stdout, stderr = trace.communicate(timeout=seconds + 5)
        (output / 'trace.log').write_text(ready_line + stdout + stderr, encoding='utf-8')
        if trace.returncode:
            raise RuntimeError(f'Hand recorder exited {trace.returncode}')
        data = json.loads((output / 'trace.json').read_text(encoding='utf-8'))
        report['post_process_preshape'] = hand_feedback(data['frames'], goals)
        last = report['post_process_preshape']
        feedbacks = [hand_feedback(row['frames'], goals) for row in data['hand_samples']] + [last]
        recent = [row for row in feedbacks if row['complete'] and last['complete'] and
                  last['stamp_min_sec'] - 1.5 <= row['stamp_min_sec'] <= last['stamp_min_sec']]
        report['post_command_wall_seconds'] = (data['started_monotonic'] +
            data['hand_samples'][-1]['wall_seconds'] - report['command_return_monotonic'])
        report['preshape_hold_verified'] = bool(recent and
            recent[-1]['stamp_min_sec'] - recent[0]['stamp_min_sec'] >= 1 and
            report['post_command_wall_seconds'] >= 1 and
            all(row['maximum_error_rad'] < .03 and row['stamp_max_sec'] - row['stamp_min_sec'] < .1 for row in recent) and
            all(max(row['joint_rotation_error_rad'][name] for row in recent) -
                min(row['joint_rotation_error_rad'][name] for row in recent) <= .005 for name in goals))
        if report.get('preshape_error') or not report['preshape_hold_verified']:
            raise RuntimeError('Hand target was not verified through the later observation; see saved trace')
        return report
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if trace is not None and trace.poll() is None:
            trace.terminate()
            try:
                trace.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                trace.kill()
                trace.communicate(timeout=3)
        try:
            if ramp_seconds:
                report['recovery_kind'] = 'direct_ramped_open'
                report['reopened'] = hand_command('open', wait_open=True, ramp_seconds=ramp_seconds)
            elif report.get('force_limit_request_accepted') and force_limit != 1.:
                report['recovery_kind'] = 'direct_open_after_force_limit_request'
                report['reopened'] = hand_command('open', wait_open=True)
            else:
                report['recovery_kind'] = 'initialization_close_then_open'
                report['recovery'] = initialize_empty_hand(arm, output / 'hand-recovery')
                report['reopened'] = report['recovery']['opened']
            report['after'] = measured(arm)
            report['arm_joint_drift_rad'] = max(abs(a-b) for a, b in zip(before['joints'], report['after']['joints']))
            if report['arm_joint_drift_rad'] > .005:
                raise RuntimeError('Arm drifted during the hand-only diagnostic')
        except Exception as error:
            report['reopen_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            if 'after' not in report:
                try:
                    report['after'] = measured(arm)
                    report['arm_joint_drift_rad'] = max(abs(a-b) for a, b in zip(before['joints'], report['after']['joints']))
                except RuntimeError as error:
                    report['arm_feedback_error'] = str(error)
            report['wall_seconds'] = time.monotonic() - started
            (output / 'hold.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--motion', action='store_true', help='Initialize, preshape, observe and reopen; no arm targets')
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument('--prepared-thumb', action='store_true',
                        help='At the initial arm pose, compare 45%% thumb bending instead of an open thumb')
    shape.add_argument('--sdk-close', action='store_true', help='Compare the SDK close route at the initial arm pose')
    parser.add_argument('--thumb-yaw', type=float, help='Native preshape yaw endpoint in radians; excludes SDK close')
    parser.add_argument('--ramp-seconds', type=float, default=0., help='Native ramp duration in [0,1) seconds')
    parser.add_argument('--force-limit', type=float, default=1.,
                        help='SDK normalized limit for all six hand groups during observation; initialization uses 1')
    args = parser.parse_args()
    if not 15 <= args.seconds <= 25:
        parser.error('--seconds must be in [15, 25]')
    if not math.isfinite(args.force_limit) or not 0 < args.force_limit <= 1:
        parser.error('--force-limit must be in (0, 1]')
    if not math.isfinite(args.ramp_seconds) or not 0 <= args.ramp_seconds < 1:
        parser.error('--ramp-seconds must be in [0,1)')
    if args.sdk_close and args.ramp_seconds:
        parser.error('--ramp-seconds excludes --sdk-close')
    if args.thumb_yaw is not None:
        if args.sdk_close:
            parser.error('--thumb-yaw excludes --sdk-close')
        try:
            targets('close', args.thumb_yaw)
        except ValueError as error:
            parser.error(str(error))
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run only in the Rabokit simulation terminal')
    if args.output.exists():
        parser.error('--output must be a new directory')
    args.output.mkdir(parents=True)
    if not args.motion:
        subprocess.run([sys.executable, str(Path(__file__).with_name('rabokit_robot_frames.py')),
            '--hand-trace', '--seconds', str(args.seconds), '--output', str(args.output / 'trace.json')],
            check=True, timeout=args.seconds + 8)
        return
    from rabo_robolib import LbotDualArm
    robot = LbotDualArm(mode='sim')
    try:
        time.sleep(1)
        run_motion(robot, args.output, args.seconds, args.prepared_thumb, args.force_limit, args.sdk_close,
                   args.thumb_yaw, args.ramp_seconds)
        print('PASS: initial-pose hand observation and verified reopening; no grasp or score', flush=True)
    finally:
        robot.shutdown()


if __name__ == '__main__':
    main()
