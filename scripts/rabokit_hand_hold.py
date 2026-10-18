#!/usr/bin/env python3
"""Observe an empty-hand preshape at the tested initial arm pose; --motion is explicit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rabokit_calibration_probe import hand_command, initialize_empty_hand, measured
from rabokit_hand import hand_feedback, targets


def run_motion(robot, output, seconds=20):
    arm = robot.left_arm
    before = measured(arm)
    if any(abs(value) > .01 for value in before['joints']):
        raise ValueError('Hand hold diagnostic requires the tested initial arm pose')
    report = {'before': before, 'arm_motion_commanded': False, 'official_score': None}
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
        report['preshape_command_wall_seconds'] = time.monotonic() - started
        report['preshape_command_monotonic'] = time.monotonic()
        try:
            report['preshape_arrival'] = hand_command('pinch_preshape_without_pinky', wait_arrival=True)
        except (RuntimeError, subprocess.SubprocessError) as error:
            report['preshape_error'] = f'{type(error).__name__}: {error}'
        report['preshape_process_exit_wall_seconds'] = time.monotonic() - started
        report['preshape_process_exit_monotonic'] = time.monotonic()
        stdout, stderr = trace.communicate(timeout=seconds + 5)
        (output / 'trace.log').write_text(ready_line + stdout + stderr, encoding='utf-8')
        if trace.returncode:
            raise RuntimeError(f'Hand recorder exited {trace.returncode}')
        data = json.loads((output / 'trace.json').read_text(encoding='utf-8'))
        report['post_process_preshape'] = hand_feedback(data['frames'], targets('pinch_preshape_without_pinky'))
        if (report.get('preshape_error') or not report['post_process_preshape']['complete'] or
                report['post_process_preshape']['maximum_error_rad'] >= .03 or
                report['post_process_preshape']['stamp_min_sec'] <
                    report['preshape_arrival']['feedback']['stamp_max_sec'] + 1):
            raise RuntimeError('Preshape was not verified through the later observation; see saved trace')
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
            report['reopened'] = hand_command('open', wait_open=True)
            report['after'] = measured(arm)
            report['arm_joint_drift_rad'] = max(abs(a-b) for a, b in zip(before['joints'], report['after']['joints']))
            if report['arm_joint_drift_rad'] > .005:
                raise RuntimeError('Arm drifted during the hand-only diagnostic')
        except Exception as error:
            report['reopen_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            report['wall_seconds'] = time.monotonic() - started
            (output / 'hold.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--motion', action='store_true', help='Initialize, preshape, observe and reopen; no arm targets')
    args = parser.parse_args()
    if not 15 <= args.seconds <= 25:
        parser.error('--seconds must be in [15, 25]')
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
        run_motion(robot, args.output, args.seconds)
        print('PASS: initial-pose hand observation and verified reopening; no grasp or score', flush=True)
    finally:
        robot.shutdown()


if __name__ == '__main__':
    main()
