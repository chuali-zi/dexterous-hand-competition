#!/usr/bin/env python3
"""Experimental bounded DIP feedback hold for the two-lift simulation diagnostic.

Joint target minus measured position is an uncalibrated control signal, not
contact force, contact identity or evidence of a successful grasp.
"""
import argparse
from collections import deque
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from rabokit_hand import HAND_JOINTS, targets

DIPS = ('lh_index_dip', 'lh_middle_dip')
ERROR_GOAL = .004
STEP = .002
MAX_INCREMENT = .02


def adjusted_targets(initial, current, positions):
    """Advance a DIP with small target/position error; retain the other nine goals."""
    if any(not math.isfinite(positions.get(name, float('nan'))) for name in HAND_JOINTS):
        raise ValueError('Expected complete finite hand positions')
    result = dict(initial)
    for name in DIPS:
        if not initial[name] <= current[name] <= initial[name] + MAX_INCREMENT + 1e-12:
            raise ValueError('DIP hold target exceeds its bounded increment')
        value = current[name]
        if value - positions[name] < ERROR_GOAL:
            value = min(initial[name] + MAX_INCREMENT, value + STEP)
        result[name] = value
    return result


def run(output, seconds=60):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64

    namespace = '/' + os.environ['ROS2_NAMESPACE'].strip('/')
    initial = targets('pinch_without_pinky', pad_grasp=True)
    current, history = dict(initial), deque()
    stopping = [False]
    started, last_received = time.monotonic(), [None]
    report = {'pid': os.getpid(), 'status': 'starting', 'initial_targets_rad': initial,
              'error_goal_rad': ERROR_GOAL, 'step_rad': STEP, 'max_dip_increment_rad': MAX_INCREMENT,
              'scope': __doc__, 'physical_grasp_verified': False,
              'joint_samples': [], 'issued_commands': [], 'adjustments': [], 'decision_count': 0}
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    previous_handler = signal.getsignal(signal.SIGTERM)
    node = None
    try:
        rclpy.init()
        signal.signal(signal.SIGTERM, lambda *args: stopping.__setitem__(0, True))
        node = rclpy.create_node('competition_bounded_grip_hold')
        def receive(msg):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if history and stamp <= history[-1]['seconds']:
                return
            indices = {name: i for i, name in enumerate(msg.name) if name in HAND_JOINTS}
            fields = {field: {name: getattr(msg, field)[i] for name, i in indices.items()
                              if i < len(getattr(msg, field))} for field in ('position', 'velocity', 'effort')}
            if any(not math.isfinite(fields['position'].get(name, float('nan'))) for name in HAND_JOINTS):
                raise RuntimeError('Incomplete/nonfinite hand feedback; hold stopped')
            last_received[0] = time.monotonic()
            sample = {'wall_seconds': last_received[0] - started,
                      'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec], **fields}
            report['joint_samples'].append(sample)
            history.append({'seconds': stamp, 'position': fields['position']})
            while history[0]['seconds'] < stamp - .3:
                history.popleft()

        subscription = node.create_subscription(JointState, f'{namespace}/arm_left/joint_states',
                                                receive, qos_profile_sensor_data)
        publishers = {name: node.create_publisher(Float64, f'{namespace}/arm_left/{name}/cmd', 10)
                      for name in initial}
        while not history or not all(pub.get_subscription_count() for pub in publishers.values()):
            if time.monotonic() - started > 3:
                raise TimeoutError('Hand feedback/command subscribers unavailable; no targets published')
            rclpy.spin_once(node, timeout_sec=.05)
        next_adjust = history[-1]['seconds'] + .25
        last_decision_wall = time.monotonic()
        next_publish = -1.
        while not stopping[0]:
            if time.monotonic() - started >= seconds:
                raise TimeoutError('Feedback hold reached its wall-time limit')
            rclpy.spin_once(node, timeout_sec=.02)
            if time.monotonic() - last_received[0] > 2:
                raise TimeoutError('Hand feedback stopped advancing; hold stopped')
            if time.monotonic() - last_decision_wall > 3:
                raise TimeoutError('Insufficient recent hand history; hold stopped')
            stamp = history[-1]['seconds']
            recent = [row for row in history if row['seconds'] >= stamp - .2]
            if stamp >= next_adjust and stamp - recent[0]['seconds'] >= .16:
                positions = {name: statistics.median(row['position'][name] for row in recent)
                             for name in initial}
                updated = adjusted_targets(initial, current, positions)
                if updated != current:
                    report['adjustments'].append({'joint_reference_seconds': stamp,
                        'previous_targets_rad': current, 'targets_rad': updated,
                        'median_position_rad': positions})
                    current = updated
                next_adjust = stamp + .25
                report['decision_count'] += 1
                last_decision_wall = time.monotonic()
            if stamp >= next_publish:
                for name, pub in publishers.items():
                    pub.publish(Float64(data=current[name]))
                report['issued_commands'].append({'wall_seconds': time.monotonic() - started,
                    'joint_reference_seconds': stamp, 'targets_rad': dict(current)})
                if report['status'] == 'starting':
                    report['status'] = 'running'
                    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                next_publish = stamp + .05
        report['status'] = 'stopped'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(final_targets_rad=current, wall_seconds=time.monotonic() - started)
        try:
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.try_shutdown()
            signal.signal(signal.SIGTERM, previous_handler)


@contextmanager
def hold_process(root):
    """Own the publisher through sampling/lifts and stop it before SDK cleanup."""
    output = root / 'grip-hold.json'
    if output.exists():
        raise FileExistsError(output)
    with (root / 'grip-hold.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(output)],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 5
            while True:
                check_process(process)
                try:
                    ready = json.loads(output.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    ready = {}
                if ready.get('pid') == process.pid and ready.get('status') == 'running':
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Feedback publisher did not become ready')
                time.sleep(.05)
            yield process
        finally:
            caller_failed = sys.exc_info()[0] is not None
            if process.poll() is None:
                process.terminate()
                try:
                    exit_code = process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError('Feedback publisher required forced termination')
                if exit_code != 0 and not caller_failed:
                    raise RuntimeError(f'Feedback publisher stopped with exit code {exit_code}')
            elif not caller_failed:
                check_process(process)


def check_process(process):
    if process is not None and process.poll() is not None:
        raise RuntimeError(f'Feedback publisher exited unexpectedly ({process.returncode})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run only in the Rabokit simulation terminal')
    if args.output.exists():
        parser.error('--output must be a new file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.output)


if __name__ == '__main__':
    main()
