#!/usr/bin/env python3
"""Check a running Rabokit simulation; optionally move and restore one wrist."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import time


def observe(seconds):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node('competition_probe')
    samples = {}
    subscriptions = []
    namespace = '/' + os.environ['ROS2_NAMESPACE'].strip('/')
    try:
        for side in ('arm_left', 'arm_right'):
            def receive(msg, side=side):
                previous = samples.get(side, {})
                samples[side] = {'messages': previous.get('messages', 0) + 1,
                                 'names': list(msg.name), 'positions': list(msg.position)}
            subscriptions.append(node.create_subscription(
                JointState, f'{namespace}/{side}/joint_states', receive,
                qos_profile_sensor_data))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        topics = node.get_topic_names_and_types()
        cameras = [(name, types) for name, types in topics
                   if name.startswith(namespace + '/') and
                   any(t in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage',
                             'sensor_msgs/msg/CameraInfo') for t in types)]
        return {'joint_states': samples, 'camera_topics': cameras}
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def move_wrist():
    from rabo_robolib import LbotDualArm

    robot = LbotDualArm(mode='sim')
    arm = robot.left_arm
    try:
        time.sleep(1)
        start = list(arm.get_joint_angles() or [])
        if len(start) != 7 or not all(math.isfinite(q) for q in start):
            raise RuntimeError('Expected seven measured arm joint angles')
        target = start.copy()
        target[6] += 0.08
        if not arm.enable():
            raise RuntimeError('Arm enable failed')

        def move(goal):
            if not arm.move_joints(goal, speed=10, blocking=False):
                raise RuntimeError('Movement command was rejected')
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                actual = list(arm.get_joint_angles() or [])
                if len(actual) == 7 and all(math.isfinite(q) for q in actual):
                    if max(abs(a - b) for a, b in zip(actual, goal)) < 0.015:
                        return actual
                time.sleep(0.1)
            raise TimeoutError('Joint feedback did not reach the target in 12 seconds')

        reached = move(target)
        restored = move(start)
        return {'start': start, 'target': target, 'reached': reached,
                'restored': restored, 'joint_7_delta': reached[6] - start[6]}
    finally:
        try:
            robot.stop()
        finally:
            robot.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', action='store_true', help='Move left wrist +0.08 rad, then restore')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run this in the Rabokit cloud terminal with the simulation started')
    try:
        sdk_version = version('rabo_robolib')
    except PackageNotFoundError:
        sdk_version = None
    if args.motion and sdk_version is None:
        parser.error('Wrist motion requires rabo_robolib; install rabokit/requirements.txt first')
    report = {'sdk_version': sdk_version, 'observation': observe(4)}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not all(report['observation']['joint_states'].get(side, {}).get('messages', 0)
               for side in ('arm_left', 'arm_right')):
        raise RuntimeError('No joint feedback from both arms; check that the simulation is running')
    if args.motion:
        print(json.dumps({'motion': move_wrist()}, ensure_ascii=False), flush=True)
    print('PASS: Rabokit simulation probe' + (' and wrist motion/return' if args.motion else ''))


if __name__ == '__main__':
    main()
