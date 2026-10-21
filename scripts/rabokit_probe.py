#!/usr/bin/env python3
"""Check a running Rabokit simulation; optionally move and restore one wrist."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
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


def rgb_ready(seconds=20):
    """Wait once for three synchronized RGB pairs and fresh robot feedback; no SDK."""
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from rabokit_camera import camera_publishers, rgb_bytes
    from rabokit_hand import HAND_JOINTS

    namespace = '/' + os.environ['ROS2_NAMESPACE'].strip('/')
    topics = {name: (f'{namespace}/base/{name}/image_raw', f'{namespace}/base/{name}/camera_info')
              for name in ('base_cam', 'base_cam2', 'base_cam3')}
    counts, stamps, pending, joints, subscriptions = {}, {}, {}, {}, []
    matched = [None]
    started = time.monotonic()
    rclpy.init()
    node = rclpy.create_node('competition_rgb_readiness')
    try:
        def receive_joint(side, msg):
            values = dict(zip(msg.name, msg.position))
            required = [f'{side}_J{i}' for i in range(1, 8)]
            if side == 'arm_left':
                required.extend(HAND_JOINTS)
            if any(not math.isfinite(values.get(n, float('nan'))) for n in required):
                raise ValueError(f'Incomplete or nonfinite {side} feedback')
            counts[side] = counts.get(side, 0) + 1
            stamps[side] = [msg.header.stamp.sec, msg.header.stamp.nanosec]
            joints[side] = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                            time.monotonic())

        def receive_camera(name, kind, msg):
            key = name + '/' + kind
            counts[key] = counts.get(key, 0) + 1
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            stamps[key] = list(stamp)
            group = pending.setdefault(stamp, {})
            group.setdefault(name, {})[kind] = msg
            if all(set(group.get(n, {})) == {'image', 'info'} for n in topics):
                for pair in group.values():
                    image, info = pair['image'], pair['info']
                    if (not image.header.frame_id or image.header.frame_id != info.header.frame_id
                            or (image.width, image.height) != (info.width, info.height)
                            or len(info.k) != 9 or not all(math.isfinite(v) for v in info.k)
                            or info.k[0] <= 0 or info.k[4] <= 0):
                        raise ValueError('Inconsistent RGB/CameraInfo frame, dimensions or intrinsics')
                    rgb_bytes(image)
                if matched[0] is None or stamp >= matched[0][0]:
                    matched[0] = (stamp, time.monotonic())
            while len(pending) > 16:
                del pending[next(iter(pending))]

        for side in ('arm_left', 'arm_right'):
            subscriptions.append(node.create_subscription(JointState, f'{namespace}/{side}/joint_states',
                lambda msg, side=side: receive_joint(side, msg), qos_profile_sensor_data))
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        for name, pair in topics.items():
            for kind, topic, message in zip(('image', 'info'), pair, (Image, CameraInfo)):
                subscriptions.append(node.create_subscription(message, topic,
                    lambda msg, name=name, kind=kind: receive_camera(name, kind, msg), qos))
        ready = False
        while time.monotonic() - started < seconds:
            rclpy.spin_once(node, timeout_sec=.1)
            if matched[0] is not None:
                stamp, received = matched[0]
                frame_seconds = stamp[0] + stamp[1] * 1e-9
                now = time.monotonic()
                ready = (now - received <= 1 and all(side in joints
                    and now - joints[side][1] <= 1 and abs(joints[side][0] - frame_seconds) <= .5
                    for side in ('arm_left', 'arm_right')))
                if ready:
                    break
        expected = ['arm_left', 'arm_right', *(n + '/' + k for n in topics for k in ('image', 'info'))]
        return {'ready': ready, 'namespace': namespace, 'wall_seconds': time.monotonic() - started,
                'message_counts': counts, 'last_stamps': stamps, 'required_cameras': list(topics),
                'matched_stamp': list(matched[0][0]) if matched[0] else None,
                'missing_messages': [key for key in expected if not counts.get(key)],
                'camera_publishers': camera_publishers(node, topics), 'motion_commanded': False,
                'scope': 'One read-only readiness snapshot; not sustained stream, motion or grasp verification.'}
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--motion', action='store_true', help='Move left wrist +0.08 rad, then restore')
    mode.add_argument('--rgb-ready', action='store_true', help='Wait up to 20s once for three fixed RGB cameras and robot feedback; no motion')
    parser.add_argument('--output', type=Path, help='New JSON file for --rgb-ready; never overwrite')
    args = parser.parse_args()
    if args.output and (not args.rgb_ready or args.output.exists()):
        parser.error('--output requires --rgb-ready and a new file')
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run this in the Rabokit cloud terminal with the simulation started')
    if args.rgb_ready:
        report = rgb_ready()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(report, indent=2) + '\n')
        print('READINESS_RESULT=' + json.dumps(report), flush=True)
        if not report['ready']:
            raise SystemExit(1)
        return
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
