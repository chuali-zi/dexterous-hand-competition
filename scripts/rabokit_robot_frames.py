#!/usr/bin/env python3
"""Record only known robot-link/camera transforms from the current brick scene."""
import argparse
import json
import math
import os
from pathlib import Path
import time

# Actual robot link IDs in the user's copied LBot scene; object frames are excluded.
ROBOT_PREFIXES = ('r6ef2dc_link_', 'r412d23_link_', 'rbd03eb_link_',
                  'r136d7b_link_', 'rcd72e2_link_')
CAMERA_FRAMES = ('base_cam', 'base_cam_optical', 'base_cam2', 'base_cam2_optical')


def robot_frame(name):
    return name.startswith(ROBOT_PREFIXES) or name in CAMERA_FRAMES


def capture(seconds):
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from tf2_msgs.msg import TFMessage

    namespace = '/' + os.environ['ROS2_NAMESPACE'].strip('/')
    rclpy.init()
    node = rclpy.create_node('competition_robot_frames')
    frames = {}
    subscriptions = []

    def receive(msg):
        for item in msg.transforms:
            parent, child = item.header.frame_id, item.child_frame_id
            if not robot_frame(child) or not (parent == 'world' or robot_frame(parent)):
                continue
            t, q = item.transform.translation, item.transform.rotation
            frames[child] = {'parent': parent, 'xyz': [t.x, t.y, t.z],
                             'xyzw': [q.x, q.y, q.z, q.w],
                             'stamp': [item.header.stamp.sec, item.header.stamp.nanosec]}

    try:
        for topic, qos in (
            ('tf_static', QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                    reliability=ReliabilityPolicy.RELIABLE)),
            ('tf', qos_profile_sensor_data),
        ):
            subscriptions.append(node.create_subscription(TFMessage, f'{namespace}/{topic}', receive, qos))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        for name in ('base_cam', 'base_cam_optical', 'r6ef2dc_link_57d50b5359',
                     'r412d23_link_c7c36cf32f', 'rbd03eb_link_13df66db66'):
            if name not in frames:
                raise RuntimeError(f'Missing robot/camera transform: {name}')
        return {'namespace': namespace, 'frames': frames,
                'source': 'robot kinematics and manually installed camera; no object transforms',
                'extrinsics_validated': False}
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=4)
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run in the Rabokit cloud simulation terminal')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        parser.error('--seconds must be in (0, 30]')
    if args.output.exists():
        parser.error('--output must be a new file')
    report = capture(args.seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(report["frames"])} robot/camera frames saved to {args.output}')
    for name in ('r6ef2dc_link_57d50b5359', 'r412d23_link_c7c36cf32f',
                 'rbd03eb_link_13df66db66', 'base_cam', 'base_cam_optical'):
        print(name, json.dumps(report['frames'][name]))


if __name__ == '__main__':
    main()
