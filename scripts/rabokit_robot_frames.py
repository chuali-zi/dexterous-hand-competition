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
CAMERA_FRAMES = ('base_cam', 'base_cam_optical', 'base_cam2', 'base_cam2_optical',
                 'base_cam3', 'base_cam3_optical')


def robot_frame(name):
    return name.startswith(ROBOT_PREFIXES) or name in CAMERA_FRAMES


def capture(seconds, trace_hand=False):
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from tf2_msgs.msg import TFMessage

    namespace = '/' + os.environ['ROS2_NAMESPACE'].strip('/')
    rclpy.init()
    node = rclpy.create_node('competition_robot_frames')
    frames = {}
    subscriptions = []
    hand_samples, commands, joint_samples = [], [], []
    started = time.monotonic()
    ready = False
    def message_info(info):
        return {key: info.get(key) for key in ('source_timestamp', 'received_timestamp',
                'publication_sequence_number', 'reception_sequence_number')}
    if trace_hand:
        from rabokit_hand import HAND_JOINTS
        from std_msgs.msg import Float64
        from sensor_msgs.msg import JointState
        hand_children = ['r136d7b_link_' + item[0] for item in HAND_JOINTS.values()]
        command_topic = f'{namespace}/arm_left/lh_thumb_cmc_yaw/cmd'
        command_topics = {name: f'{namespace}/arm_left/{name}/cmd'
                          for name in ('lh_thumb_cmc_yaw', 'lh_thumb_cmc_pitch', 'lh_thumb_ip')}
        def receive_command(name, msg, info):
            commands.append({'wall_seconds': time.monotonic() - started, 'joint': name, 'value': msg.data,
                             'message_info': message_info(info)})
        # Best-effort requests accept both reliable and best-effort publishers.
        for name, topic in command_topics.items():
            callback = lambda msg, info, name=name: receive_command(name, msg, info)
            subscriptions.append(node.create_subscription(Float64, topic, callback, qos_profile_sensor_data))
        def receive_joints(msg, info):
            elapsed = time.monotonic() - started
            if joint_samples and elapsed - joint_samples[-1]['wall_seconds'] < .25:
                return
            indices = {name: i for i, name in enumerate(msg.name) if name in HAND_JOINTS and i < len(msg.position)}
            joint_samples.append({'wall_seconds': elapsed,
                'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec],
                'position': {name: msg.position[i] for name, i in indices.items()},
                'velocity': {name: msg.velocity[i] for name, i in indices.items() if i < len(msg.velocity)},
                'effort': {name: msg.effort[i] for name, i in indices.items() if i < len(msg.effort)},
                'message_info': message_info(info)})
        subscriptions.append(node.create_subscription(JointState, f'{namespace}/arm_left/joint_states',
                                                     receive_joints, qos_profile_sensor_data))

    def receive(msg, info):
        for item in msg.transforms:
            parent, child = item.header.frame_id, item.child_frame_id
            if not robot_frame(child) or not (parent == 'world' or robot_frame(parent)):
                continue
            t, q = item.transform.translation, item.transform.rotation
            frames[child] = {'parent': parent, 'xyz': [t.x, t.y, t.z],
                             'xyzw': [q.x, q.y, q.z, q.w],
                             'stamp': [item.header.stamp.sec, item.header.stamp.nanosec]}
            if trace_hand and info is not None:
                frames[child]['message_info'] = message_info(info)

    try:
        for topic, qos in (
            ('tf_static', QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                    reliability=ReliabilityPolicy.RELIABLE)),
            ('tf', qos_profile_sensor_data),
        ):
            callback = receive if trace_hand else lambda msg: receive(msg, None)
            subscriptions.append(node.create_subscription(TFMessage, f'{namespace}/{topic}', callback, qos))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if trace_hand and all(name in frames for name in hand_children):
                if not ready:
                    print('HAND_TRACE_READY', flush=True)
                    ready = True
                elapsed = time.monotonic() - started
                if not hand_samples or elapsed - hand_samples[-1]['wall_seconds'] >= .25:
                    hand_samples.append({'wall_seconds': elapsed,
                        'frames': {name: frames[name] for name in hand_children},
                        'yaw_publisher_count': node.count_publishers(command_topic)})
        for name in ('base_cam', 'base_cam_optical', 'r6ef2dc_link_57d50b5359',
                     'r412d23_link_c7c36cf32f', 'rbd03eb_link_13df66db66'):
            if name not in frames:
                raise RuntimeError(f'Missing robot/camera transform: {name}')
        report = {'namespace': namespace, 'frames': frames,
                'source': 'robot kinematics and manually installed camera; no object transforms',
                'extrinsics_validated': False}
        if trace_hand:
            def publishers(topic):
                return [{'node': item.node_name, 'namespace': item.node_namespace,
                         'gid': list(item.endpoint_gid),
                         'reliability': getattr(item.qos_profile.reliability, 'name', str(item.qos_profile.reliability))}
                        for item in node.get_publishers_info_by_topic(topic)]
            report.update(hand_samples=hand_samples,
                          yaw_commands=[row for row in commands if row['joint'] == 'lh_thumb_cmc_yaw'],
                          yaw_command_topic=command_topic, thumb_commands=commands,
                          thumb_command_publishers={name: publishers(topic) for name, topic in command_topics.items()},
                          started_monotonic=started,
                          yaw_publishers=publishers(command_topic),
                          hand_joint_states=joint_samples,
                          tf_publishers=publishers(f'{namespace}/tf'),
                          joint_state_publishers=publishers(f'{namespace}/arm_left/joint_states'))
        return report
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=4)
    parser.add_argument('--hand-trace', action='store_true', help='Record hand TF, JointState and native thumb commands; read-only')
    args = parser.parse_args()
    if not os.getenv('ROS2_NAMESPACE'):
        parser.error('Run in the Rabokit cloud simulation terminal')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        parser.error('--seconds must be in (0, 30]')
    if args.output.exists():
        parser.error('--output must be a new file')
    report = capture(args.seconds, trace_hand=args.hand_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(report["frames"])} robot/camera frames saved to {args.output}')
    for name in ('r6ef2dc_link_57d50b5359', 'r412d23_link_c7c36cf32f',
                 'rbd03eb_link_13df66db66', 'base_cam', 'base_cam_optical'):
        print(name, json.dumps(report['frames'][name]))


if __name__ == '__main__':
    main()
