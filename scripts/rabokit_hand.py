#!/usr/bin/env python3
"""Publish measured SDK-equivalent left-hand targets in the Rabokit simulation."""
import argparse
import json
import os
import time


def targets(pose):
    # Native radians observed on all 11 SDK /cmd topics in this scene, SDK 3.0.0.
    if pose == 'open':
        mcp, dip, yaw, pitch, ip = -.082, -.0837, .34705882352941186, -.0922, -.0872
    elif pose == 'close':
        mcp, dip, yaw, pitch, ip = 1.7, 1.53, .511764705882353, .4772, .8472
    else:
        raise ValueError('Expected open or close')
    result = {f'lh_{finger}_{joint}': value for finger in ('index', 'middle', 'ring', 'pinky')
              for joint, value in (('mcp_pitch', mcp), ('dip', dip))}
    result.update(lh_thumb_cmc_yaw=yaw, lh_thumb_cmc_pitch=pitch, lh_thumb_ip=ip)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', choices=('open', 'close'), required=True)
    args = parser.parse_args()
    namespace = os.getenv('ROS2_NAMESPACE', '').strip('/')
    if not namespace:
        parser.error('Run only in the Rabokit simulation terminal')
    import rclpy
    from std_msgs.msg import Float64

    values = targets(args.pose)
    rclpy.init()
    node = rclpy.create_node('competition_left_hand_targets')
    try:
        topics = {name: f'/{namespace}/arm_left/{name}/cmd' for name in values}
        deadline = time.monotonic() + 3
        available = dict(node.get_topic_names_and_types())
        while any(available.get(topic) != ['std_msgs/msg/Float64'] for topic in topics.values()):
            if time.monotonic() >= deadline:
                raise RuntimeError('Expected the current simulation left-hand Float64 command topics')
            rclpy.spin_once(node, timeout_sec=.05)
            available = dict(node.get_topic_names_and_types())
        publishers = {name: node.create_publisher(Float64, topic, 10) for name, topic in topics.items()}
        deadline = time.monotonic() + 3
        while not all(pub.get_subscription_count() for pub in publishers.values()):
            if time.monotonic() >= deadline:
                raise TimeoutError('Left-hand command subscribers were not ready')
            rclpy.spin_once(node, timeout_sec=.05)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            for name, pub in publishers.items():
                pub.publish(Float64(data=values[name]))
            rclpy.spin_once(node, timeout_sec=.05)
        print(json.dumps({'pose': args.pose, 'native_targets_rad': values,
                          'command_published': True, 'hand_arrival_verified': False}), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
