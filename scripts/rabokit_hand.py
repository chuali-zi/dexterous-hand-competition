#!/usr/bin/env python3
"""Publish SDK-equivalent left-hand targets; requires simulation hand force setup."""
import argparse
import json
import math
import os
import time


def targets(pose, thumb_yaw=None):
    # Native radians observed on all 11 SDK /cmd topics in this scene, SDK 3.0.0.
    if pose == 'open':
        mcp, dip, yaw, pitch, ip = -.082, -.0837, .34705882352941186, -.0922, -.0872
    elif pose in ('close', 'close_without_pinky'):
        mcp, dip, yaw, pitch, ip = 1.7, 1.53, .511764705882353, .4772, .8472
    else:
        raise ValueError('Expected open, close or close_without_pinky')
    result = {f'lh_{finger}_{joint}': value for finger in ('index', 'middle', 'ring', 'pinky')
              for joint, value in (('mcp_pitch', mcp), ('dip', dip))}
    result.update(lh_thumb_cmc_yaw=yaw, lh_thumb_cmc_pitch=pitch, lh_thumb_ip=ip)
    if pose == 'close_without_pinky':
        result.update(lh_pinky_mcp_pitch=-.082, lh_pinky_dip=-.0837)
    if thumb_yaw is not None:
        if pose == 'open' or not math.isfinite(thumb_yaw) or not 0 <= thumb_yaw <= 1.3:
            raise ValueError('Thumb yaw override requires closure and a finite angle in [0, 1.3] rad')
        result['lh_thumb_cmc_yaw'] = thumb_yaw
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', choices=('open', 'close', 'close_without_pinky'), required=True)
    parser.add_argument('--thumb-yaw', type=float, help='Closure-only thumb CMC yaw target in native radians')
    args = parser.parse_args()
    try:
        values = targets(args.pose, args.thumb_yaw)
    except ValueError as error:
        parser.error(str(error))
    namespace = os.getenv('ROS2_NAMESPACE', '').strip('/')
    if not namespace:
        parser.error('Run only in the Rabokit simulation terminal')
    import rclpy
    from std_msgs.msg import Float64

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
