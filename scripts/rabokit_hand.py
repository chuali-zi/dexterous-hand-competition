#!/usr/bin/env python3
"""Publish SDK-equivalent left-hand targets; requires simulation hand force setup."""
import argparse
import json
import math
import os
import time


# Parent-relative revolute joints in the current O6 scene. Fixed rotations agree
# with recorded robot TF and the vendor O6 URDF; no object frames are consumed.
HAND_JOINTS = {
    'lh_thumb_cmc_yaw': ('31fc6c92a4', '2848c4d4a8', (0., 0., 0.), 2),
    'lh_thumb_cmc_pitch': ('b0fdbe757f', '31fc6c92a4', (0., -1.1529, 2.0944), 1),
    'lh_thumb_ip': ('2a0b523d7a', 'b0fdbe757f', (0., 0., 0.), 1),
}
for _finger, _proximal, _distal, _roll in (
        ('index', 'ce25aa231d', 'c58a174e90', .05236),
        ('middle', '5edf37010f', 'bbef50a1fb', 0.),
        ('ring', '5a6f84872e', '18bb8bf370', -.05236),
        ('pinky', 'b784b556bb', '5d106ae0c2', -.087266)):
    HAND_JOINTS[f'lh_{_finger}_mcp_pitch'] = (_proximal, '2848c4d4a8', (_roll, 0., 0.), 1)
    HAND_JOINTS[f'lh_{_finger}_dip'] = (_distal, _proximal, (0., 0., 0.), 1)


def quaternion_product(a, b):
    x, y, z, w = a
    u, v, t, s = b
    return (w*u + x*s + y*t - z*v, w*v - x*t + y*s + z*u,
            w*t + x*v - y*u + z*s, w*s - x*u - y*v - z*t)


def target_quaternion(origin, axis, value):
    q = (0., 0., 0., 1.)
    # URDF RPY is Rz(yaw) Ry(pitch) Rx(roll), followed by the joint rotation.
    for index, angle in [(2, origin[2]), (1, origin[1]), (0, origin[0]), (axis, value)]:
        part = [0., 0., 0., math.cos(angle / 2)]
        part[index] = math.sin(angle / 2)
        q = quaternion_product(q, part)
    return q


def hand_feedback(frames, values):
    prefix = 'r136d7b_link_'
    errors, stamps = {}, []
    for name, value in values.items():
        child, parent, origin, axis = HAND_JOINTS[name]
        frame = frames.get(prefix + child)
        if frame is None or frame['parent'] != prefix + parent:
            return {'complete': False, 'reason': f'Missing current hand transform: {name}'}
        actual = frame['xyzw']
        norm = math.sqrt(sum(v*v for v in actual))
        if len(actual) != 4 or not math.isfinite(norm) or norm < 1e-9:
            return {'complete': False, 'reason': f'Invalid hand rotation: {name}'}
        expected = target_quaternion(origin, axis, value)
        dot = abs(sum(a*b for a, b in zip(actual, expected))) / norm
        errors[name] = 2 * math.acos(min(1., dot))
        stamps.append(frame['stamp'][0] + frame['stamp'][1] * 1e-9)
    return {'complete': True, 'joint_rotation_error_rad': errors,
            'maximum_error_rad': max(errors.values()), 'stamp_min_sec': min(stamps),
            'stamp_max_sec': max(stamps)}


def targets(pose, thumb_yaw=None):
    # Native radians observed on all 11 SDK /cmd topics in this scene, SDK 3.0.0.
    if pose == 'open':
        mcp, dip, yaw, pitch, ip = -.082, -.0837, .34705882352941186, -.0922, -.0872
    elif pose in ('close', 'close_without_pinky', 'preshape_without_pinky'):
        mcp, dip, yaw, pitch, ip = 1.7, 1.53, .511764705882353, .4772, .8472
    else:
        raise ValueError('Expected open, close, close_without_pinky or preshape_without_pinky')
    result = {f'lh_{finger}_{joint}': value for finger in ('index', 'middle', 'ring', 'pinky')
              for joint, value in (('mcp_pitch', mcp), ('dip', dip))}
    result.update(lh_thumb_cmc_yaw=yaw, lh_thumb_cmc_pitch=pitch, lh_thumb_ip=ip)
    if pose == 'close_without_pinky':
        result.update(lh_pinky_mcp_pitch=-.082, lh_pinky_dip=-.0837)
    if pose == 'preshape_without_pinky':
        opened = targets('open')
        result = {name: (opened[name] + value) / 2
                  if name.startswith(('lh_index_', 'lh_middle_', 'lh_ring_')) else opened[name]
                  for name, value in result.items()}
    if thumb_yaw is not None:
        if pose not in ('close', 'close_without_pinky') or not math.isfinite(thumb_yaw) or not 0 <= thumb_yaw <= 1.3:
            raise ValueError('Thumb yaw override requires closure and a finite angle in [0, 1.3] rad')
        result['lh_thumb_cmc_yaw'] = thumb_yaw
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', choices=('open', 'close', 'close_without_pinky', 'preshape_without_pinky'), required=True)
    parser.add_argument('--thumb-yaw', type=float, help='Closure-only thumb CMC yaw target in native radians')
    parser.add_argument('--wait-open', action='store_true',
                        help='Require fresh robot TF to verify all open joints before returning')
    args = parser.parse_args()
    if args.wait_open and args.pose != 'open':
        parser.error('--wait-open requires --pose open; a blocked grasp need not reach its closure targets')
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
        frames = {}
        if args.wait_open:
            from rclpy.qos import qos_profile_sensor_data
            from tf2_msgs.msg import TFMessage
            allowed = {'r136d7b_link_' + item[0] for item in HAND_JOINTS.values()}
            def receive(msg):
                for item in msg.transforms:
                    if item.child_frame_id in allowed:
                        q = item.transform.rotation
                        frames[item.child_frame_id] = {'parent': item.header.frame_id,
                            'xyzw': [q.x, q.y, q.z, q.w],
                            'stamp': [item.header.stamp.sec, item.header.stamp.nanosec]}
            subscription = node.create_subscription(TFMessage, f'/{namespace}/tf', receive,
                                                     qos_profile_sensor_data)
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
        command_stamp = max((f['stamp'][0] + f['stamp'][1] * 1e-9 for f in frames.values()), default=-1.)
        started = time.monotonic()
        deadline = started + (12 if args.wait_open else 1)
        stable_stamp = None
        feedback = None
        arrived = False
        while time.monotonic() < deadline:
            # Preserve the original one-second command burst. Waiting for TF
            # must not keep resubmitting goals throughout the observation.
            if time.monotonic() - started < 1:
                for name, pub in publishers.items():
                    pub.publish(Float64(data=values[name]))
            rclpy.spin_once(node, timeout_sec=.05)
            if args.wait_open:
                feedback = hand_feedback(frames, values)
                if (feedback['complete'] and feedback['maximum_error_rad'] < .03 and
                        feedback['stamp_min_sec'] > command_stamp and
                        feedback['stamp_max_sec'] - feedback['stamp_min_sec'] < .1):
                    if stable_stamp is None:
                        stable_stamp = feedback['stamp_min_sec']
                    if feedback['stamp_min_sec'] - stable_stamp >= .1:
                        arrived = True
                        break
                else:
                    stable_stamp = None
        print(json.dumps({'pose': args.pose, 'native_targets_rad': values,
                          'command_published': True, 'hand_arrival_verified': arrived,
                          'wall_seconds': time.monotonic() - started,
                          'feedback_start_stamp_sec': command_stamp,
                          'stable_feedback_start_stamp_sec': stable_stamp, 'feedback': feedback}), flush=True)
        if args.wait_open and not arrived:
            raise TimeoutError('Open-hand TF did not reach all targets; no arm continuation permitted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
