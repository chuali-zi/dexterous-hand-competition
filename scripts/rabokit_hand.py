#!/usr/bin/env python3
"""Publish SDK-equivalent left-hand targets; requires simulation hand force setup."""
import argparse
import json
import math
import os
from pathlib import Path
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


def targets(pose, thumb_yaw=None, pad_grasp=False):
    # Native radians observed on all 11 SDK /cmd topics in this scene, SDK 3.0.0.
    if pad_grasp:
        if pose == 'open':
            return targets('open', thumb_yaw)
        high = pose in ('pinch_preshape_without_pinky', 'pinch_high_thumb_ready_without_pinky')
        ready = pose in ('pinch_finger_ready_without_pinky', 'pinch_contact_ready_without_pinky')
        opposed = pose in ('pinch_high_thumb_opposed_without_pinky', 'pinch_thumb_opposed_without_pinky')
        if not (high or ready or opposed or pose == 'pinch_without_pinky'):
            raise ValueError('Pad grasp requires an open or limited-pinch pose')
        result = targets('open')
        # Public-mesh pad fit: MCPs stay open; DIP motion closes the long fingers.
        # Trial 18 RGB edges rotated before final closure while the ring stayed
        # unloaded. Bring both long fingers to the side before opposing the thumb,
        # then add a small final squeeze. The high opening stays unchanged.
        dip_offset = -.20*1.6137 if high else .06 if ready or opposed else .11
        result.update(lh_middle_dip=.9662820796229564+dip_offset,
                      lh_ring_dip=.9087363921157451+dip_offset,
                      lh_thumb_cmc_pitch=.3088039736573618,
                      lh_thumb_ip=.8472, lh_thumb_cmc_yaw=.9652294980590825)
        if high or ready:
            result['lh_thumb_cmc_pitch'] -= .20*.5694
            result['lh_thumb_cmc_yaw'] = .60
        if thumb_yaw is not None:
            targets('close', thumb_yaw)  # Apply the existing native yaw validation.
            result['lh_thumb_cmc_yaw'] = thumb_yaw
        return result
    if pose in ('pinch_preshape_without_pinky', 'pinch_high_thumb_ready_without_pinky',
                'pinch_finger_ready_without_pinky', 'pinch_contact_ready_without_pinky',
                'pinch_high_thumb_opposed_without_pinky', 'pinch_thumb_opposed_without_pinky',
                'pinch_thumb_ready_without_pinky', 'pinch_without_pinky'):
        opened = targets('open')
        opposed = pose in ('pinch_high_thumb_opposed_without_pinky', 'pinch_thumb_opposed_without_pinky')
        default_yaw = 1.15 if opposed or pose == 'pinch_without_pinky' else .62
        closed = targets('close_without_pinky', default_yaw if thumb_yaw is None else thumb_yaw)
        proximal_fraction = (.74 if pose == 'pinch_without_pinky' else
                             .66 if opposed or pose in ('pinch_finger_ready_without_pinky', 'pinch_contact_ready_without_pinky') else .5)
        # Opposite pitch changes preserve the distal orientation from preshape.
        thumb_fraction = (0. if pose == 'pinch_preshape_without_pinky' else
                          .70 if opposed or pose == 'pinch_without_pinky' else .45)
        result = dict(opened)
        # Retain the original three-finger shape for the existing hold diagnostic.
        fingers = (('lh_index_', 'lh_middle_', 'lh_ring_') if pose == 'pinch_thumb_ready_without_pinky'
                   else ('lh_middle_', 'lh_ring_'))
        for name in result:
            if name.startswith(fingers):
                # The longer middle finger needs more MCP bend to align its pad
                # with the ring finger. Compensate DIP to retain pad orientation.
                fraction = proximal_fraction + (.07 if name.startswith('lh_middle_') and
                            pose != 'pinch_thumb_ready_without_pinky' else 0.)
                finger_fraction = (.5 - (fraction-.5) * 1.782 / 1.6137
                                   if name.endswith('_dip') else fraction)
                result[name] += finger_fraction * (closed[name] - opened[name])
            elif name in ('lh_thumb_cmc_pitch', 'lh_thumb_ip'):
                result[name] += thumb_fraction * (closed[name] - opened[name])
        result['lh_thumb_cmc_yaw'] = closed['lh_thumb_cmc_yaw']
        return result
    if pose == 'thumb_ready_without_pinky':
        result = targets('preshape_without_pinky')
        opened, closed = targets('open'), targets('close_without_pinky', thumb_yaw)
        for name in result:
            if name.startswith('lh_thumb_'):
                result[name] = opened[name] + .45 * (closed[name] - opened[name])
        return result
    if pose == 'open':
        mcp, dip, yaw, pitch, ip = -.082, -.0837, .34705882352941186, -.0922, -.0872
    elif pose in ('close', 'close_without_pinky', 'preshape_without_pinky'):
        mcp, dip, yaw, pitch, ip = 1.7, 1.53, .511764705882353, .4772, .8472
    else:
        raise ValueError('Unknown hand pose')
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


def preparation_feedback(frames, pose, thumb_yaw=None, pad_grasp=False):
    feedback = hand_feedback(frames, targets(pose, thumb_yaw, pad_grasp))
    # This low closure step can meet the brick. Joint goals are not a contact
    # sensor; retain all errors without requiring an empty-hand arrival pose.
    feedback['goal_arrival_required'] = pose not in ('thumb_ready_without_pinky', 'pinch_thumb_ready_without_pinky',
                                                   'pinch_contact_ready_without_pinky', 'pinch_thumb_opposed_without_pinky')
    return feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', choices=('open', 'close', 'close_without_pinky', 'preshape_without_pinky',
                                         'thumb_ready_without_pinky', 'pinch_preshape_without_pinky',
                                         'pinch_high_thumb_ready_without_pinky',
                                         'pinch_finger_ready_without_pinky',
                                         'pinch_contact_ready_without_pinky',
                                         'pinch_high_thumb_opposed_without_pinky', 'pinch_thumb_opposed_without_pinky',
                                         'pinch_thumb_ready_without_pinky', 'pinch_without_pinky'), required=True)
    parser.add_argument('--thumb-yaw', type=float, help='Thumb CMC yaw closure endpoint in native radians')
    parser.add_argument('--pad-grasp', action='store_true', help='Use the fitted opposing-pad hand targets')
    parser.add_argument('--ramp-seconds', type=float, default=0.,
                        help='Ramp over [0,3] robot TF simulation seconds, then publish the endpoint for at least 0.5 seconds')
    parser.add_argument('--trace-output', type=Path,
                        help='Record known hand joint feedback during the ramp without extending it')
    waiting = parser.add_mutually_exclusive_group()
    waiting.add_argument('--wait-open', action='store_true',
                        help='Require fresh robot TF to verify all open joints before returning')
    waiting.add_argument('--wait-arrival', action='store_true',
                         help='Wait for fresh TF at an empty-hand high preshape; not a contact pose')
    args = parser.parse_args()
    if not math.isfinite(args.ramp_seconds) or not 0 <= args.ramp_seconds <= 3:
        parser.error('--ramp-seconds must be in [0,3]')
    if args.trace_output and (not args.ramp_seconds or args.trace_output.exists()):
        parser.error('--trace-output requires a ramp and a new output file')
    if args.wait_open and args.pose != 'open':
        parser.error('--wait-open requires --pose open; a blocked grasp need not reach its closure targets')
    if args.wait_arrival and args.pose not in ('preshape_without_pinky', 'pinch_preshape_without_pinky',
                                              'pinch_high_thumb_ready_without_pinky', 'pinch_finger_ready_without_pinky',
                                              'pinch_high_thumb_opposed_without_pinky'):
        parser.error('--wait-arrival requires an empty-hand preshape')
    wait_arrival = args.wait_open or args.wait_arrival
    try:
        values = targets(args.pose, args.thumb_yaw, args.pad_grasp)
    except ValueError as error:
        parser.error(str(error))
    namespace = os.getenv('ROS2_NAMESPACE', '').strip('/')
    if not namespace:
        parser.error('Run only in the Rabokit simulation terminal')
    import rclpy
    from std_msgs.msg import Float64

    rclpy.init()
    node = rclpy.create_node('competition_left_hand_targets')
    trace_started = time.monotonic()
    trace = {'pose': args.pose, 'native_targets_rad': values, 'ramp_seconds': args.ramp_seconds,
             'pad_grasp': args.pad_grasp,
             'ramp_start_rad': None, 'command_start_stamp_sec': None,
             'command_complete': False, 'joint_samples': []}
    try:
        frames = {}
        if wait_arrival or args.ramp_seconds:
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
        if args.trace_output:
            from sensor_msgs.msg import JointState
            def receive_joints(msg):
                indices = {name: i for i, name in enumerate(msg.name)
                           if name in HAND_JOINTS and i < len(msg.position)}
                trace['joint_samples'].append({'wall_seconds': time.monotonic()-trace_started,
                    'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec],
                    'position': {name: msg.position[i] for name, i in indices.items()},
                    'velocity': {name: msg.velocity[i] for name, i in indices.items() if i < len(msg.velocity)},
                    'effort': {name: msg.effort[i] for name, i in indices.items() if i < len(msg.effort)}})
            joint_subscription = node.create_subscription(JointState, f'/{namespace}/arm_left/joint_states',
                                                         receive_joints, qos_profile_sensor_data)
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
        ramp_start = None
        if args.ramp_seconds:
            deadline = time.monotonic() + 3
            while True:
                measured = hand_feedback(frames, values)
                if measured['complete'] and measured['stamp_max_sec'] - measured['stamp_min_sec'] < .1:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('No coherent hand TF for the ramp start; no targets published')
                rclpy.spin_once(node, timeout_sec=.05)
            ramp_start = {}
            for name, (child, _, origin, axis) in HAND_JOINTS.items():
                rest = target_quaternion(origin, axis, 0.)
                relative = quaternion_product([-v for v in rest[:3]] + [rest[3]],
                                               frames['r136d7b_link_' + child]['xyzw'])
                ramp_start[name] = math.remainder(2 * math.atan2(relative[axis], relative[3]), 2 * math.pi)
        command_stamp = max((f['stamp'][0] + f['stamp'][1] * 1e-9 for f in frames.values()), default=-1.)
        trace.update(command_start_stamp_sec=command_stamp, ramp_start_rad=ramp_start)
        started = time.monotonic()
        command_seconds = max(1., args.ramp_seconds + .5)
        deadline = started + (12 if wait_arrival or ramp_start else command_seconds)
        next_publish = 0.
        command_complete = False
        progress = 0.
        stable_stamp = None
        stable_errors = None
        feedback = None
        feedback_samples = []
        arrived = False
        while time.monotonic() < deadline:
            # TF time follows slow or paused physics. Wall time only bounds
            # waiting; it must not accelerate a ramp when simulation slows.
            progress = (min(f['stamp'][0] + f['stamp'][1] * 1e-9 for f in frames.values()) - command_stamp
                        if ramp_start else time.monotonic() - started)
            if progress < command_seconds and (not ramp_start or progress >= next_publish):
                fraction = min(1., max(0., progress) / args.ramp_seconds) if ramp_start else 1.
                for name, pub in publishers.items():
                    value = ramp_start[name] + fraction * (values[name] - ramp_start[name]) if ramp_start else values[name]
                    pub.publish(Float64(data=value))
                next_publish = max(next_publish + .05, progress)
            if ramp_start and progress >= command_seconds:
                command_complete = True
                if not wait_arrival:
                    break
            rclpy.spin_once(node, timeout_sec=.05)
            if wait_arrival:
                feedback = hand_feedback(frames, values)
                elapsed = time.monotonic() - started
                if args.wait_arrival and feedback['complete'] and (not feedback_samples or
                        elapsed - feedback_samples[-1]['wall_seconds'] >= .5):
                    feedback_samples.append({'wall_seconds': elapsed, **feedback})
                if (progress >= args.ramp_seconds and feedback['complete'] and feedback['maximum_error_rad'] < .03 and
                        feedback['stamp_min_sec'] > command_stamp and
                        feedback['stamp_max_sec'] - feedback['stamp_min_sec'] < .1):
                    errors = feedback['joint_rotation_error_rad']
                    # The high thumb can pass through its goal before overshooting.
                    # Require the errors to remain nearly constant, not just small.
                    if stable_stamp is None or ((args.wait_arrival or args.ramp_seconds) and any(
                            abs(errors[name] - stable_errors[name]) > .005 for name in errors)):
                        stable_stamp = feedback['stamp_min_sec']
                        stable_errors = dict(errors)
                    if feedback['stamp_min_sec'] - stable_stamp >= (1. if args.wait_arrival or args.ramp_seconds else .1):
                        arrived = True
                        break
                else:
                    stable_stamp = None
                    stable_errors = None
        print(json.dumps({'pose': args.pose, 'native_targets_rad': values,
                          'ramp_seconds': args.ramp_seconds, 'ramp_start_rad': ramp_start,
                          'ramp_time_basis': 'robot_tf_simulation' if ramp_start else None,
                          'command_progress_seconds': progress,
                          'command_published': True, 'hand_arrival_verified': arrived,
                          'wall_seconds': time.monotonic() - started,
                          'feedback_start_stamp_sec': command_stamp,
                          'stable_feedback_start_stamp_sec': stable_stamp, 'feedback': feedback,
                          'feedback_samples': feedback_samples}), flush=True)
        if wait_arrival and not arrived:
            raise TimeoutError('Hand TF did not reach all targets; no arm continuation permitted')
        if ramp_start and not wait_arrival and not command_complete:
            raise TimeoutError('Simulation did not complete the ramp publication window within 12 wall seconds')
        trace['command_complete'] = True
    except Exception as error:
        trace['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if args.trace_output:
                args.trace_output.parent.mkdir(parents=True, exist_ok=True)
                args.trace_output.write_text(json.dumps(trace, indent=2)+'\n', encoding='utf-8')
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
