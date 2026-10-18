"""The hand-only diagnostic preserves the arm pose and reopens after failed holding."""
import io
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_hand import HAND_JOINTS, target_quaternion, targets
from rabokit_hand_hold import run_motion
import rabokit_robot_frames as frame_probe


class HandHoldTests(unittest.TestCase):
    def robot(self, joints=None):
        return SimpleNamespace(left_arm=SimpleNamespace(
            get_joint_angles=lambda: joints or [0.] * 7,
            get_pose=lambda: [[0., .222, -.6082], [0.] * 3], enable=Mock(return_value=True)),
            left_hand=SimpleNamespace(set_force=Mock(return_value=True), close=Mock(return_value=True)))

    def test_noninitial_arm_is_rejected_before_hand_or_recorder_commands(self):
        robot = self.robot([.02] + [0.] * 6)
        with patch('rabokit_hand_hold.hand_command') as hand, patch('rabokit_hand_hold.subprocess.Popen') as recorder:
            with self.assertRaises(ValueError):
                run_motion(robot, Path('unused'))
            hand.assert_not_called()
            recorder.assert_not_called()
        robot.left_arm.enable.assert_not_called()
        robot.left_hand.set_force.assert_not_called()

    def test_invalid_force_limit_is_rejected_before_actuation(self):
        for value in (0., -1., 1.01, float('nan')):
            robot = self.robot()
            with self.assertRaises(ValueError):
                run_motion(robot, Path('unused'), force_limit=value)
            robot.left_arm.enable.assert_not_called()
            robot.left_hand.set_force.assert_not_called()

        for options in ({'thumb_yaw': float('nan')}, {'thumb_yaw': 1.4},
                        {'thumb_yaw': .51, 'sdk_close': True}, {'ramp_seconds': 3.1},
                        {'ramp_seconds': float('nan')}, {'ramp_seconds': .5, 'sdk_close': True},
                        {'reopen_ramp_seconds': 2.}, {'ramp_seconds': .5, 'reopen_ramp_seconds': 3.1},
                        {'ramp_seconds': .5, 'reopen_ramp_seconds': float('nan')}):
            robot = self.robot()
            with self.assertRaises(ValueError):
                run_motion(robot, Path('unused'), **options)
            robot.left_arm.enable.assert_not_called()

    def test_failed_preshape_still_finishes_recording_and_verifies_reopening(self):
        runs = Path(__file__).resolve().parents[1] / 'runs'
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs) as temporary:
            output = Path(temporary)
            recorder = SimpleNamespace(stdout=io.StringIO('HAND_TRACE_READY\n'), returncode=0,
                communicate=Mock(return_value=('PASS: recorded\n', '')), poll=lambda: 0)
            for failed, prepared, recovery_failed, limit, sdk, yaw, reopen_ramp in ((False, False, False, 1., False, None, None),
                    (True, False, False, 1., False, None, None), (False, True, False, 1., False, None, None),
                    (False, False, True, 1., False, None, None), (False, True, False, .05, False, None, None),
                    (False, False, False, .05, True, None, None), (True, False, False, .05, True, None, None),
                    (False, True, False, 1., False, .511764705882353, None),
                    (False, True, False, 1., False, .511764705882353, 2.)):
                pose = 'close' if sdk else 'pinch_thumb_ready_without_pinky' if prepared else 'pinch_preshape_without_pinky'
                frames = {}
                for name, angle in targets(pose, yaw).items():
                    child, parent, origin, axis = HAND_JOINTS[name]
                    frames['r136d7b_link_'+child] = {'parent': 'r136d7b_link_'+parent,
                        'xyzw': target_quaternion(origin, axis, angle), 'stamp': [3, 0]}
                earlier = {name: {**frame, 'stamp': [1, 800000000]} for name, frame in frames.items()}
                (output/'trace.json').write_text(json.dumps({'frames': frames, 'started_monotonic': time.monotonic(),
                    'hand_samples': [{'wall_seconds': 18., 'frames': earlier}, {'wall_seconds': 19.5, 'frames': frames}]}))
                def command(pose, **kwargs):
                    if pose == 'pinch_preshape_without_pinky':
                        if failed:
                            raise subprocess.CalledProcessError(1, 'hand')
                        return {'hand_arrival_verified': True, 'feedback': {'stamp_max_sec': 1.}}
                    return {'hand_arrival_verified': True}
                recorder.stdout.seek(0)
                robot = self.robot()
                robot.left_hand.close.return_value = not failed
                def initialize_hand(arm, destination, ramp_seconds=0.):
                    if destination.name == 'hand-recovery' and recovery_failed:
                        raise RuntimeError('Reopening not verified')
                    return {'initialized': True, 'opened': {'hand_arrival_verified': True}}
                with patch('rabokit_hand_hold.initialize_empty_hand', side_effect=initialize_hand) as initialize, \
                        patch('rabokit_hand_hold.subprocess.Popen', return_value=recorder), \
                        patch('rabokit_hand_hold.hand_command', side_effect=command) as hand:
                    if failed or recovery_failed:
                        with self.assertRaises(RuntimeError):
                            run_motion(robot, output, prepared_thumb=prepared, force_limit=limit, sdk_close=sdk, thumb_yaw=yaw)
                    else:
                        run_motion(robot, output, prepared_thumb=prepared, force_limit=limit, sdk_close=sdk,
                                   thumb_yaw=yaw, ramp_seconds=2. if reopen_ramp is not None else .5 if yaw is not None else 0.,
                                   reopen_ramp_seconds=reopen_ramp)
                if sdk:
                    robot.left_hand.close.assert_called_once_with(blocking=False)
                    self.assertTrue(all(call.args == ('open',) for call in hand.call_args_list))
                else:
                    motion_call = hand.call_args_list[0]
                    self.assertEqual(motion_call.args[0], pose)
                    self.assertEqual(motion_call.kwargs['thumb_yaw'], yaw)
                    self.assertEqual(motion_call.kwargs['ramp_seconds'],
                                     2. if reopen_ramp is not None else .5 if yaw is not None else 0.)
                    self.assertEqual(motion_call.kwargs.get('wait_arrival', False), not prepared)
                self.assertEqual(initialize.call_count, 1 if yaw is not None or limit != 1. else 2)
                if yaw is not None:
                    self.assertEqual(initialize.call_args.kwargs['ramp_seconds'], .5 if reopen_ramp is None else reopen_ramp)
                if limit == 1. and yaw is None:
                    self.assertEqual(initialize.call_args.args[1], output/'hand-recovery')
                else:
                    self.assertEqual(hand.call_args.args, ('open',))
                    self.assertTrue(hand.call_args.kwargs['wait_open'])
                    if yaw is not None:
                        self.assertEqual(hand.call_args.kwargs['ramp_seconds'], .5 if reopen_ramp is None else reopen_ramp)
                self.assertEqual(robot.left_hand.set_force.call_args.args, ([limit]*6,))
                saved = json.loads((output/'hold.json').read_text())
                self.assertTrue(saved['initialization']['initialized'])
                self.assertEqual(saved['native_targets_rad'], targets(pose, yaw))
                self.assertEqual('reopen_error' in saved, recovery_failed)
                if not recovery_failed:
                    self.assertTrue(saved['reopened']['hand_arrival_verified'])
                self.assertTrue(saved['preshape_hold_verified'])
                self.assertEqual('error' in saved, failed)
                self.assertFalse(saved['arm_motion_commanded'])
                self.assertEqual(saved['arm_joint_drift_rad'], 0.)
                self.assertTrue(recorder.communicate.called)


class FrameTraceTests(unittest.TestCase):
    def test_trace_keeps_robot_tf_and_yaw_commands_but_excludes_object_frames(self):
        clock, callbacks, qos_used, joint_stamps = [0.], {}, {}, []
        def subscribe(kind, topic, callback, qos):
            callbacks[topic] = callback
            qos_used[topic] = qos
        def spin(node, timeout_sec):
            clock[0] += .1
            parents = {'r136d7b_link_'+child: 'r136d7b_link_'+parent
                       for child, parent, _, _ in HAND_JOINTS.values()}
            parents.update({'base_cam': 'world', 'base_cam_optical': 'base_cam',
                'r6ef2dc_link_57d50b5359': 'world', 'r412d23_link_c7c36cf32f': 'world',
                'rbd03eb_link_13df66db66': 'world', 'prop_hidden': 'world'})
            msg = SimpleNamespace(transforms=[SimpleNamespace(child_frame_id=child,
                header=SimpleNamespace(frame_id=parent, stamp=SimpleNamespace(sec=0, nanosec=int(clock[0]*1e9))),
                transform=SimpleNamespace(translation=SimpleNamespace(x=0., y=0., z=0.),
                    rotation=SimpleNamespace(x=0., y=0., z=0., w=1.))) for child, parent in parents.items()])
            callbacks['/test/tf'](msg, {'source_timestamp': 3})
            if '/test/arm_left/lh_thumb_cmc_yaw/cmd' in callbacks:
                for joint, value in (('lh_thumb_cmc_yaw', .62), ('lh_thumb_cmc_pitch', .16403), ('lh_thumb_ip', .33328)):
                    callbacks[f'/test/arm_left/{joint}/cmd'](SimpleNamespace(data=value), {'source_timestamp': 5})
                callbacks['/test/arm_left/joint_states'](SimpleNamespace(
                    name=['lh_thumb_cmc_yaw', 'prop_hidden'], position=[.42, 99.], velocity=[], effort=[],
                    header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=int(clock[0]*1e9)))),
                    {'source_timestamp': 1})
                joint_stamps.append([0, int(clock[0]*1e9)])
        node = SimpleNamespace(create_subscription=subscribe, count_publishers=Mock(return_value=1),
                               get_publishers_info_by_topic=lambda topic: [],
                               destroy_node=Mock())
        ros = SimpleNamespace(init=lambda: None, create_node=lambda name: node, spin_once=spin, try_shutdown=Mock())
        sensor_qos = object()
        modules = {'rclpy': ros, 'rclpy.qos': SimpleNamespace(DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL=1),
            ReliabilityPolicy=SimpleNamespace(RELIABLE=1), QoSProfile=SimpleNamespace, qos_profile_sensor_data=sensor_qos),
            'tf2_msgs.msg': SimpleNamespace(TFMessage=object), 'std_msgs.msg': SimpleNamespace(Float64=object)}
        modules['sensor_msgs.msg'] = SimpleNamespace(JointState=object)
        with patch.dict(sys.modules, modules), patch.dict('os.environ', {'ROS2_NAMESPACE': 'test'}), \
                patch('rabokit_robot_frames.time.monotonic', side_effect=lambda: clock[0]), \
                patch('builtins.print') as output:
            result = frame_probe.capture(.8, trace_hand=True)
        self.assertNotIn('prop_hidden', result['frames'])
        self.assertTrue(result['hand_samples'])
        self.assertEqual(len(result['hand_samples'][0]['frames']), 11)
        self.assertTrue(all(row['value'] == .62 for row in result['yaw_commands']))
        self.assertTrue(result['yaw_commands'])
        self.assertEqual({row['joint'] for row in result['thumb_commands']},
                         {'lh_thumb_cmc_yaw', 'lh_thumb_cmc_pitch', 'lh_thumb_ip'})
        self.assertIs(qos_used['/test/arm_left/lh_thumb_cmc_yaw/cmd'], sensor_qos)
        self.assertEqual(result['hand_joint_states'][0]['position'], {'lh_thumb_cmc_yaw': .42})
        self.assertEqual([row['stamp'] for row in result['hand_joint_states']], joint_stamps)
        self.assertEqual(result['hand_joint_states'][0]['message_info']['source_timestamp'], 1)
        self.assertEqual(next(iter(result['hand_samples'][0]['frames'].values()))['message_info']['source_timestamp'], 3)
        self.assertEqual(result['yaw_commands'][0]['message_info']['source_timestamp'], 5)
        for topic in ('/test/tf', '/test/arm_left/joint_states', '/test/arm_left/lh_thumb_cmc_yaw/cmd'):
            with self.assertRaises(TypeError):
                inspect.signature(callbacks[topic]).bind(object())
        output.assert_called_once_with('HAND_TRACE_READY', flush=True)
        node.destroy_node.assert_called_once()
        ros.try_shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
