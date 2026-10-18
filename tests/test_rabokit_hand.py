"""A narrower closure leaves the neighboring-row finger open."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_hand import HAND_JOINTS, hand_feedback, main, preparation_feedback, target_quaternion, targets
from rabokit_calibration_probe import hand_command, initialize_empty_hand


class HandTargetsTests(unittest.TestCase):
    def test_contact_preparation_records_errors_without_claiming_goal_arrival(self):
        values = targets('thumb_ready_without_pinky', .62)
        frames = {}
        for name, (child, parent, origin, axis) in HAND_JOINTS.items():
            angle = .127688 if name == 'lh_thumb_cmc_pitch' else values[name]
            frames['r136d7b_link_' + child] = {'parent': 'r136d7b_link_' + parent,
                'xyzw': target_quaternion(origin, axis, angle), 'stamp': [446, 720000001]}
        feedback = preparation_feedback(frames, 'thumb_ready_without_pinky', .62)
        self.assertGreater(feedback['maximum_error_rad'], .03)
        self.assertFalse(feedback['goal_arrival_required'])
        self.assertTrue(preparation_feedback(frames, 'preshape_without_pinky')['goal_arrival_required'])
        child, _, origin, axis = HAND_JOINTS['lh_index_mcp_pitch']
        frames['r136d7b_link_' + child]['xyzw'] = target_quaternion(origin, axis, values['lh_index_mcp_pitch'] + .05)
        recorded = preparation_feedback(frames, 'thumb_ready_without_pinky', .62)
        self.assertGreater(recorded['joint_rotation_error_rad']['lh_index_mcp_pitch'], .03)
        self.assertFalse(recorded['goal_arrival_required'])
        frames.pop('r136d7b_link_' + HAND_JOINTS['lh_thumb_cmc_pitch'][0])
        self.assertFalse(preparation_feedback(frames, 'thumb_ready_without_pinky', .62)['complete'])

    def test_preshape_uses_half_finger_targets_and_keeps_thumb_and_pinky_open(self):
        opened = targets('open')
        support = targets('preshape_without_pinky')
        self.assertEqual(set(support), set(opened))
        for name, value in support.items():
            if name.startswith(('lh_thumb_', 'lh_pinky_')):
                self.assertEqual(value, opened[name])
            else:
                self.assertAlmostEqual(value, .809 if name.endswith('mcp_pitch') else .72315)
        with self.assertRaises(ValueError):
            targets('preshape_without_pinky', thumb_yaw=.62)
        with self.assertRaises(ValueError):
            targets('support_without_pinky')

    def test_thumb_preparation_uses_the_closure_endpoint_and_keeps_fingers_half_shaped(self):
        opened, closed = targets('open'), targets('close_without_pinky', .62)
        half, ready = targets('preshape_without_pinky'), targets('thumb_ready_without_pinky', .62)
        for name, value in ready.items():
            if name.startswith('lh_thumb_'):
                self.assertAlmostEqual(value, opened[name] + .45 * (closed[name] - opened[name]))
            else:
                self.assertEqual(value, half[name])
        for invalid in (float('nan'), -.01, 1.31):
            with self.assertRaises(ValueError):
                targets('thumb_ready_without_pinky', invalid)

    def test_empty_initialization_checks_closure_and_always_attempts_verified_reopening(self):
        arm = SimpleNamespace(get_joint_angles=lambda: [0.] * 7,
                              get_pose=lambda: [[0., .222, -.6082], [0.] * 3])
        def capture(command, **kwargs):
            Path(command[-1]).write_text('{"frames": {}}')
        workspace_runs = Path(__file__).resolve().parents[1] / 'runs'
        workspace_runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_runs, prefix='hand-init-test-') as temporary:
            for error in (.001, .1):
                output = Path(temporary) / str(error)
                with patch('rabokit_calibration_probe.hand_command',
                           return_value={'hand_arrival_verified': True}) as hand, \
                        patch('rabokit_calibration_probe.subprocess.run', side_effect=capture) as run, \
                        patch('rabokit_calibration_probe.time.sleep'), \
                        patch('rabokit_hand.hand_feedback',
                              return_value={'complete': True, 'maximum_error_rad': error}):
                    if error < .03:
                        self.assertTrue(initialize_empty_hand(arm, output)['initialized'])
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'closure was not verified'):
                            initialize_empty_hand(arm, output)
                    self.assertEqual(hand.call_args_list[0].args, ('close',))
                    self.assertEqual(hand.call_args_list[0].kwargs, {'thumb_yaw': .62})
                    self.assertEqual(hand.call_args_list[1].args, ('open',))
                    self.assertTrue(hand.call_args_list[1].kwargs['wait_open'])
                    self.assertEqual(Path(run.call_args.args[0][1]).name, 'rabokit_robot_frames.py')
                    saved = json.loads((output / 'initialization.json').read_text())
                    self.assertEqual(saved['initialized'], error < .03)
                    self.assertFalse(saved['arm_motion_commanded'])

    def test_empty_initialization_rejects_an_unverified_arm_position_before_hand_commands(self):
        arm = SimpleNamespace(get_joint_angles=lambda: [.02] + [0.] * 6,
                              get_pose=lambda: [[0., .222, -.6082], [0.] * 3])
        with patch('rabokit_calibration_probe.hand_command') as hand:
            with self.assertRaisesRegex(ValueError, 'tested initial arm pose'):
                initialize_empty_hand(arm, 'unused-initialization-directory')
            hand.assert_not_called()

    def test_open_feedback_timeout_stops_commands_after_the_original_burst(self):
        clock, published = [0.], []
        def spin(node, timeout_sec):
            clock[0] += timeout_sec
        publisher = SimpleNamespace(get_subscription_count=lambda: 1,
                                    publish=lambda msg: published.append(clock[0]))
        node = SimpleNamespace(
            create_subscription=lambda *args: object(),
            get_topic_names_and_types=lambda: [(f'/test/arm_left/{name}/cmd', ['std_msgs/msg/Float64'])
                                               for name in targets('open')],
            create_publisher=lambda *args: publisher, destroy_node=lambda: None)
        ros = SimpleNamespace(init=lambda: None, create_node=lambda name: node,
                              spin_once=spin, shutdown=lambda: None)
        modules = {'rclpy': ros, 'rclpy.qos': SimpleNamespace(qos_profile_sensor_data=object()),
                   'std_msgs.msg': SimpleNamespace(Float64=SimpleNamespace),
                   'tf2_msgs.msg': SimpleNamespace(TFMessage=object)}
        with patch.dict(sys.modules, modules), patch.dict('os.environ', {'ROS2_NAMESPACE': 'test'}), \
                patch.object(sys, 'argv', ['rabokit_hand.py', '--pose', 'open', '--wait-open']), \
                patch('rabokit_hand.time.monotonic', side_effect=lambda: clock[0]), \
                patch('builtins.print') as output:
            with self.assertRaises(TimeoutError):
                main()
        self.assertTrue(published)
        self.assertLess(max(published), 1.)
        self.assertGreaterEqual(clock[0], 12.)
        self.assertIn('"hand_arrival_verified": false', output.call_args.args[0])

    def test_robot_tf_checks_native_open_rotations_including_fixed_mounts(self):
        # Actual empty-hand TF, native-03 opened, not an angle-generated fixture.
        frames = {
            'r136d7b_link_b0fdbe757f': {'parent': 'r136d7b_link_31fc6c92a4',
                'xyzw': [.5048574760632105, -.2914774441867064, .7036483309874745, .40624974423624133],
                'stamp': [97, 379999999]},
            'r136d7b_link_ce25aa231d': {'parent': 'r136d7b_link_2848c4d4a8',
                'xyzw': [.02615501967344601, -.040965252662800725, -.001072715360322655, .998817607124499],
                'stamp': [97, 379999999]}}
        values = {'lh_thumb_cmc_pitch': -.0922, 'lh_index_mcp_pitch': -.082}
        feedback = hand_feedback(frames, values)
        self.assertTrue(feedback['complete'])
        self.assertLess(feedback['maximum_error_rad'], .001)
        closed = hand_feedback(frames, {'lh_thumb_cmc_pitch': .4772})
        self.assertGreater(closed['maximum_error_rad'], .5)
        frames['r136d7b_link_b0fdbe757f']['xyzw'] = [-v for v in
            frames['r136d7b_link_b0fdbe757f']['xyzw']]
        self.assertLess(hand_feedback(frames, values)['maximum_error_rad'], .001)
        frames['r136d7b_link_b0fdbe757f']['parent'] = 'wrong_parent'
        self.assertFalse(hand_feedback(frames, values)['complete'])
        self.assertFalse(hand_feedback({}, targets('open'))['complete'])

    def test_verified_open_is_requested_and_returned_by_the_subprocess_wrapper(self):
        with patch('rabokit_calibration_probe.subprocess.run', return_value=SimpleNamespace(
                stdout='{"hand_arrival_verified": true}\n', stderr='', check_returncode=lambda: None)) as run:
            self.assertTrue(hand_command('open', wait_open=True)['hand_arrival_verified'])
            self.assertEqual(run.call_args.args[0][-1], '--wait-open')
            self.assertGreater(run.call_args.kwargs['timeout'], 18)
            run.return_value.stdout = '{"hand_arrival_verified": false}\n'
            with self.assertRaises(RuntimeError):
                hand_command('open', wait_open=True)

    def test_closure_can_keep_only_the_pinky_open(self):
        opened, closed = targets('open'), targets('close')
        narrowed = targets('close_without_pinky')
        self.assertEqual(len(narrowed), 11)
        for name, value in narrowed.items():
            self.assertEqual(value, (opened if name.startswith('lh_pinky_') else closed)[name])

    def test_thumb_opposition_only_changes_the_requested_yaw(self):
        original = targets('close_without_pinky')
        adjusted = targets('close_without_pinky', thumb_yaw=.62)
        self.assertEqual(adjusted.pop('lh_thumb_cmc_yaw'), .62)
        original.pop('lh_thumb_cmc_yaw')
        self.assertEqual(adjusted, original)
        for pose, value in [('open', .62), ('close', float('nan')), ('close', 1.31)]:
            with self.subTest(pose=pose, value=value), self.assertRaises(ValueError):
                targets(pose, value)

    def test_thumb_override_reaches_the_native_command_but_opening_stays_default(self):
        with patch('rabokit_calibration_probe.subprocess.run',
                   return_value=SimpleNamespace(stdout='', stderr='', check_returncode=lambda: None)) as run:
            hand_command('close_without_pinky', thumb_yaw=.62)
            self.assertEqual(run.call_args.args[0][-4:],
                             ['--pose', 'close_without_pinky', '--thumb-yaw', '0.62'])
            hand_command('open')
            self.assertNotIn('--thumb-yaw', run.call_args.args[0])
