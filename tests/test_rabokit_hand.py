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
from rabokit_calibration_probe import hand_command, initialize_empty_hand, sample


class HandTargetsTests(unittest.TestCase):
    def test_hand_only_sample_skips_cameras_and_still_rejects_arm_drift(self):
        before = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}
        for drift in (0., .01):
            after = {**before, 'joints': [drift] + [0.] * 6}
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'runs') as folder, \
                    patch('rabokit_calibration_probe.measured', side_effect=[before, after]), \
                    patch('rabokit_calibration_probe.subprocess.run', return_value=SimpleNamespace(
                        stdout='', stderr='', returncode=0)) as run, patch('builtins.print'):
                if drift:
                    with self.assertRaisesRegex(RuntimeError, 'arm drifted'):
                        sample(None, Path(folder), 'baseline', hand_trace=True, hand_only=True)
                else:
                    result = sample(None, Path(folder), 'baseline', hand_trace=True, hand_only=True)
                    self.assertEqual(result['max_joint_drift_rad'], 0.)
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertTrue(command[1].endswith('rabokit_robot_frames.py'))
                self.assertIn('--hand-trace', command)

    def test_ramp_starts_at_measured_angles_reaches_goal_and_requires_complete_feedback(self):
        start, goal = targets('open'), targets('pinch_thumb_ready_without_pinky', .511764705882353)
        start['lh_thumb_cmc_yaw'] += .012
        for missing, ramp_duration, sim_rate in ((False, .5, 1.), (False, 2., 1.),
                (False, .5, .4), (False, .5, 0.), (True, .5, 1.)):
            clock, callbacks, published = [0.], [], {name: [] for name in goal}
            def spin(node, timeout_sec):
                clock[0] += timeout_sec
                if missing:
                    return
                transforms = []
                stamp = 2 + clock[0] * sim_rate
                for name, (child, parent, origin, axis) in HAND_JOINTS.items():
                    q = target_quaternion(origin, axis, start[name])
                    transforms.append(SimpleNamespace(child_frame_id='r136d7b_link_'+child,
                        header=SimpleNamespace(frame_id='r136d7b_link_'+parent,
                            stamp=SimpleNamespace(sec=int(stamp), nanosec=int((stamp % 1)*1e9))),
                        transform=SimpleNamespace(rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]))))
                callbacks[0](SimpleNamespace(transforms=transforms))
            def publisher(kind, topic, qos):
                name = topic.split('/')[-2]
                return SimpleNamespace(get_subscription_count=lambda: 1,
                    publish=lambda msg: published[name].append((clock[0], msg.data)))
            node = SimpleNamespace(create_subscription=lambda kind, topic, callback, qos: callbacks.append(callback),
                destroy_subscription=lambda sub: callbacks.clear(),
                get_topic_names_and_types=lambda: [(f'/test/arm_left/{name}/cmd', ['std_msgs/msg/Float64']) for name in goal],
                create_publisher=publisher, destroy_node=lambda: None)
            def maybe_spin(node, timeout_sec):
                if callbacks:
                    spin(node, timeout_sec)
                else:
                    clock[0] += timeout_sec
            modules = {'rclpy': SimpleNamespace(init=lambda: None, create_node=lambda name: node,
                            spin_once=maybe_spin, shutdown=lambda: None),
                       'rclpy.qos': SimpleNamespace(qos_profile_sensor_data=object()),
                       'std_msgs.msg': SimpleNamespace(Float64=SimpleNamespace),
                       'tf2_msgs.msg': SimpleNamespace(TFMessage=object)}
            with patch.dict(sys.modules, modules), patch.dict('os.environ', {'ROS2_NAMESPACE': 'test'}), \
                    patch.object(sys, 'argv', ['rabokit_hand.py', '--pose', 'pinch_thumb_ready_without_pinky',
                        '--thumb-yaw', '.511764705882353', '--ramp-seconds', str(ramp_duration)]), \
                    patch('rabokit_hand.time.monotonic', side_effect=lambda: clock[0]), patch('builtins.print') as output:
                if missing:
                    with self.assertRaises(TimeoutError):
                        main()
                    self.assertTrue(all(not rows for rows in published.values()))
                    self.assertLess(clock[0], 3.1)
                    continue
                if not sim_rate:
                    with self.assertRaises(TimeoutError):
                        main()
                    self.assertTrue(all(len(rows) == 1 for rows in published.values()))
                    self.assertLess(clock[0], 12.2)
                    continue
                main()
            record = json.loads(output.call_args.args[0])
            for name, rows in published.items():
                self.assertAlmostEqual(rows[0][1], start[name], places=6)
                self.assertAlmostEqual(rows[-1][1], goal[name])
                self.assertTrue(any(min(start[name], goal[name]) < v < max(start[name], goal[name])
                    for _, v in rows) or abs(start[name]-goal[name]) < 1e-6)
                self.assertLess((rows[-1][0] - rows[0][0]) * sim_rate, max(1., ramp_duration + .5) + 1e-8)
                if abs(start[name]-goal[name]) > 1e-6:
                    first_endpoint = next(t for t, v in rows if abs(v-goal[name]) < 1e-8)
                    self.assertGreaterEqual((first_endpoint - rows[0][0]) * sim_rate, ramp_duration - .06)
            self.assertEqual(record['native_targets_rad'], goal)
            self.assertEqual(record['ramp_time_basis'], 'robot_tf_simulation')
            self.assertFalse(record['hand_arrival_verified'])
        with patch('rabokit_calibration_probe.subprocess.run', return_value=SimpleNamespace(stdout='', stderr='',
                check_returncode=lambda: None)) as run:
            hand_command('pinch_thumb_ready_without_pinky', ramp_seconds=.5)
            self.assertEqual(run.call_args.args[0][-2:], ['--ramp-seconds', '0.5'])
            self.assertEqual(run.call_args.kwargs['timeout'], 25)

    def high_wait(self, overshoot=False, ramped_open=False, pose='pinch_preshape_without_pinky', record_ramp=False):
        clock, callbacks, published = [0.], [], []
        values = targets('open' if ramped_open else pose)
        def spin(node, timeout_sec):
            clock[0] += timeout_sec
            transforms = []
            for name, (child, parent, origin, axis) in HAND_JOINTS.items():
                offset = .163 if clock[0] < 3 else -.07334 if overshoot and clock[0] >= 3.4 else 0.
                angle = values[name] - (offset if name == 'lh_thumb_cmc_yaw' else 0.)
                q = target_quaternion(origin, axis, angle)
                transforms.append(SimpleNamespace(child_frame_id='r136d7b_link_'+child,
                    header=SimpleNamespace(frame_id='r136d7b_link_'+parent,
                        stamp=SimpleNamespace(sec=int(clock[0]), nanosec=int((clock[0] % 1)*1e9))),
                    transform=SimpleNamespace(rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]))))
            callbacks[0](SimpleNamespace(transforms=transforms))
            if record_ramp:
                names = [*values, 'unrelated_joint']
                callbacks[1](SimpleNamespace(name=names,
                    header=SimpleNamespace(stamp=SimpleNamespace(sec=int(clock[0]), nanosec=int((clock[0] % 1)*1e9))),
                    position=[values.get(name, 999.) for name in names], velocity=[0.] * len(names),
                    effort=[1. if clock[0] < 3 else 0.] * len(names)))
        publisher = SimpleNamespace(get_subscription_count=lambda: 1,
                                    publish=lambda msg: published.append(clock[0]))
        node = SimpleNamespace(create_subscription=lambda kind, topic, callback, qos: callbacks.append(callback),
            get_topic_names_and_types=lambda: [(f'/test/arm_left/{name}/cmd', ['std_msgs/msg/Float64']) for name in values],
            create_publisher=lambda *args: publisher, destroy_node=lambda: None)
        modules = {'rclpy': SimpleNamespace(init=lambda: None, create_node=lambda name: node,
                        spin_once=spin, shutdown=lambda: None),
                   'rclpy.qos': SimpleNamespace(qos_profile_sensor_data=object()),
                   'std_msgs.msg': SimpleNamespace(Float64=SimpleNamespace),
                   'sensor_msgs.msg': SimpleNamespace(JointState=object),
                   'tf2_msgs.msg': SimpleNamespace(TFMessage=object)}
        arguments = ['rabokit_hand.py', '--pose', 'open', '--wait-open', '--ramp-seconds', '.5'] if ramped_open else \
                    ['rabokit_hand.py', '--pose', pose, '--wait-arrival']
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'runs') as folder:
            trace_path = Path(folder) / 'command.json'
            if record_ramp:
                arguments.extend(('--ramp-seconds', '.5', '--trace-output', str(trace_path)))
            with patch.dict(sys.modules, modules), patch.dict('os.environ', {'ROS2_NAMESPACE': 'test'}), \
                    patch.object(sys, 'argv', arguments), \
                    patch('rabokit_hand.time.monotonic', side_effect=lambda: clock[0]), patch('builtins.print') as output:
                if overshoot:
                    with self.assertRaises(TimeoutError):
                        main()
                else:
                    main()
            trace = json.loads(trace_path.read_text()) if record_ramp else None
        record = json.loads(output.call_args.args[0])
        if record_ramp:
            record['saved_trace'] = trace
        return record, clock[0], published

    def test_command_trace_retains_transient_load_and_partial_feedback_on_timeout(self):
        for overshoot in (False, True):
            record, elapsed, _ = self.high_wait(overshoot=overshoot, record_ramp=True)
            trace = record['saved_trace']
            self.assertEqual(trace['command_complete'], not overshoot)
            if overshoot:
                self.assertIn('error', trace)
            else:
                self.assertNotIn('error', trace)
            self.assertIsNotNone(trace['command_start_stamp_sec'])
            self.assertEqual(set(trace['ramp_start_rad']), set(HAND_JOINTS))
            self.assertTrue(all(set(row['position']) == set(HAND_JOINTS) for row in trace['joint_samples']))
            effort = [row['effort']['lh_thumb_cmc_yaw'] for row in trace['joint_samples']]
            self.assertEqual(set(effort), {0., 1.})
            self.assertEqual(effort[-1], 0.)
            self.assertLess(elapsed, 12.1 if overshoot else 5.)

    def test_high_preshape_waits_for_late_fresh_arrival_without_republishing(self):
        record, elapsed, published = self.high_wait()
        self.assertGreaterEqual(elapsed, 4.)
        self.assertLess(elapsed, 5.)
        self.assertLess(max(published), 1.)
        self.assertTrue(record['hand_arrival_verified'])
        self.assertGreater(record['feedback_samples'][0]['maximum_error_rad'], .16)
        self.assertLess(record['feedback']['maximum_error_rad'], .03)

    def test_high_finger_ready_matches_low_geometry_but_requires_empty_hand_settling(self):
        pose = 'pinch_finger_ready_without_pinky'
        self.assertEqual(targets(pose, .511764705882353),
                         targets('pinch_contact_ready_without_pinky', .511764705882353))
        self.assertFalse(preparation_feedback({}, 'pinch_contact_ready_without_pinky')['goal_arrival_required'])
        self.assertTrue(preparation_feedback({}, pose)['goal_arrival_required'])
        for high_pose in ('pinch_high_thumb_ready_without_pinky', pose):
            for overshoot in (False, True):
                record, elapsed, published = self.high_wait(overshoot=overshoot, pose=high_pose)
                self.assertEqual(record['hand_arrival_verified'], not overshoot)
                self.assertGreaterEqual(elapsed, 12. if overshoot else 4.)
                self.assertLess(max(published), 1.)

    def test_short_goal_crossing_followed_by_measured_scale_overshoot_does_not_verify_settling(self):
        for ramped_open in (False, True):
            record, elapsed, published = self.high_wait(overshoot=True, ramped_open=ramped_open)
            self.assertGreaterEqual(elapsed, 12.)
            self.assertLess(max(published), 1.1 if ramped_open else 1.)
            self.assertFalse(record['hand_arrival_verified'])
            self.assertAlmostEqual(record['feedback']['joint_rotation_error_rad']['lh_thumb_cmc_yaw'], .07334)

    def test_high_wait_is_forwarded_and_contact_wait_is_rejected(self):
        with patch('rabokit_calibration_probe.subprocess.run', return_value=SimpleNamespace(
                stdout='{"hand_arrival_verified": true}\n', stderr='', check_returncode=lambda: None)) as run:
            self.assertTrue(hand_command('pinch_preshape_without_pinky', wait_arrival=True)['hand_arrival_verified'])
            self.assertEqual(run.call_args.args[0][-1], '--wait-arrival')
            self.assertEqual(run.call_args.kwargs['timeout'], 22)
        for pose in ('pinch_thumb_ready_without_pinky', 'pinch_contact_ready_without_pinky'):
            with patch.object(sys, 'argv', ['rabokit_hand.py', '--pose', pose, '--wait-arrival']), \
                    patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_limited_pinch_prepositions_yaw_and_preserves_distal_orientation(self):
        opened, closed = targets('open'), targets('close_without_pinky', .62)
        high = targets('pinch_preshape_without_pinky')
        ready = targets('pinch_contact_ready_without_pinky')
        pinch = targets('pinch_without_pinky')
        for name in opened:
            if name == 'lh_thumb_cmc_yaw':
                self.assertEqual([shape[name] for shape in (high, ready, pinch)], [.62, .62, .85])
            elif name.startswith(('lh_index_', 'lh_pinky_')):
                self.assertEqual([shape[name] for shape in (high, ready, pinch)], [opened[name]] * 3)
            else:
                fractions = ((0., .45, .70) if name.startswith('lh_thumb_') else
                             (.57, .73, .81) if name.startswith('lh_middle_') else (.5, .66, .74))
                if name.endswith('_dip'):
                    fractions = tuple(.5-(f-.5)*1.782/1.6137 for f in fractions)
                for shape, fraction in zip((high, ready, pinch), fractions):
                    self.assertAlmostEqual(shape[name], opened[name] + fraction * (closed[name] - opened[name]))
        self.assertAlmostEqual(pinch['lh_ring_mcp_pitch'], 1.23668)
        self.assertAlmostEqual(pinch['lh_ring_dip'], .29547)
        self.assertAlmostEqual(pinch['lh_middle_mcp_pitch'], 1.36142)
        self.assertAlmostEqual(pinch['lh_middle_dip'], .17073)
        self.assertAlmostEqual(ready['lh_ring_mcp_pitch'], 1.09412)
        self.assertAlmostEqual(ready['lh_ring_dip'], .43803)
        legacy = targets('pinch_thumb_ready_without_pinky')
        self.assertAlmostEqual(legacy['lh_ring_mcp_pitch'], .809)
        self.assertAlmostEqual(legacy['lh_ring_dip'], .72315)
        prepared = targets('pinch_high_thumb_ready_without_pinky')
        for finger in ('index', 'middle', 'ring'):
            self.assertAlmostEqual(legacy[f'lh_{finger}_mcp_pitch'], .809)
            self.assertAlmostEqual(legacy[f'lh_{finger}_dip'], .72315)
        self.assertEqual(prepared['lh_index_mcp_pitch'], opened['lh_index_mcp_pitch'])
        self.assertAlmostEqual(prepared['lh_middle_mcp_pitch'], .93374)
        for finger in ('middle', 'ring'):
            # The two pitch joints share an axis after the fixed finger roll.
            change = sum(pinch[f'lh_{finger}_{joint}'] - high[f'lh_{finger}_{joint}']
                         for joint in ('mcp_pitch', 'dip'))
            self.assertAlmostEqual(change, 0., places=12)
        for pose in ('pinch_preshape_without_pinky', 'pinch_thumb_ready_without_pinky',
                     'pinch_contact_ready_without_pinky', 'pinch_finger_ready_without_pinky', 'pinch_without_pinky'):
            self.assertEqual(targets(pose, .6)['lh_thumb_cmc_yaw'], .6)
            with self.assertRaises(ValueError):
                targets(pose, float('nan'))

    def test_limited_thumb_contact_records_error_but_high_preshape_requires_arrival(self):
        frames = {}
        values = targets('pinch_thumb_ready_without_pinky')
        for name, (child, parent, origin, axis) in HAND_JOINTS.items():
            angle = values[name] + (.1 if name == 'lh_thumb_cmc_pitch' else 0.)
            frames['r136d7b_link_' + child] = {'parent': 'r136d7b_link_' + parent,
                'xyzw': target_quaternion(origin, axis, angle), 'stamp': [10, 0]}
        feedback = preparation_feedback(frames, 'pinch_thumb_ready_without_pinky')
        self.assertTrue(feedback['complete'])
        self.assertFalse(feedback['goal_arrival_required'])
        self.assertGreater(feedback['maximum_error_rad'], .03)
        self.assertTrue(preparation_feedback(frames, 'pinch_preshape_without_pinky')['goal_arrival_required'])

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

    def test_ramped_initialization_opens_directly_and_records_failures(self):
        arm = SimpleNamespace(get_joint_angles=lambda: [0.] * 7,
                              get_pose=lambda: [[0., .222, -.6082], [0.] * 3])
        runs = Path(__file__).resolve().parents[1] / 'runs'
        with tempfile.TemporaryDirectory(dir=runs) as temporary:
            for fail in (False, True):
                output = Path(temporary) / str(fail)
                with patch('rabokit_calibration_probe.hand_command', return_value={'hand_arrival_verified': True},
                        side_effect=RuntimeError('opening failed') if fail else None) as hand, \
                        patch('rabokit_calibration_probe.subprocess.run') as recorder:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'opening failed'):
                            initialize_empty_hand(arm, output, ramp_seconds=2.)
                    else:
                        self.assertTrue(initialize_empty_hand(arm, output, ramp_seconds=2.)['initialized'])
                    hand.assert_called_once_with('open', wait_open=True, ramp_seconds=2.)
                    recorder.assert_not_called()
                saved = json.loads((output/'initialization.json').read_text())
                self.assertEqual(saved['initialized'], not fail)
                self.assertEqual(saved['kind'], 'direct_ramped_open')

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
