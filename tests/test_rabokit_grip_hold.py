"""Bounded feedback behavior and ownership of the experimental hold publisher."""
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_grip_hold import DIPS, MAX_INCREMENT, STEP, adjusted_targets, check_process, hold_process, run
from rabokit_hand import HAND_JOINTS, targets


class GripHoldTests(unittest.TestCase):
    def test_only_underloaded_dip_advances_and_free_following_cannot_exceed_bound(self):
        initial = targets('pinch_without_pinky', pad_grasp=True)
        positions = {name: value - .006 for name, value in initial.items()}
        positions[DIPS[0]] = initial[DIPS[0]] - .0002
        current = adjusted_targets(initial, initial, positions)
        self.assertAlmostEqual(current[DIPS[0]], initial[DIPS[0]] + STEP)
        self.assertEqual(current[DIPS[1]], initial[DIPS[1]])
        for name in set(initial) - set(DIPS):
            self.assertEqual(current[name], initial[name])
        self.assertEqual(adjusted_targets(initial, initial,
                         {n: v - .006 for n, v in initial.items()}), initial)
        for _ in range(30):
            current = adjusted_targets(initial, current, dict(current))
        for name in DIPS:
            self.assertAlmostEqual(current[name], initial[name] + MAX_INCREMENT)

    def test_incomplete_or_nonfinite_positions_cannot_change_targets(self):
        initial = targets('pinch_without_pinky', pad_grasp=True)
        for bad in ({}, {**initial, DIPS[0]: float('nan')}):
            with self.assertRaises(ValueError):
                adjusted_targets(initial, initial, bad)

    def fake_runtime(self, folder, stall=False, sparse=False):
        initial = targets('pinch_without_pinky', pad_grasp=True)
        clock, handler, callback, published = [0.], {}, [None], []
        last = dict(initial)
        last_emit = [-1.]
        destroyed, shutdown = [], []

        def publisher(name):
            def publish(msg):
                last[name] = msg.data
                published.append((clock[0], name, msg.data))
            return SimpleNamespace(get_subscription_count=lambda: 1, publish=publish)

        def subscribe(kind, topic, receive, qos):
            callback[0] = receive
            return object()

        node = SimpleNamespace(create_subscription=subscribe,
            create_publisher=lambda kind, topic, depth: publisher(topic.split('/')[-2]),
            destroy_node=lambda: destroyed.append(True))

        def spin(node, timeout_sec):
            clock[0] += .02
            if (not stall or clock[0] < .4) and (not sparse or clock[0] - last_emit[0] >= 1.):
                last_emit[0] = clock[0]
                ns = round(clock[0] * 10**9)
                position = dict(last)
                position[DIPS[0]] -= .0002
                position[DIPS[1]] -= .006
                callback[0](SimpleNamespace(name=list(initial),
                    position=[position[n] for n in initial], velocity=[0.] * 11, effort=[0.] * 11,
                    header=SimpleNamespace(stamp=SimpleNamespace(sec=ns // 10**9, nanosec=ns % 10**9))))
            if not stall and clock[0] >= 3.2:
                handler[signal.SIGTERM](None, None)

        def set_signal(key, value):
            old = handler.get(key)
            handler[key] = value
            return old

        modules = {'rclpy': SimpleNamespace(init=lambda: handler.update({signal.SIGTERM: 'rclpy-installed'}),
                       create_node=lambda name: node,
                       spin_once=spin, try_shutdown=lambda: shutdown.append(True)),
                   'rclpy.qos': SimpleNamespace(qos_profile_sensor_data=object()),
                   'sensor_msgs.msg': SimpleNamespace(JointState=object),
                   'std_msgs.msg': SimpleNamespace(Float64=lambda data: SimpleNamespace(data=data))}
        output = Path(folder) / 'hold.json'
        with patch.dict(sys.modules, modules), patch.dict('os.environ', {'ROS2_NAMESPACE': 'test'}), \
                patch('rabokit_grip_hold.time.monotonic', side_effect=lambda: 100 + clock[0]), \
                patch('rabokit_grip_hold.signal.signal', side_effect=set_signal):
            if stall or sparse:
                with self.assertRaisesRegex(TimeoutError, 'recent hand history' if sparse else 'stopped advancing'):
                    run(output)
            else:
                run(output)
        self.assertTrue(destroyed and shutdown)
        return initial, published, json.loads(output.read_text())

    def test_actual_publisher_preserves_nine_targets_and_stops_on_parent_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            initial, publications, trace = self.fake_runtime(folder)
        self.assertEqual(trace['status'], 'stopped')
        self.assertFalse(trace['physical_grasp_verified'])
        self.assertTrue(trace['adjustments'])
        self.assertGreater(trace['decision_count'], 0)
        for stamp, name, value in publications:
            if name == DIPS[0]:
                self.assertLessEqual(value, initial[name] + MAX_INCREMENT)
            else:
                self.assertEqual(value, initial[name])
        self.assertEqual({name for stamp, name, value in publications}, set(HAND_JOINTS))
        self.assertEqual([value for stamp, name, value in publications[:11]], list(initial.values()))

    def test_stalled_feedback_fails_without_advancing_or_republishing_on_old_time(self):
        with tempfile.TemporaryDirectory() as folder:
            initial, publications, trace = self.fake_runtime(folder, stall=True)
        self.assertEqual(trace['status'], 'failed')
        self.assertLess(max(stamp for stamp, name, value in publications), .4)
        self.assertIn('stopped advancing', trace['error'])

    def test_owned_publisher_is_stopped_even_if_the_caller_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            process = SimpleNamespace(pid=42, poll=lambda: None, terminate=Mock(), wait=Mock(return_value=0))
            def launch(*args, **kwargs):
                (root / 'grip-hold.json').write_text(json.dumps({'pid': 42, 'status': 'running'}))
                return process
            with patch('rabokit_grip_hold.subprocess.Popen', side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                    with hold_process(root):
                        raise RuntimeError('capture failed')
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=3)
        with self.assertRaisesRegex(RuntimeError, 'unexpectedly'):
            check_process(SimpleNamespace(poll=lambda: 1, returncode=1))

    def test_advancing_but_sparse_feedback_cannot_silently_skip_the_control_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            initial, publications, trace = self.fake_runtime(folder, sparse=True)
        self.assertEqual(trace['status'], 'failed')
        self.assertEqual(trace['decision_count'], 0)
        self.assertFalse(trace['adjustments'])

    def test_feedback_option_rejects_other_routes_before_observing_or_moving(self):
        from rabokit_grasp import main
        argv = ['rabokit_grasp.py', '--output', 'unused-feedback-output', '--feedback-hold']
        with patch.dict('os.environ', {'ROS2_NAMESPACE': 'test'}), patch('sys.argv', argv), \
                patch('rabokit_grasp.observe') as observe, patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                main()
        observe.assert_not_called()

    def test_abnormal_shutdown_cannot_be_reported_as_normal_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            process = SimpleNamespace(pid=42, poll=lambda: None, terminate=Mock(), wait=Mock(return_value=1))
            def launch(*args, **kwargs):
                (root / 'grip-hold.json').write_text(json.dumps({'pid': 42, 'status': 'running'}))
                return process
            with patch('rabokit_grip_hold.subprocess.Popen', side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, 'exit code 1'):
                    with hold_process(root):
                        pass


if __name__ == '__main__':
    unittest.main()
