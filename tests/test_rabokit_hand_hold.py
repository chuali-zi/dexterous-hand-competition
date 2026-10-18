"""The hand-only diagnostic preserves the arm pose and reopens after failed holding."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
            left_hand=SimpleNamespace(set_force=Mock(return_value=True)))

    def test_noninitial_arm_is_rejected_before_hand_or_recorder_commands(self):
        robot = self.robot([.02] + [0.] * 6)
        with patch('rabokit_hand_hold.hand_command') as hand, patch('rabokit_hand_hold.subprocess.Popen') as recorder:
            with self.assertRaises(ValueError):
                run_motion(robot, Path('unused'))
            hand.assert_not_called()
            recorder.assert_not_called()
        robot.left_arm.enable.assert_not_called()
        robot.left_hand.set_force.assert_not_called()

    def test_failed_preshape_still_finishes_recording_and_verifies_reopening(self):
        frames = {}
        for name, angle in targets('pinch_preshape_without_pinky').items():
            child, parent, origin, axis = HAND_JOINTS[name]
            frames['r136d7b_link_'+child] = {'parent': 'r136d7b_link_'+parent,
                'xyzw': target_quaternion(origin, axis, angle), 'stamp': [3, 0]}
        runs = Path(__file__).resolve().parents[1] / 'runs'
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs) as temporary:
            output = Path(temporary)
            (output/'trace.json').write_text(json.dumps({'frames': frames}))
            recorder = SimpleNamespace(stdout=io.StringIO('HAND_TRACE_READY\n'), returncode=0,
                communicate=Mock(return_value=('PASS: recorded\n', '')), poll=lambda: 0)
            for failed in (False, True):
                def command(pose, **kwargs):
                    if pose == 'pinch_preshape_without_pinky':
                        if failed:
                            raise subprocess.CalledProcessError(1, 'hand')
                        return {'hand_arrival_verified': True, 'feedback': {'stamp_max_sec': 1.}}
                    return {'hand_arrival_verified': True}
                recorder.stdout.seek(0)
                with patch('rabokit_hand_hold.initialize_empty_hand', return_value={'initialized': True}), \
                        patch('rabokit_hand_hold.subprocess.Popen', return_value=recorder), \
                        patch('rabokit_hand_hold.hand_command', side_effect=command) as hand:
                    if failed:
                        with self.assertRaises(RuntimeError):
                            run_motion(self.robot(), output)
                    else:
                        run_motion(self.robot(), output)
                self.assertEqual(hand.call_args_list[-1].args, ('open',))
                self.assertTrue(hand.call_args_list[-1].kwargs['wait_open'])
                saved = json.loads((output/'hold.json').read_text())
                self.assertTrue(saved['reopened']['hand_arrival_verified'])
                self.assertEqual('error' in saved, failed)
                self.assertFalse(saved['arm_motion_commanded'])
                self.assertEqual(saved['arm_joint_drift_rad'], 0.)
                self.assertTrue(recorder.communicate.called)


class FrameTraceTests(unittest.TestCase):
    def test_trace_keeps_robot_tf_and_yaw_commands_but_excludes_object_frames(self):
        clock, callbacks = [0.], {}
        def subscribe(kind, topic, callback, qos):
            callbacks[topic] = callback
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
            callbacks['/test/tf'](msg)
            if len(callbacks) == 3:
                callbacks['/test/arm_left/lh_thumb_cmc_yaw/cmd'](SimpleNamespace(data=.62))
        node = SimpleNamespace(create_subscription=subscribe, count_publishers=Mock(return_value=1),
                               destroy_node=Mock())
        ros = SimpleNamespace(init=lambda: None, create_node=lambda name: node, spin_once=spin, try_shutdown=Mock())
        modules = {'rclpy': ros, 'rclpy.qos': SimpleNamespace(DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL=1),
            ReliabilityPolicy=SimpleNamespace(RELIABLE=1), QoSProfile=SimpleNamespace, qos_profile_sensor_data=object()),
            'tf2_msgs.msg': SimpleNamespace(TFMessage=object), 'std_msgs.msg': SimpleNamespace(Float64=object)}
        with patch.dict(sys.modules, modules), patch.dict('os.environ', {'ROS2_NAMESPACE': 'test'}), \
                patch('rabokit_robot_frames.time.monotonic', side_effect=lambda: clock[0]), \
                patch('builtins.print') as output:
            result = frame_probe.capture(.8, trace_hand=True)
        self.assertNotIn('prop_hidden', result['frames'])
        self.assertTrue(result['hand_samples'])
        self.assertEqual(len(result['hand_samples'][0]['frames']), 11)
        self.assertTrue(all(row['value'] == .62 for row in result['yaw_commands']))
        self.assertTrue(result['yaw_commands'])
        output.assert_called_once_with('HAND_TRACE_READY', flush=True)
        node.destroy_node.assert_called_once()
        ros.try_shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
