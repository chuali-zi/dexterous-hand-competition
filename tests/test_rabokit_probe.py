"""Three-camera readiness waits once, observes only, and keeps legacy probe behavior."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rabokit_probe as probe
from rabokit_hand import HAND_JOINTS


class ReadinessTests(unittest.TestCase):
    def messages(self):
        header = lambda frame: NS(stamp=NS(sec=5, nanosec=0), frame_id=frame)
        events = []
        for side in ('arm_left', 'arm_right'):
            names = [f'{side}_J{i}' for i in range(1, 8)]
            if side == 'arm_left':
                names += list(HAND_JOINTS)
            events.append((f'/test/{side}/joint_states',
                           NS(header=header(side), name=names, position=[0.] * len(names))))
        for camera in ('base_cam', 'base_cam2', 'base_cam3'):
            events.extend([(f'/test/base/{camera}/image_raw', NS(header=header(camera),
                width=1, height=1, encoding='rgb8', step=3, data=bytes((0, 255, 0)))),
                (f'/test/base/{camera}/camera_info', NS(header=header(camera),
                width=1, height=1, k=[100., 0., .5, 0., 100., .5, 0., 0., 1.]))])
        return events

    def observe(self, events):
        clock, cleanup, callbacks = [0.], [], {}
        def subscribe(kind, topic, callback, qos):
            callbacks[topic] = callback
            self.assertEqual(qos, {'depth':5, 'reliability':1} if '/base/' in topic else 'sensor')
            return object()
        def spin(node, timeout_sec):
            clock[0] += .05
            if events:
                topic, message = events.pop(0)
                if topic:
                    callbacks[topic](message)
        endpoint = NS(node_name='camera', node_namespace='/test',
                      qos_profile=NS(reliability=NS(name='RELIABLE')))
        # No publisher factory or SDK is provided: the tested path can only subscribe.
        node = NS(create_subscription=subscribe, get_publishers_info_by_topic=lambda topic:[endpoint],
                  destroy_node=lambda: cleanup.append('destroy'))
        modules = {'rclpy':NS(init=lambda:None, create_node=lambda name:node,
                   spin_once=spin, try_shutdown=lambda:cleanup.append('shutdown')),
                   'rclpy.qos':NS(QoSProfile=lambda **kw:kw,
                       ReliabilityPolicy=NS(RELIABLE=1), qos_profile_sensor_data='sensor'),
                   'sensor_msgs.msg':NS(Image=object, CameraInfo=object, JointState=object)}
        with patch.dict(sys.modules, modules), patch.dict('os.environ', {'ROS2_NAMESPACE':'test'}), \
                patch.object(probe.time, 'monotonic', lambda:clock[0]):
            try:
                return probe.rgb_ready(2.)
            finally:
                self.assertEqual(cleanup, ['destroy','shutdown'])

    def test_exits_early_only_after_three_valid_pairs_and_both_arms(self):
        result = self.observe(self.messages())
        self.assertTrue(result['ready'])
        self.assertLess(result['wall_seconds'], 1.)
        self.assertEqual(result['matched_stamp'], [5,0])
        self.assertEqual(result['missing_messages'], [])
        self.assertFalse(result['motion_commanded'])
        self.assertEqual(result['required_cameras'], ['base_cam','base_cam2','base_cam3'])

    def test_graph_publishers_cannot_substitute_for_missing_third_camera(self):
        result = self.observe([x for x in self.messages() if '/base_cam3/' not in x[0]])
        self.assertFalse(result['ready'])
        self.assertEqual(result['missing_messages'], ['base_cam3/image','base_cam3/info'])
        self.assertEqual(result['camera_publishers']['base_cam3']['image'][0]['node'], 'camera')

    def test_stamp_mismatch_and_stale_joint_feedback_do_not_pass(self):
        events = self.messages()
        events[-1][1].header.stamp.sec = 6
        self.assertFalse(self.observe(events)['ready'])
        events = self.messages()
        self.assertFalse(self.observe(events[:2] + [(None,None)] * 25 + events[2:])['ready'])

    def test_bad_pixels_calibration_or_incomplete_hand_fail_and_cleanup(self):
        for change in ('pixels','calibration','hand'):
            events = self.messages()
            if change == 'pixels':
                events[2][1].data = b''
            elif change == 'calibration':
                events[-1][1].k[0] = float('nan')
            else:
                events[0][1].position.pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.observe(events)

    def test_cli_readiness_is_exclusive_and_default_probe_remains_read_only(self):
        with patch.dict('os.environ', {'ROS2_NAMESPACE':'test'}), \
                patch.object(probe, 'rgb_ready', return_value={'ready':True}) as ready, \
                patch.object(probe, 'observe', return_value={'joint_states':{
                    side:{'messages':1} for side in ('arm_left','arm_right')}}) as observe, \
                patch.object(probe, 'move_wrist') as motion, patch.object(probe, 'version', return_value='3.0.0'), \
                patch('sys.stdout',new_callable=io.StringIO), patch('sys.stderr',new_callable=io.StringIO):
            with patch.object(sys,'argv',['probe','--rgb-ready','--motion']), self.assertRaises(SystemExit) as raised:
                probe.main()
            self.assertEqual(raised.exception.code,2)
            ready.assert_not_called()
            with patch.object(sys,'argv',['probe','--rgb-ready']):
                probe.main()
            ready.assert_called_once_with()
            observe.assert_not_called()
            with patch.object(sys,'argv',['probe']):
                probe.main()
            observe.assert_called_once_with(4)
            motion.assert_not_called()


if __name__ == '__main__':
    unittest.main()
