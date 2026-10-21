"""RGB transport checks without a ROS installation."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location(
    'rabokit_camera', Path(__file__).resolve().parents[1] / 'scripts/rabokit_camera.py')
camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera)


class CameraTests(unittest.TestCase):
    def ramp_trace(self):
        callbacks = {}
        node = SimpleNamespace(create_subscription=lambda kind, topic, callback, qos:
                               callbacks.setdefault(topic, callback))
        modules = {'rclpy.qos': SimpleNamespace(QoSProfile=lambda **kw: kw,
                                               ReliabilityPolicy=SimpleNamespace(RELIABLE=1)),
                   'sensor_msgs.msg': SimpleNamespace(CameraInfo=object, Image=object)}
        with patch.dict(sys.modules, modules):
            trace = camera.RampedRgbTrace(node, 'test')
        self.assertEqual(len(callbacks), 4)
        return trace

    def test_ramp_rgb_saves_only_matched_stereo_in_the_command_window(self):
        trace = self.ramp_trace()
        for stamp in ((0,600000000), (0,800000000), (1,0), (1,200000000),
                      (1,400000000), (1,600000000)):
            for name in trace.names:
                def header(frame):
                    return SimpleNamespace(stamp=SimpleNamespace(sec=stamp[0], nanosec=stamp[1]), frame_id=frame)
                image = SimpleNamespace(header=header(name), width=1, height=1, encoding='rgb8',
                                        step=3, data=bytes((0,255,0)))
                info = SimpleNamespace(header=header('wrong' if stamp==(1,200000000) and name=='base_cam2' else name),
                    width=1, height=1, k=[100.,0.,.5,0.,100.,.5,0.,0.,1.], d=[], r=[], p=[], distortion_model='')
                trace.receive(name, 'image', image)
                if not (stamp==(1,400000000) and name=='base_cam2'):
                    trace.receive(name, 'info', info)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as folder:
            root = Path(folder)/'rgb'
            report = trace.save(root, 1., 1.5)
            self.assertEqual([f['stamp'] for f in report['frames']], [[0,800000000],[1,0]])
            self.assertEqual(report['rejected_unmatched_stamps'], 2)
            for frame in report['frames']:
                for name in trace.names:
                    saved = root/frame['directory']/name
                    self.assertEqual((saved/'frame.png').read_bytes(), camera.png_bytes(1,1,bytes((0,255,0))))
                    self.assertEqual(json.loads((saved/'camera.json').read_text())['frame_id'], name)

    def test_ramp_rgb_bounds_memory_and_reports_no_frames_without_waiting(self):
        trace = self.ramp_trace()
        for i in (*range(32),0):
            trace.receive('base_cam','image',self.message(i,'base_cam'))
        self.assertEqual(sorted(trace.pending), [(i,0) for i in range(8,32)])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as folder:
            report = trace.save(Path(folder)/'rgb',100.,101.)
        self.assertEqual(report['frames'], [])
        self.assertEqual(report['message_counts']['base_cam'], {'image':33,'info':0})

    def capture_events(self, events, names=('a', 'b'), hand_samples=None, publisher_info=None):
        callbacks, cleanup = {}, []
        clock = [0.0]

        def spin(node, timeout_sec):
            clock[0] += 0.1
            if events:
                topic, message = events.pop(0)
                callbacks[topic](message)

        def subscribe(message, topic, callback, qos):
            callbacks[topic] = callback
            return object()

        node = SimpleNamespace(create_subscription=subscribe,
                               get_publishers_info_by_topic=lambda topic: (publisher_info or {}).get(topic, []),
                               destroy_node=lambda: cleanup.append('destroyed'))
        modules = {'rclpy': SimpleNamespace(init=lambda: None, create_node=lambda name: node,
                                            spin_once=spin, try_shutdown=lambda: cleanup.append('shutdown')),
                   'rclpy.qos': SimpleNamespace(QoSProfile=lambda **kw: kw,
                                               ReliabilityPolicy=SimpleNamespace(RELIABLE=1),
                                               qos_profile_sensor_data=object()),
                   'sensor_msgs': SimpleNamespace(),
                   'sensor_msgs.msg': SimpleNamespace(CameraInfo=object, Image=object, JointState=object)}
        with patch.dict(sys.modules, modules), patch.object(camera.time, 'monotonic', lambda: clock[0]):
            try:
                options = {'hand_topic': '/test/arm_left/joint_states', 'hand_samples': hand_samples} if hand_samples is not None else {}
                return camera.capture_cameras({name: (name + '/image', name + '/info') for name in names}, 1.2, **options)
            finally:
                self.assertEqual(cleanup, ['destroyed', 'shutdown'])

    def message(self, stamp, frame):
        return SimpleNamespace(header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp, nanosec=0), frame_id=frame))

    def test_stereo_rejects_different_timestamps_or_camera_info_frame(self):
        for other_stamp, info_frame in ((2, 'b'), (1, 'wrong-frame')):
            events = [('a/image', self.message(1, 'a')), ('a/info', self.message(1, 'a')),
                      ('b/image', self.message(other_stamp, 'b')),
                      ('b/info', self.message(other_stamp, info_frame))]
            with self.subTest(stamp=other_stamp, frame=info_frame), self.assertRaises(TimeoutError):
                self.capture_events(events)

    def test_timeout_reports_real_publishers_and_qos_without_changing_capture(self):
        endpoint = SimpleNamespace(node_name='camera_service', node_namespace='/test',
                                   qos_profile=SimpleNamespace(reliability=SimpleNamespace(name='BEST_EFFORT')))
        for offered in ([], [endpoint]):
            with self.assertRaises(TimeoutError) as raised:
                self.capture_events([], publisher_info={'a/image': offered})
            publishers = json.loads(str(raised.exception).split('; publishers=')[1])
            self.assertEqual(publishers['a']['image'], [{'node': 'camera_service', 'namespace': '/test',
                              'reliability': 'BEST_EFFORT'}] if offered else [])
            self.assertEqual(publishers['a']['info'], [])
            self.assertEqual(publishers['b'], {'image': [], 'info': []})

    def test_stereo_matches_all_four_messages_and_ignores_late_older_pair(self):
        events = [(name + '/' + kind, self.message(stamp, name))
                  for stamp in (2, 1) for name in ('a', 'b') for kind in ('info', 'image')]
        pairs, counts = self.capture_events(events)
        self.assertEqual([pairs[name][0].header.stamp.sec for name in ('a', 'b')], [2, 2])
        self.assertEqual(counts, {'a': {'image': 2, 'info': 2}, 'b': {'image': 2, 'info': 2}})

    def test_hand_feedback_is_collected_in_the_rgb_window_without_filling_missing_fields(self):
        msg = self.message(1, 'arm')
        msg.name = ['arm_joint', 'lh_thumb_ip', 'lh_index_dip']
        msg.position, msg.velocity, msg.effort = [99., .1, .2], [], [99., .4]
        events = [('/test/arm_left/joint_states', msg)]
        events += [(name + '/' + kind, self.message(2, name))
                   for name in ('a', 'b') for kind in ('info', 'image')]
        later = self.message(2, 'arm')
        later.name, later.position, later.velocity, later.effort = ['lh_thumb_ip'], [.3], [.5], [.6]
        events.append(('/test/arm_left/joint_states', later))
        samples = []
        pairs, _ = self.capture_events(events, hand_samples=samples)
        self.assertEqual(pairs['a'][0].header.stamp.sec, 2)
        self.assertEqual(samples, [
            {'stamp': [1, 0], 'position': {'lh_thumb_ip': .1, 'lh_index_dip': .2},
             'velocity': {}, 'effort': {'lh_thumb_ip': .4}},
            {'stamp': [2, 0], 'position': {'lh_thumb_ip': .3},
             'velocity': {'lh_thumb_ip': .5}, 'effort': {'lh_thumb_ip': .6}}])

    def test_three_views_require_all_six_messages_at_one_timestamp(self):
        events = [(name + '/' + kind, self.message(2, name))
                  for name in ('a', 'b') for kind in ('info', 'image')]
        events += [('c/' + kind, self.message(1, 'c')) for kind in ('info', 'image')]
        with self.assertRaises(TimeoutError):
            self.capture_events(events.copy(), ('a', 'b', 'c'))
        events += [('c/' + kind, self.message(2, 'c')) for kind in ('info', 'image')]
        pairs, _ = self.capture_events(events, ('a', 'b', 'c'))
        self.assertEqual([pairs[name][0].header.stamp.sec for name in ('a', 'b', 'c')], [2, 2, 2])

    def test_padded_bgr_rows_preserve_colors_and_skip_padding(self):
        msg = SimpleNamespace(encoding='bgr8', width=1, height=2, step=4,
                              data=bytes((10, 20, 30, 99, 40, 50, 60, 99)))
        self.assertEqual(camera.rgb_bytes(msg), bytes((30, 20, 10, 60, 50, 40)))

    def test_rejects_depth_truncation_and_short_stride(self):
        for encoding, step, data in [('32FC1', 4, b'1234'), ('rgb8', 3, b'12'),
                                      ('rgb8', 2, b'12')]:
            with self.subTest(encoding=encoding, step=step), self.assertRaises(ValueError):
                camera.rgb_bytes(SimpleNamespace(encoding=encoding, width=1,
                                                 height=1, step=step, data=data))

    def test_png_rows_and_checksums_round_trip(self):
        pixels = bytes((255, 0, 0, 0, 255, 0))
        encoded = camera.png_bytes(1, 2, pixels)
        self.assertEqual(encoded[:8], b'\x89PNG\r\n\x1a\n')
        offset = 8
        chunks = {}
        while offset < len(encoded):
            length, = struct.unpack('!I', encoded[offset:offset + 4])
            kind = encoded[offset + 4:offset + 8]
            data = encoded[offset + 8:offset + 8 + length]
            crc, = struct.unpack('!I', encoded[offset + 8 + length:offset + 12 + length])
            self.assertEqual(crc, zlib.crc32(kind + data))
            chunks[kind] = data
            offset += 12 + length
        self.assertEqual(zlib.decompress(chunks[b'IDAT']), b'\0\xff\0\0\0\0\xff\0')


if __name__ == '__main__':
    unittest.main()
