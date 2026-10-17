"""RGB transport checks without a ROS installation."""
import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

spec = importlib.util.spec_from_file_location(
    'rabokit_camera', Path(__file__).resolve().parents[1] / 'scripts/rabokit_camera.py')
camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera)


class CameraTests(unittest.TestCase):
    def capture_events(self, events):
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
                               destroy_node=lambda: cleanup.append('destroyed'))
        modules = {'rclpy': SimpleNamespace(init=lambda: None, create_node=lambda name: node,
                                            spin_once=spin, try_shutdown=lambda: cleanup.append('shutdown')),
                   'rclpy.qos': SimpleNamespace(QoSProfile=lambda **kw: kw,
                                               ReliabilityPolicy=SimpleNamespace(RELIABLE=1)),
                   'sensor_msgs': SimpleNamespace(),
                   'sensor_msgs.msg': SimpleNamespace(CameraInfo=object, Image=object)}
        with patch.dict(sys.modules, modules), patch.object(camera.time, 'monotonic', lambda: clock[0]):
            try:
                return camera.capture_cameras({'a': ('a/image', 'a/info'),
                                                'b': ('b/image', 'b/info')}, 1.2)
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

    def test_stereo_matches_all_four_messages_and_ignores_late_older_pair(self):
        events = [(name + '/' + kind, self.message(stamp, name))
                  for stamp in (2, 1) for name in ('a', 'b') for kind in ('info', 'image')]
        pairs, counts = self.capture_events(events)
        self.assertEqual([pairs[name][0].header.stamp.sec for name in ('a', 'b')], [2, 2])
        self.assertEqual(counts, {'a': {'image': 2, 'info': 2}, 'b': {'image': 2, 'info': 2}})

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
