"""RGB transport checks without a ROS installation."""
import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
import zlib

spec = importlib.util.spec_from_file_location(
    'rabokit_camera', Path(__file__).resolve().parents[1] / 'scripts/rabokit_camera.py')
camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera)


class CameraTests(unittest.TestCase):
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
