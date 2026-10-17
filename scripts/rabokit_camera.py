#!/usr/bin/env python3
"""Save one timestamp-matched RGB/CameraInfo pair from a Rabokit simulation."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import time
import zlib


def rgb_bytes(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'Expected rgb8/bgr8, received {msg.encoding}')
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * 3:
        raise ValueError('Invalid RGB image dimensions or row stride')
    data = bytes(msg.data)
    if len(data) != msg.step * msg.height:
        raise ValueError('Image data length does not match its row stride')
    pixels = bytearray().join(data[y * msg.step:y * msg.step + msg.width * 3]
                              for y in range(msg.height))
    if msg.encoding == 'bgr8':
        pixels[0::3], pixels[2::3] = pixels[2::3], pixels[0::3]
    return bytes(pixels)


def png_bytes(width, height, pixels):
    if width <= 0 or height <= 0 or len(pixels) != width * height * 3:
        raise ValueError('Invalid packed RGB image')

    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data +
                struct.pack('!I', zlib.crc32(kind + data)))

    rows = b''.join(b'\0' + pixels[y * width * 3:(y + 1) * width * 3]
                    for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def capture(image_topic, info_topic, seconds):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node('competition_camera_capture')
    pending = {'image': {}, 'info': {}}
    counts = {'image': 0, 'info': 0}
    matched = None
    subscriptions = []
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

    def receive(kind, msg):
        nonlocal matched
        counts[kind] += 1
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id)
        pending[kind][key] = msg
        if key in pending['image'] and key in pending['info']:
            matched = (pending['image'][key], pending['info'][key])
        while len(pending[kind]) > 8:
            del pending[kind][next(iter(pending[kind]))]

    try:
        for kind, topic, message in (('image', image_topic, Image),
                                     ('info', info_topic, CameraInfo)):
            subscriptions.append(node.create_subscription(
                message, topic, lambda msg, kind=kind: receive(kind, msg),
                qos))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if matched is None:
            raise TimeoutError(f'No matching RGB/CameraInfo pair in {seconds}s: {counts}')
        return *matched, counts
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', required=True, help='Relative RGB topic from the scene interface list')
    parser.add_argument('--output', required=True, type=Path, help='New capture directory')
    parser.add_argument('--seconds', type=float, default=6)
    args = parser.parse_args()
    namespace = os.getenv('ROS2_NAMESPACE', '').strip('/')
    if not namespace:
        parser.error('Run in the Rabokit cloud terminal with the simulation started')
    parts = args.topic.split('/')
    if args.topic.startswith('/') or any(not p or p in ('.', '..') for p in parts):
        parser.error('--topic must be relative to the current simulation namespace')
    if not args.topic.endswith('/image_raw'):
        parser.error('--topic must name an RGB image_raw topic')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error('--seconds must be in (0, 60]')
    if args.output.exists():
        parser.error('--output must be a new directory')
    image_topic = f'/{namespace}/{args.topic}'
    info_topic = image_topic.rsplit('/', 1)[0] + '/camera_info'
    image, info, counts = capture(image_topic, info_topic, args.seconds)
    pixels = rgb_bytes(image)
    if (not image.header.frame_id or (image.width, image.height) != (info.width, info.height)
            or len(info.k) != 9 or not all(math.isfinite(v) for v in info.k)
            or info.k[0] <= 0 or info.k[4] <= 0):
        raise ValueError('Missing frame or inconsistent/uncalibrated CameraInfo')
    report = {'image_topic': image_topic, 'camera_info_topic': info_topic,
              'frame_id': image.header.frame_id,
              'stamp': {'sec': image.header.stamp.sec, 'nanosec': image.header.stamp.nanosec},
              'width': image.width, 'height': image.height, 'encoding': image.encoding,
              'step': image.step, 'messages': counts, 'observation_wall_seconds': args.seconds,
              'qos_reliability': 'reliable',
              'k': list(info.k), 'd': list(info.d), 'r': list(info.r), 'p': list(info.p),
              'distortion_model': info.distortion_model,
              'extrinsics_calibrated': False, 'official_score': None}
    args.output.mkdir(parents=True)
    (args.output / 'frame.png').write_bytes(png_bytes(image.width, image.height, pixels))
    (args.output / 'frame.ppm').write_bytes(
        f'P6\n{image.width} {image.height}\n255\n'.encode() + pixels)
    (args.output / 'camera.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    print(f'PASS: RGB and matching CameraInfo saved to {args.output}')


if __name__ == '__main__':
    main()
