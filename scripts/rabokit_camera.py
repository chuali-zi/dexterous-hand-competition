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


def capture_cameras(topics, seconds):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node('competition_camera_capture')
    pending = {}
    counts = {name: {'image': 0, 'info': 0} for name in topics}
    matched = None
    matched_stamp = None
    subscriptions = []
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

    def receive(name, kind, msg):
        nonlocal matched, matched_stamp
        counts[name][kind] += 1
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        group = pending.setdefault(key, {})
        group.setdefault(name, {})[kind] = msg
        if all(name in group and 'image' in group[name] and 'info' in group[name]
               and group[name]['image'].header.frame_id == group[name]['info'].header.frame_id
               for name in topics):
            if matched_stamp is None or key >= matched_stamp:
                matched = {name: (group[name]['image'], group[name]['info']) for name in topics}
                matched_stamp = key
        while len(pending) > 16:
            del pending[next(iter(pending))]

    try:
        for name, (image_topic, info_topic) in topics.items():
            for kind, topic, message in (('image', image_topic, Image), ('info', info_topic, CameraInfo)):
                subscriptions.append(node.create_subscription(
                    message, topic, lambda msg, name=name, kind=kind: receive(name, kind, msg), qos))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if matched is None:
            raise TimeoutError(f'No matching RGB/CameraInfo pair in {seconds}s: {counts}')
        return matched, counts
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def capture(image_topic, info_topic, seconds):
    pairs, counts = capture_cameras({'camera': (image_topic, info_topic)}, seconds)
    return *pairs['camera'], counts['camera']


def save_pair(image, info, counts, seconds, image_topic, info_topic, output):
    pixels = rgb_bytes(image)
    if (not image.header.frame_id or (image.width, image.height) != (info.width, info.height)
            or len(info.k) != 9 or not all(math.isfinite(v) for v in info.k)
            or info.k[0] <= 0 or info.k[4] <= 0):
        raise ValueError('Missing frame or inconsistent/uncalibrated CameraInfo')
    report = {'image_topic': image_topic, 'camera_info_topic': info_topic,
              'frame_id': image.header.frame_id,
              'stamp': {'sec': image.header.stamp.sec, 'nanosec': image.header.stamp.nanosec},
              'width': image.width, 'height': image.height, 'encoding': image.encoding,
              'step': image.step, 'messages': counts, 'observation_wall_seconds': seconds,
              'qos_reliability': 'reliable',
              'k': list(info.k), 'd': list(info.d), 'r': list(info.r), 'p': list(info.p),
              'distortion_model': info.distortion_model,
              'extrinsics_calibrated': False, 'official_score': None}
    output.mkdir(parents=True)
    (output / 'frame.png').write_bytes(png_bytes(image.width, image.height, pixels))
    (output / 'frame.ppm').write_bytes(f'P6\n{image.width} {image.height}\n255\n'.encode() + pixels)
    (output / 'camera.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


class RampedRgbTrace:
    """Retain existing stereo messages on a caller's node; never wait or control it."""

    def __init__(self, node, namespace):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        self.names = ('base_cam', 'base_cam2')
        self.pending = {}
        self.counts = {name: {'image': 0, 'info': 0} for name in self.names}
        self.started = time.monotonic()
        self.topics = {name: (f'/{namespace}/base/{name}/image_raw',
                              f'/{namespace}/base/{name}/camera_info') for name in self.names}
        self.subscriptions = []
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        for name, (image_topic, info_topic) in self.topics.items():
            for kind, topic, message in (('image', image_topic, Image), ('info', info_topic, CameraInfo)):
                self.subscriptions.append(node.create_subscription(message, topic,
                    lambda msg, name=name, kind=kind: self.receive(name, kind, msg), qos))

    def receive(self, name, kind, msg):
        self.counts[name][kind] += 1
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.pending.setdefault(stamp, {}).setdefault(name, {})[kind] = msg
        # At the configured 5 Hz this covers the longest ramp and endpoint hold.
        while len(self.pending) > 24:
            del self.pending[min(self.pending)]

    def save(self, output, command_start, command_end):
        output.mkdir(parents=True)
        observation_wall_seconds = time.monotonic() - self.started
        frames = []
        rejected = 0
        for stamp, group in sorted(self.pending.items()):
            relative = stamp[0] + stamp[1] * 1e-9 - command_start
            if relative < -.25 - 1e-6 or relative > command_end - command_start + 1e-6:
                continue
            if not all(name in group and all(kind in group[name] for kind in ('image', 'info')) and
                       group[name]['image'].header.frame_id == group[name]['info'].header.frame_id
                       for name in self.names):
                rejected += 1
                continue
            folder = f'{stamp[0]:06d}-{stamp[1]:09d}'
            for name in self.names:
                save_pair(group[name]['image'], group[name]['info'], self.counts[name],
                          observation_wall_seconds, *self.topics[name], output / folder / name)
            frames.append({'stamp': list(stamp), 'relative_command_seconds': relative,
                           'directory': folder})
        return {'directory': str(output), 'cameras': list(self.names), 'frames': frames,
                'message_counts': self.counts, 'rejected_unmatched_stamps': rejected,
                'scope': 'Existing RGB stream during hand publication; no extra capture wait, object pose or grasp verdict'}


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
    report = save_pair(image, info, counts, args.seconds, image_topic, info_topic, args.output)
    print(json.dumps(report), flush=True)
    print(f'PASS: RGB and matching CameraInfo saved to {args.output}')


if __name__ == '__main__':
    main()
