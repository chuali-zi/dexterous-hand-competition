"""Stereo matching checks independent of ROS and scene object truth."""
from pathlib import Path
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_stereo import match_candidates, reconstruct
import rabokit_stereo as stereo
from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT
from hand_competition.closed_loop.geometry import Camera


class StereoTests(unittest.TestCase):
    def test_static_capture_feedback_replays_at_the_rgb_stamp(self):
        from rabokit_hand import HAND_JOINTS
        import rabokit_multimodal as multimodal

        def capture(topics, seconds, *, hand_topic, hand_samples):
            self.assertEqual(hand_topic, '/test/arm_left/joint_states')
            hand_samples.extend({'stamp': [1, nano],
                **{field: dict.fromkeys(HAND_JOINTS, nano / 1e9)
                   for field in ('position', 'velocity', 'effort')}}
                for nano in (0, 80000000, 160000000, 240000000, 320000000, 340000000))
            pairs = {}
            for name in topics:
                header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=330000000), frame_id=name)
                image = SimpleNamespace(header=header, width=2, height=2, encoding='rgb8', step=6, data=bytes(12))
                info = SimpleNamespace(header=header, width=2, height=2,
                    k=[1., 0., 1., 0., 1., 1., 0., 0., 1.], d=[], r=[], p=[], distortion_model='')
                pairs[name] = (image, info)
            return pairs, {name: {'image': 1, 'info': 1} for name in topics}

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as tmp:
            folder = Path(tmp)/'camera'
            with patch.dict(os.environ, ROS2_NAMESPACE='test'), patch.object(sys, 'argv', [
                    'rabokit_stereo.py', '--output', str(folder), '--capture-only', '--hand-feedback']), \
                    patch.object(stereo, 'capture_cameras', side_effect=capture):
                stereo.main()
            trace = json.loads((folder/'hand-feedback.json').read_text())
            result = Path(tmp)/'replay.json'
            with patch.object(sys, 'argv', ['rabokit_multimodal.py', str(folder/'hand-feedback.json'),
                                          '--output', str(result)]):
                multimodal.main()
            report = json.loads(result.read_text())
            self.assertEqual(trace['hand_topic'], '/test/arm_left/joint_states')
            self.assertEqual(report['usable_observation_count'], 1)
            row = report['rows'][0]
            self.assertIsNone(row['relative_command_seconds'])
            self.assertAlmostEqual(row['history'][-1]['position'][0], .32)
            self.assertEqual(row['history'][-1]['sample_stamp_ns'], 1320000000)
            self.assertFalse(report['trace_command_complete'])

    def test_sensor_numeric_intrinsics_are_normalized_before_geometry(self):
        class SensorFloat(float):
            pass

        pairs, metadata = {}, {}
        frames = {LEFT_ROOT: {'parent': 'world', 'xyz': [0, .1, 1], 'xyzw': [0, 0, 0, 1]},
                  RIGHT_ROOT: {'parent': 'world', 'xyz': [0, -.1, 1], 'xyzw': [0, 0, 0, 1]}}
        for name, left, origin in (('first', 96, [.5, 0, 2]), ('second', 196, [0, 0, 2])):
            rgb = bytearray(320 * 240 * 3)
            for y in range(96, 106):
                for x in range(left, left + 10):
                    rgb[(y * 320 + x) * 3:(y * 320 + x) * 3 + 3] = b'\xff\0\0'
            image = SimpleNamespace(width=320, height=240, encoding='rgb8', step=960, data=rgb)
            pairs[name] = (image, None)
            metadata[name] = {'frame_id': name, 'width': 320, 'height': 240,
                              'k': [SensorFloat(v) for v in (400, 0, 160, 0, 400, 120, 0, 0, 1)]}
            frames[name] = {'parent': 'world', 'xyz': origin, 'xyzw': [1, 0, 0, 0]}
        report = reconstruct(pairs, metadata, frames)
        self.assertFalse(report['issues'])
        self.assertEqual(len(report['matches']), 1)

    def cameras(self):
        rotation = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
        return [Camera(name, (x, 0, 2), rotation, 400, 400, 160, 120, 320, 240, 'test')
                for name, x in (('first', .5), ('second', 0))]

    def test_rotated_green_surfaces_reconstruct_without_accepting_merged_red(self):
        pairs, metadata = {}, {}
        frames = {LEFT_ROOT: {'parent': 'world', 'xyz': [0, .1, 1], 'xyzw': [0, 0, 0, 1]},
                  RIGHT_ROOT: {'parent': 'world', 'xyz': [0, -.1, 1], 'xyzw': [0, 0, 0, 1]}}
        c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
        for name, center, origin in (('first', 100, [.5, 0, 2]), ('second', 200, [0, 0, 2])):
            rgb = bytearray(320 * 240 * 3)
            for y in range(75, 126):
                for x in range(center - 25, center + 26):
                    u, v = (x - center) * c + (y - 100) * s, -(x - center) * s + (y - 100) * c
                    if abs(u) < 16 and abs(v) < 16:
                        rgb[(y * 320 + x) * 3:(y * 320 + x) * 3 + 3] = b'\0\xff\0'
            for y in range(150, 160):
                for x in range(30, 70):
                    rgb[(y * 320 + x) * 3:(y * 320 + x) * 3 + 3] = b'\xff\0\0'
            pairs[name] = (SimpleNamespace(width=320, height=240, encoding='rgb8', step=960, data=rgb), None)
            metadata[name] = {'frame_id': name, 'width': 320, 'height': 240,
                              'k': [400, 0, 160, 0, 400, 120, 0, 0, 1]}
            frames[name] = {'parent': 'world', 'xyz': origin, 'xyzw': [1, 0, 0, 0]}
        target = reconstruct(pairs, metadata, frames, colors={'green'})
        self.assertEqual(target['issues'], [])
        self.assertEqual([m['color'] for m in target['matches']], ['green'])
        for actual, expected in zip(target['matches'][0]['colored_surface_point_sdk_m'], (.2, .1, -1)):
            self.assertAlmostEqual(actual, expected)
        self.assertFalse(target['physical_grasp_verified'])
        self.assertTrue(reconstruct(pairs, metadata, frames)['issues'])

    def test_known_scalar_projection_recovers_sdk_surface_point(self):
        # World point (.2,.1,0); pinhole equations with z depth 2 m.
        first = {'color': 'green', 'centroid': (160 + 400 * (.2 - .5) / 2, 120 - 400 * .1 / 2)}
        second = {'color': 'green', 'centroid': (160 + 400 * .2 / 2, 120 - 400 * .1 / 2)}
        matches, issues = match_candidates(self.cameras(), [[first], [second]], (.1, 0, 1),
                                            ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self.assertFalse(issues)
        for actual, expected in zip(matches[0]['colored_surface_point_sdk_m'], (.1, .1, -1)):
            self.assertAlmostEqual(actual, expected)
        self.assertGreater(matches[0]['half_pixel_perturbation_radius_m'], 0)

    def test_duplicate_same_color_correspondence_is_not_selected(self):
        first = {'color': 'red', 'centroid': (100, 100)}
        second = {'color': 'red', 'centroid': (200, 100)}
        matches, issues = match_candidates(self.cameras(), [[first], [second, second]], (0, 0, 0),
                                            ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self.assertFalse(matches)
        self.assertTrue(issues)


if __name__ == '__main__':
    unittest.main()
