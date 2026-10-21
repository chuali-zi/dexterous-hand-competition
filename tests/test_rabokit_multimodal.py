"""Observation replay must not leak future state or invent missing modalities."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_hand import HAND_JOINTS
from rabokit_camera import png_bytes
from rabokit_multimodal import replay


def fixture():
    def sample(nano):
        return {'stamp': [1, nano], 'position': dict.fromkeys(HAND_JOINTS, nano / 10**9),
                'velocity': dict.fromkeys(HAND_JOINTS, 0.), 'effort': dict.fromkeys(HAND_JOINTS, .2)}
    return {'command_complete': True, 'native_targets_rad': dict.fromkeys(HAND_JOINTS, 99.),
            'joint_samples': [sample(nano) for nano in (0, 80000000, 160000000, 240000000, 320000000, 340000000)],
            'rgb_trace': {'cameras': ['base_cam', 'base_cam2'], 'frames': [
                {'stamp': [1, 330000000], 'relative_command_seconds': .33, 'directory': 'frame'}]}}


class MultimodalReplayTests(unittest.TestCase):
    def test_history_is_oldest_first_and_never_selects_future_samples(self):
        report = replay(fixture(), Path('absent-rgb'))
        row = report['rows'][0]
        self.assertEqual([h['sample_stamp_ns'] for h in row['history']],
                         [1000000000, 1080000000, 1160000000, 1240000000, 1320000000])
        self.assertTrue(all(0 <= h['age_seconds'] <= .04 for h in row['history']))
        self.assertAlmostEqual(row['history'][-1]['position'][0], .32)
        self.assertFalse(report['command_error_available'])
        self.assertFalse(row['usable_observation'])
        self.assertTrue(all(not camera['available'] for camera in row['cameras'].values()))

    def test_feedback_gaps_and_missing_fields_remain_unusable(self):
        for kind in ('gap', 'missing', 'nonfinite'):
            trace = fixture()
            if kind == 'gap':
                del trace['joint_samples'][2]
            elif kind == 'missing':
                trace['joint_samples'][2]['effort'].pop(next(iter(HAND_JOINTS)))
            else:
                trace['joint_samples'][2]['velocity'][next(iter(HAND_JOINTS))] = float('nan')
            report = replay(trace, Path('absent-rgb'))
            history = report['rows'][0]['history']
            self.assertFalse(report['rows'][0]['usable_observation'])
            if kind == 'gap':
                self.assertIsNone(history[2])
            else:
                self.assertIsNone(history[2]['effort' if kind == 'missing' else 'velocity'][0])
            json.dumps(report, allow_nan=False)

    def test_published_targets_do_not_use_later_publications_or_future_tf_references(self):
        trace = fixture()
        for sample in trace['joint_samples']:
            sample['wall_seconds'] = sample['stamp'][1] / 10**9 + .02
        trace['issued_commands'] = [
            {'wall_seconds': wall, 'robot_tf_reference_stamp_sec': stamp,
             'targets_rad': dict.fromkeys(HAND_JOINTS, target)}
            for wall, stamp, target in ((.01, 1., .2), (.27, 1.25, .5), (.36, 1.34, 99.))]
        report = replay(trace, Path('absent-rgb'))
        contexts = [h['published_target_context'] for h in report['rows'][0]['history']]
        self.assertEqual([c['targets_rad'][0] for c in contexts], [.2, .2, .2, .2, .5])
        self.assertAlmostEqual(contexts[-1]['target_minus_position_rad'][0], .18)
        self.assertEqual(report['published_target_context_count'], 5)
        self.assertFalse(report['command_error_available'])
        trace['issued_commands'][0]['robot_tf_reference_stamp_sec'] = 1.01
        self.assertIsNone(replay(trace, Path('.'))['rows'][0]['history'][0]['published_target_context'])
        del trace['joint_samples'][-2]['wall_seconds']
        self.assertIsNone(replay(trace, Path('.'))['rows'][0]['history'][-1]['published_target_context'])
        trace['issued_commands'][1]['wall_seconds'] = .01
        with self.assertRaisesRegex(ValueError, 'publication wall times'):
            replay(trace, Path('.'))

    def test_duplicate_stamps_and_invalid_timing_are_rejected(self):
        trace = fixture()
        trace['joint_samples'].append(deepcopy(trace['joint_samples'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            replay(trace, Path('.'))
        for options in ({'history_length': 0}, {'history_period': float('nan')}, {'max_age': 0}):
            with self.assertRaises(ValueError):
                replay(fixture(), Path('.'), **options)

    def test_local_images_need_matching_metadata_and_feedback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'runs') as temporary:
            root = Path(temporary)
            for name in ('base_cam', 'base_cam2'):
                folder = root / 'frame' / name
                folder.mkdir(parents=True)
                (folder / 'frame.png').write_bytes(png_bytes(2, 2, bytes(12)))
                (folder / 'camera.json').write_text(json.dumps({'stamp': {'sec': 1, 'nanosec': 330000000},
                    'encoding': 'rgb8', 'frame_id': name, 'width': 2, 'height': 2, 'k': [1.] * 9}))
            report = replay(fixture(), root)
            self.assertEqual(report['usable_observation_count'], 1)
            metadata_path = root / 'frame/base_cam2/camera.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['stamp']['nanosec'] += 1
            metadata_path.write_text(json.dumps(metadata))
            report = replay(fixture(), root)
            self.assertEqual(report['usable_observation_count'], 0)
            self.assertIn('stamp differs', report['rows'][0]['cameras']['base_cam2']['reason'])
            (root / 'frame/base_cam/frame.png').write_bytes(b'broken PNG')
            self.assertFalse(replay(fixture(), root)['rows'][0]['cameras']['base_cam']['available'])


if __name__ == '__main__':
    unittest.main()
