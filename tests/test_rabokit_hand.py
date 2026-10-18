"""A narrower closure leaves the neighboring-row finger open."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_hand import targets
from rabokit_calibration_probe import hand_command


class HandTargetsTests(unittest.TestCase):
    def test_closure_can_keep_only_the_pinky_open(self):
        opened, closed = targets('open'), targets('close')
        narrowed = targets('close_without_pinky')
        self.assertEqual(len(narrowed), 11)
        for name, value in narrowed.items():
            self.assertEqual(value, (opened if name.startswith('lh_pinky_') else closed)[name])

    def test_thumb_opposition_only_changes_the_requested_yaw(self):
        original = targets('close_without_pinky')
        adjusted = targets('close_without_pinky', thumb_yaw=.62)
        self.assertEqual(adjusted.pop('lh_thumb_cmc_yaw'), .62)
        original.pop('lh_thumb_cmc_yaw')
        self.assertEqual(adjusted, original)
        for pose, value in [('open', .62), ('close', float('nan')), ('close', 1.31)]:
            with self.subTest(pose=pose, value=value), self.assertRaises(ValueError):
                targets(pose, value)

    def test_thumb_override_reaches_the_native_command_but_opening_stays_default(self):
        with patch('rabokit_calibration_probe.subprocess.run',
                   return_value=SimpleNamespace(stdout='', stderr='', check_returncode=lambda: None)) as run:
            hand_command('close_without_pinky', thumb_yaw=.62)
            self.assertEqual(run.call_args.args[0][-4:],
                             ['--pose', 'close_without_pinky', '--thumb-yaw', '0.62'])
            hand_command('open')
            self.assertNotIn('--thumb-yaw', run.call_args.args[0])
