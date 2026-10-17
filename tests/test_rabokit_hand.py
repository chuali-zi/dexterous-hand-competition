"""A narrower closure leaves the neighboring-row finger open."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_hand import targets


class HandTargetsTests(unittest.TestCase):
    def test_closure_can_keep_only_the_pinky_open(self):
        opened, closed = targets('open'), targets('close')
        narrowed = targets('close_without_pinky')
        self.assertEqual(len(narrowed), 11)
        for name, value in narrowed.items():
            self.assertEqual(value, (opened if name.startswith('lh_pinky_') else closed)[name])
