"""Reject the observed IK branch jump before any path can be used for motion."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_joint_path import translation_path


class JointPathTests(unittest.TestCase):
    def test_continuous_seed_is_propagated_and_branch_jump_is_rejected(self):
        start = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}

        def forward(joints):
            pose = np.eye(4)
            pose[:3, 3] = joints[:3]
            return pose

        seeds = []

        def inverse(target, seed):
            seeds.append(list(seed))
            return [*target[:3, 3], 0., 0., 0., 0.]

        result = translation_path(start, [.03, 0, 0], inverse, forward)
        self.assertTrue(result['continuous'])
        self.assertAlmostEqual(seeds[1][0], .01)
        jump = translation_path(start, [.03, 0, 0], lambda t, s: [5.78] * 7, forward)
        self.assertFalse(jump['continuous'])
        self.assertEqual(jump['failed_step'], 1)
        self.assertEqual(jump['points'], [])

    def test_unreachable_target_and_interpolation_deviation_are_rejected(self):
        start = {'joints': [0.] * 7, 'pose': [[0.] * 3, [0.] * 3]}

        def forward(joints):
            pose = np.eye(4)
            pose[0, 3] = joints[0] * 2
            return pose

        missing = translation_path(start, [.01, 0, 0], lambda t, s: None, forward)
        self.assertFalse(missing['continuous'])
        wrong = translation_path(start, [.01, 0, 0], lambda t, s: [.01] * 7, forward)
        self.assertFalse(wrong['continuous'])
        self.assertIn('interpolation', wrong['reason'])
