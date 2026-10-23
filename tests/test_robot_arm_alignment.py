"""Exact timestamp robot-only snapshots for local mounting calibration."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rabokit_robot_frames import aligned_arm_sample

BASE = 'r6ef2dc_link_57d50b5359'
LEFT = 'r412d23_link_c7c36cf32f'
RIGHT = 'rbd03eb_link_13df66db66'
LH = 'r136d7b_link_2848c4d4a8'
RH = 'rcd72e2_link_88a3e0c332'


def frame(parent, stamp):
    return dict(parent=parent, xyz=[0., 0., 0.], xyzw=[0., 0., 0., 1.], stamp=list(stamp))


def inputs():
    static = {BASE: frame('world', (0, 0)), LEFT: frame(BASE, (0, 0)), RIGHT: frame(BASE, (0, 0))}
    dynamic = {(10, 20): {LH: frame(LEFT, (10, 20)), RH: frame(RIGHT, (10, 20))}}
    joints = {side: {(10, 20): dict(position={f'{side}_J{i}': .01*i for i in range(1, 8)})}
              for side in ('arm_left', 'arm_right')}
    return dynamic, joints, static


class ArmAlignmentTests(unittest.TestCase):
    def test_complete_common_sample_survives_a_newer_partial_batch_without_object_frames(self):
        dynamic, joints, static = inputs()
        dynamic[(10, 40)] = {LH: frame(LEFT, (10, 40))}
        dynamic[(10, 20)]['green_prop'] = frame('world', (10, 20))
        for side in joints:
            joints[side][(10, 40)] = copy.deepcopy(joints[side][(10, 20)])
        sample = aligned_arm_sample(dynamic, joints, static)
        self.assertEqual(sample['stamp'], [10, 20])
        self.assertEqual(set(sample['frames']), {BASE, LEFT, RIGHT, LH, RH})
        self.assertEqual(sample['arms']['arm_right']['position']['arm_right_J7'], .07)

    def test_mismatched_stamps_missing_parent_or_invalid_arm_position_cannot_calibrate(self):
        dynamic, joints, static = inputs()
        joints['arm_right'][(10, 40)] = joints['arm_right'].pop((10, 20))
        self.assertIsNone(aligned_arm_sample(dynamic, joints, static))
        dynamic, joints, static = inputs()
        dynamic[(10, 20)][RH]['parent'] = 'rbd03eb_link_missing'
        self.assertIsNone(aligned_arm_sample(dynamic, joints, static))
        dynamic, joints, static = inputs()
        joints['arm_right'][(10, 20)]['position']['arm_right_J3'] = float('nan')
        self.assertIsNone(aligned_arm_sample(dynamic, joints, static))


if __name__ == '__main__':
    unittest.main()
