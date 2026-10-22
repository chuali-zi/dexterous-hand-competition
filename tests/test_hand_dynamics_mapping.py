"""Coordinate conventions for the optional local hand proxy, without MuJoCo."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rabokit_hand import HAND_JOINTS, target_quaternion
from replay_hand_dynamics import closure_targets, cube_bottom_z, joint_limit_excess, joint_values, parse_delta, wxyz


class MappingTests(unittest.TestCase):
    def test_requested_angle_past_stop_is_reported_without_clamping(self):
        values = dict.fromkeys(HAND_JOINTS, 0.)
        values['lh_thumb_ip'] = 1.3
        self.assertAlmostEqual(joint_limit_excess(values)['lh_thumb_ip'], .12)
        self.assertEqual(len(joint_limit_excess(values)), 1)
        self.assertEqual(values['lh_thumb_ip'], 1.3)
        self.assertEqual(joint_limit_excess(dict.fromkeys(HAND_JOINTS, -.1)), {})

    def test_tilted_cube_can_raise_its_center_without_leaving_table(self):
        angle = np.pi / 6
        center_z = .025 * (np.cos(angle) + np.sin(angle))
        q = Rotation.from_euler('x', angle).as_quat()
        self.assertGreater(center_z - .025, .005)
        self.assertAlmostEqual(cube_bottom_z([0, 0, center_z], q[[3, 0, 1, 2]]), 0., places=12)

    def test_synthetic_correction_preserves_recorded_prefix_and_other_joints(self):
        targets = {n: i * .01 for i, n in enumerate(HAND_JOINTS)}
        name = 'lh_index_dip'
        original = dict(targets)
        for seconds, fraction in [(-1., 0.), (0., 0.), (1., .5), (2., 1.), (3., 1.)]:
            result = closure_targets(targets, {name: .08}, seconds)
            self.assertAlmostEqual(result[name], targets[name] + .08 * fraction)
            self.assertEqual({n: v for n, v in result.items() if n != name},
                             {n: v for n, v in targets.items() if n != name})
        self.assertEqual(targets, original)
        self.assertEqual(parse_delta(name + '=.08'), (name, .08))

    def test_hinge_angles_ignore_quaternion_sign_and_scale(self):
        expected={n:(i-2)*.13 for i,n in enumerate(HAND_JOINTS)}
        for scale in (1.,-2.):
            frames={'r136d7b_link_'+child:{'xyzw':[scale*v for v in target_quaternion(origin,axis,expected[n])]}
                    for n,(child,_,origin,axis) in HAND_JOINTS.items()}
            for n,value in joint_values(frames).items():
                self.assertAlmostEqual(value,expected[n],places=12)

    def test_invalid_or_nonhinge_transform_cannot_be_replayed(self):
        frames={'r136d7b_link_'+child:{'xyzw':target_quaternion(origin,axis,0.)}
                for child,_,origin,axis in HAND_JOINTS.values()}
        key='r136d7b_link_'+HAND_JOINTS['lh_thumb_cmc_yaw'][0]
        for q in ([0.,0.,0.,0.],[0.,0.,0.,float('nan')],[.1,0.,0.,.995]):
            frames[key]['xyzw']=q
            with self.subTest(q=q),self.assertRaises(ValueError):
                joint_values(frames)

    def test_mujoco_quaternion_order_rotates_a_nontrivial_vector_correctly(self):
        matrix=Rotation.from_euler('xyz',[.3,-.7,1.1]).as_matrix()
        q=wxyz(matrix)
        restored=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(restored@np.array([.2,.5,-.1]),matrix@np.array([.2,.5,-.1]),atol=1e-14)


if __name__=='__main__':
    unittest.main()
