"""Coordinate conventions for the optional local hand proxy, without MuJoCo."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rabokit_hand import HAND_JOINTS, target_quaternion
from replay_hand_dynamics import joint_values, wxyz


class MappingTests(unittest.TestCase):
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
