"""Contact-force sign and feasibility, using analytic geometry without O6 assets."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_thumb_self_contact import contact_balance_rays, fit_contact_balance

try:
    import mujoco
except ImportError:
    mujoco = None


class ContactBalanceTests(unittest.TestCase):
    def test_a_friction_cone_fit_cannot_explain_an_unsupported_joint(self):
        columns = np.array([[1., 1.], [-1., 1.], [0., 0.]])
        fit = fit_contact_balance(columns, np.array([2., 0., 3.]))
        self.assertAlmostEqual(fit['fit_error_norm'], 3.)
        self.assertTrue(all(w >= 0 for w in fit['nonnegative_ray_weights']))
        self.assertEqual(fit['column_rank'], 2)
        self.assertAlmostEqual(fit_contact_balance(np.zeros((3, 0)), np.array([1., 2., 2.]))['fit_error_norm'], 3.)

    @unittest.skipIf(mujoco is None, 'Optional MuJoCo runtime not installed')
    def test_balance_ray_sign_opposes_the_solver_contact_torque(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 0" cone="pyramidal"/>
          <worldbody>
            <geom type="plane" size="1 1 .01"/>
            <body pos="0 0 .009"><joint type="slide" axis="0 0 1"/>
              <geom type="sphere" size=".01" mass=".1"/>
            </body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 1)
        columns = contact_balance_rays(model, data, data.contact[0], [0])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, 0, force)
        # A sphere below the plane surface receives upward contact torque.
        self.assertGreater(float(data.qfrc_constraint[0]), 0.)
        np.testing.assert_allclose(columns @ np.full(4, force[0]/4), -data.qfrc_constraint, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
