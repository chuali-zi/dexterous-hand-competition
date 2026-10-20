"""Analytical examples for the optional offline point-contact calculation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from analyze_grasp_balance import balance_proxy
except ModuleNotFoundError as error:
    if error.name not in ('numpy', 'scipy'):
        raise
    balance_proxy = None


@unittest.skipIf(balance_proxy is None, 'Optional offline numpy/scipy dependencies are unavailable')
class BalanceProxyTests(unittest.TestCase):
    points = [[-.025, 0., 0.], [.025, -.012, 0.], [.025, .012, 0.]]
    opposed = [[1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]]

    def test_balanced_opposed_contacts_span_all_wrenches_with_friction(self):
        result = balance_proxy(self.points, self.opposed, .2)
        self.assertTrue(result['full_dimensional_positive_equilibrium_proxy'])
        self.assertFalse(result['physical_grasp_verified'])

    def test_same_direction_forces_cannot_balance(self):
        result = balance_proxy(self.points, [[1., 0., 0.]] * 3, .4)
        self.assertFalse(result['positive_internal_equilibrium'])

    def test_coincident_contacts_balance_force_but_not_all_moments(self):
        result = balance_proxy([[0., 0., 0.]] * 3, self.opposed, .4)
        self.assertTrue(result['positive_internal_equilibrium'])
        self.assertLess(result['wrench_rank'], 6)
        self.assertFalse(result['full_dimensional_positive_equilibrium_proxy'])

    def test_translation_and_normal_scaling_do_not_change_balance(self):
        base = balance_proxy(self.points, self.opposed, .4)
        moved = [[p[0] + 1., p[1] - 2., p[2] + 3.] for p in self.points]
        scaled = [[v * 3. for v in n] for n in self.opposed]
        result = balance_proxy(moved, scaled, .4)
        self.assertAlmostEqual(base['minimum_normalized_ray_weight'], result['minimum_normalized_ray_weight'], places=9)

    def test_vertical_normals_have_a_valid_tangent_basis(self):
        rotated = [[p[2], p[1], -p[0]] for p in self.points]
        normals = [[n[2], n[1], -n[0]] for n in self.opposed]
        self.assertTrue(balance_proxy(rotated, normals, .4)['full_dimensional_positive_equilibrium_proxy'])

    def test_invalid_physical_inputs_are_rejected(self):
        for mu, normals in ((-.1, self.opposed), (.4, [[0., 0., 0.]] * 3), (float('nan'), self.opposed)):
            with self.assertRaises(ValueError):
                balance_proxy(self.points, normals, mu)


if __name__ == '__main__':
    unittest.main()
