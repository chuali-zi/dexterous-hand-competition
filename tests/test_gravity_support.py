"""Analytical gravity cases distinguish load support from all-wrench closure."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_grasp_balance import balance_proxy, gravity_support_proxy


class GravitySupportTests(unittest.TestCase):
    def test_opposed_point_pair_carries_gravity_without_full_force_closure(self):
        p, n = [[-0.025, 0, 0], [0.025, 0, 0]], [[1, 0, 0], [-1, 0, 0]]
        out = gravity_support_proxy(p, n, [0, 0, 0], 0.1, 0.4, [2, 2])
        self.assertTrue(out["gravity_support_feasible"])
        self.assertAlmostEqual(out["minimum_total_normal_force_n"], 0.981 / 0.4)
        self.assertFalse(
            balance_proxy(p, n, 0.4)["full_dimensional_positive_equilibrium_proxy"]
        )
        self.assertFalse(out["physical_grasp_verified"])

    def test_friction_or_force_limit_cannot_be_ignored(self):
        p, n = [[-0.025, 0, 0], [0.025, 0, 0]], [[1, 0, 0], [-1, 0, 0]]
        for mu, cap in [(0.0, 20.0), (0.4, 1.0)]:
            self.assertFalse(
                gravity_support_proxy(p, n, [0, 0, 0], 0.1, mu, [cap, cap])[
                    "gravity_support_feasible"
                ]
            )

    def test_downward_pushing_is_not_support_even_with_large_force(self):
        self.assertFalse(
            gravity_support_proxy(
                [[-0.01, 0, 0.025], [0.01, 0, 0.025]],
                [[0, 0, -1]] * 2,
                [0, 0, 0],
                0.1,
                0.8,
                [100, 100],
            )["gravity_support_feasible"]
        )

    def test_palm_support_depends_on_gravity_moment(self):
        centered = gravity_support_proxy(
            [[0, 0, -0.025]], [[0, 0, 1]], [0, 0, 0], 0.1, 0.4, [2]
        )
        offset = gravity_support_proxy(
            [[0.01, 0, -0.025]], [[0, 0, 1]], [0, 0, 0], 0.1, 0.4, [2]
        )
        self.assertTrue(centered["gravity_support_feasible"])
        self.assertFalse(offset["gravity_support_feasible"])

    def test_com_is_not_replaced_by_contact_centroid(self):
        p, n = [[-0.025, 0, 0], [0.025, 0, 0]], [[1, 0, 0], [-1, 0, 0]]
        a = gravity_support_proxy(p, n, [0, 0, 0], 0.1, 0.4, [2, 2])
        b = gravity_support_proxy(
            [[x + 1, y + 2, z + 3] for x, y, z in p], n, [1, 2, 3], 0.1, 0.4, [2, 2]
        )
        self.assertAlmostEqual(
            a["minimum_total_normal_force_n"], b["minimum_total_normal_force_n"]
        )

    def test_nonfinite_or_negative_physical_inputs_are_rejected(self):
        for mass, limits in [(float("nan"), [1]), (0.1, [-1]), (0, [1])]:
            with self.assertRaises(ValueError):
                gravity_support_proxy(
                    [[0, 0, 0]], [[0, 0, 1]], [0, 0, 0], mass, 0.4, limits
                )


if __name__ == "__main__":
    unittest.main()
