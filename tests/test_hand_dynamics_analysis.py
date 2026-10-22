"""Retained feedback comparisons must respect target changes and sample coverage."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_hand_dynamics import compare_trajectory, fit_effort
from rabokit_hand import HAND_JOINTS


class AnalysisTests(unittest.TestCase):
    def test_effort_fit_uses_constant_targets_and_velocity(self):
        varying = next(iter(HAND_JOINTS))
        commands = [{'robot_tf_reference_stamp_sec': t,
                     'targets_rad': {n: (t if n == varying else 1.) for n in HAND_JOINTS}}
                    for t in [0., 1.]]
        samples = []
        for i in range(1, 6):
            sample = {'stamp': [i, 0], 'position': {}, 'velocity': {}, 'effort': {}}
            for j, n in enumerate(HAND_JOINTS):
                error, velocity = .001 * (i + j), .002 * i * i
                sample['position'][n] = 1. - error
                sample['velocity'][n] = velocity
                sample['effort'][n] = 20. * error - velocity if n != varying else 999.
            samples.append(sample)
        result = fit_effort({'closed': {'issued_commands': commands, 'joint_samples': samples}})
        self.assertAlmostEqual(result['kp'], 20., places=10)
        self.assertAlmostEqual(result['kv'], 1., places=10)
        self.assertNotIn(varying, result['joint_samples_by_stage']['closed'])

    def test_effort_fit_rejects_rank_deficient_or_nonfinite_data(self):
        trace = {'issued_commands': [{'robot_tf_reference_stamp_sec': 0.,
                                     'targets_rad': {n: 1. for n in HAND_JOINTS}}],
                 'joint_samples': [{'stamp': [1, 0], 'position': {n: 0. for n in HAND_JOINTS},
                                    'velocity': {n: 0. for n in HAND_JOINTS},
                                    'effort': {n: 20. for n in HAND_JOINTS}}]}
        with self.assertRaises(ValueError):
            fit_effort({'closed': trace})
        trace['joint_samples'][0]['effort'][next(iter(HAND_JOINTS))] = float('nan')
        with self.assertRaises(ValueError):
            fit_effort({'closed': trace})

    def test_position_comparison_does_not_extrapolate_across_gaps_or_lift(self):
        sample = lambda nano: {'stamp': [5, nano], 'position': {n: 1. for n in HAND_JOINTS}}
        traces = {'closed': {'joint_samples': [sample(t) for t in [0, 20_000_000, 80_000_000, 110_000_000]]}}
        proxy = {'rows': [{'seconds': t, 'tool_rise_m': rise,
                           'hand_position_rad': {n: 1.001 for n in HAND_JOINTS}}
                          for t, rise in [(.01, 0.), (.10, .001)]]}
        result = compare_trajectory(traces, proxy)['closed']
        self.assertEqual(result['matched'], 1)
        self.assertEqual(result['unmatched'], 3)
        self.assertAlmostEqual(result['rmse_rad'], .001)
        proxy['rows'][1]['seconds'] = .005
        with self.assertRaises(ValueError):
            compare_trajectory(traces, proxy)


if __name__ == '__main__':
    unittest.main()
