"""Independent geometry fixtures and regressions for stationary observations."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from hand_competition.closed_loop.contracts import Config
from hand_competition.closed_loop.geometry import Camera, distance
from hand_competition.closed_loop.robust import Sample, estimate_bundle, BundleError, bounded_placement
from hand_competition.closed_loop.experiments import run_trial, replay
from hand_competition.closed_loop.synthetic import SensorNoise
from test_offline_hardening import rehash


class BundleTests(unittest.TestCase):
    def setUp(self):
        rotation=((1,0,0),(0,1,0),(0,0,1))
        self.cameras=(Camera('a',(-.1,0,0),rotation,fx=100,fy=100),Camera('b',(.1,0,0),rotation,fx=100,fy=100))
        self.config=Config(robust_observations=True)

    def samples(self,n=9):
        # Independently written closed-form pinhole fixture: z=1, x=y=0.
        return [Sample(i,(i+1)*.1,(0,.4,.2),('a','b'),((90.,60.),(70.,60.)),'red') for i in range(n)]

    def test_independent_pixels_recover_point_and_constant_bias_remains(self):
        samples=self.samples(); value=estimate_bundle(samples,self.cameras,self.config)
        self.assertLess(distance(value.position_m,(0,0,1)),1e-10)
        shifted=tuple(replace(c,origin=(c.origin[0]+.01,0,0)) for c in self.cameras)
        biased=estimate_bundle(samples,shifted,self.config)
        self.assertAlmostEqual(distance(biased.position_m,(0,0,1)),.01)
        self.assertEqual(biased.radius_m,0) # random variance cannot detect calibration bias

    def test_late_single_frame_step_is_not_erased_by_mean(self):
        samples=self.samples()
        samples[-1]=replace(samples[-1],pixels=((92.,60.),(72.,60.)))
        with self.assertRaisesRegex(BundleError,'temporal_change'): estimate_bundle(samples,self.cameras,self.config)

    def test_sensor_gap_robot_motion_identity_and_pixels_are_stationary_requirements(self):
        for fault in ('gap','tip','color','span'):
            samples=self.samples()
            if fault=='gap': samples[-1]=replace(samples[-1],sensor_time_s=2.)
            elif fault=='tip': samples[-1]=replace(samples[-1],tip=(.1,.4,.2))
            elif fault=='color': samples[-1]=replace(samples[-1],color='green')
            else: samples[-1]=replace(samples[-1],pixels=((98.,60.),(70.,60.)))
            with self.subTest(fault=fault),self.assertRaises(BundleError): estimate_bundle(samples,self.cameras,self.config)

    def test_uncertainty_tightens_goal_and_hand_clearance(self):
        value=estimate_bundle(self.samples(),self.cameras,self.config)
        ideal=replace(value,position_m=self.config.target)
        self.assertTrue(bounded_placement(ideal,(.16,.42,.20),self.config))
        near_edge=replace(ideal,position_m=(.184,.42,.025))
        self.assertFalse(bounded_placement(near_edge,(.16,.42,.20),self.config))
        self.assertFalse(bounded_placement(ideal,(.16,.42,.105),self.config))

    def test_config_thresholds_unchanged_and_bad_bundle_parameters_rejected(self):
        c=self.config
        self.assertEqual((c.goal_tolerance_m,c.stability_m,c.hand_separation_m,c.settle_s),(.025,.015,.08,3.))
        for kwargs in ({'robust_min_samples':True},{'robust_min_samples':2},{'robust_max_samples':100},
                       {'robust_sigma_factor':0},{'robust_change_floor_px':float('nan')}):
            with self.assertRaises(ValueError): Config(**kwargs)

    def test_robust_transient_recoveries_use_new_rgb_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            for scenario in ('empty_grasp_once','slip_once','release_stuck_once','drift_after_release'):
                path=Path(directory)/scenario
                with self.subTest(scenario=scenario):
                    result=run_trial(path,scenario,config=self.config)
                    self.assertTrue(result['success'],result)
                    self.assertGreater(result['recoveries'],0)
                    self.assertTrue(replay(path)['trace_verified'])

    def test_noisy_development_fixture_has_full_replay_and_forged_statistics_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'valid'
            result=run_trial(path,seed=20,noise=SensorNoise(1))
            self.assertTrue(result['success']); self.assertGreater(replay(path)['bundles_recomputed'],0)
            original=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            for fault in ('point','uncertainty','omit-sample','duplicate-sample','stability-label'):
                bad=root/fault; shutil.copytree(path,bad); events=json.loads(json.dumps(original))
                event=next(e for e in events if e['kind']==('stability_verified' if fault=='stability-label' else 'robust_observation'))
                p=event['payload']
                if fault=='point': p['position_m'][0]+=.01
                elif fault=='uncertainty': p['uncertainty_xyz_m']=[0,0,0]
                elif fault=='omit-sample': p['sample_indices'].pop(3)
                elif fault=='duplicate-sample': p['sample_indices'][3]=p['sample_indices'][2]
                else: p['position_m'][0]+=.01
                rehash(bad,events)
                with self.subTest(fault=fault),self.assertRaises(ValueError): replay(bad)


if __name__=='__main__': unittest.main()
