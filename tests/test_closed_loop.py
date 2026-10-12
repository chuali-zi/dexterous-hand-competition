"""Sensor-path, recovery and artifact tests; all robot scenes are synthetic."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import unittest

from hand_competition.closed_loop.artifacts import Trace, canonical, digest
from hand_competition.closed_loop.contracts import Config, ObservationGate, ObservationError, guard_move
from hand_competition.closed_loop.controller import Controller
from hand_competition.closed_loop.experiments import run_trial, replay, contained_file, code_identity
from hand_competition.closed_loop.geometry import look_at, triangulate, distance
from hand_competition.closed_loop.preflight import inspect_template
from hand_competition.closed_loop.synthetic import SyntheticBackend
from hand_competition.closed_loop.vision import estimate_single_block


class GeometryAndSensorTests(unittest.TestCase):
    def test_pinhole_stereo_round_trip_at_multiple_heights(self):
        a, b = SyntheticBackend.cameras
        for point in [(-0.18, 0.4, 0.025), (0.16, 0.42, 0.025), (0, 0.4, 0.2)]:
            result, residual = triangulate(a, a.project(point), b, b.project(point))
            self.assertLess(distance(point, result), 1e-10)
            self.assertLess(residual, 1e-10)

    def test_parallel_rays_and_behind_camera_rejected(self):
        a, _ = SyntheticBackend.cameras
        with self.assertRaises(ValueError): triangulate(a, (80, 60), a, (80, 60))
        with self.assertRaises(ValueError): a.project(a.origin)

    def test_rgb_estimation_uses_raster_not_pose_field(self):
        backend = SyntheticBackend(seed=7)
        raw = backend.observe()
        self.assertEqual(set(raw), {'schema_version', 'producer', 'calibration_id', 'time_s', 'frames', 'robot'})
        observation = ObservationGate(backend.cameras, backend.producer).accept(raw, backend.now())
        estimate = estimate_single_block(observation, backend.cameras)
        self.assertTrue(-0.20 < estimate.position_m[0] < -0.12)
        self.assertTrue(0.38 < estimate.position_m[1] < 0.46)
        self.assertEqual(estimate.color, 'red')

    def test_source_fields_calibration_encoding_and_truth_fail_closed(self):
        for fault in ('truth', 'producer', 'calibration', 'encoding', 'nested_truth', 'camera_type', 'dimension_type', 'truncated'):
            backend = SyntheticBackend(); raw = backend.observe()
            if fault == 'truth': raw['object_pose'] = [0, 0, 0]
            elif fault == 'producer': raw['producer'] = 'unreviewed_topic'
            elif fault == 'calibration': raw['calibration_id'] = 'wrong'
            elif fault == 'encoding': raw['frames'][0]['encoding'] = 'bgr8'
            elif fault == 'nested_truth': raw['robot']['object_tf'] = {}
            elif fault == 'camera_type': raw['frames'][0]['camera_id'] = []
            elif fault == 'dimension_type': raw['frames'][0]['width'] = 160.0
            else: raw['frames'][0]['rgb'] = b'bad'
            with self.subTest(fault=fault), self.assertRaises(ObservationError):
                ObservationGate(backend.cameras, backend.producer).accept(raw, backend.now())

    def test_stale_duplicate_clock_and_skew_rejected(self):
        for fault in ('stale', 'duplicate', 'clock', 'skew'):
            backend = SyntheticBackend(); gate = ObservationGate(backend.cameras, backend.producer)
            first = backend.observe(); gate.accept(first, backend.now())
            raw = backend.observe()
            if fault == 'stale': raw['frames'][0]['captured_s'] = -1
            elif fault == 'duplicate': raw['frames'][0]['sequence'] = first['frames'][0]['sequence']
            elif fault == 'clock': raw['time_s'] = 0
            else: raw['frames'][0]['captured_s'] -= 0.1
            with self.subTest(fault=fault), self.assertRaises(ObservationError): gate.accept(raw, backend.now())

    def test_low_lateral_waypoint_and_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError): guard_move((0, 0.4, 0.025), (0.16, 0.4, 0.025), Config())
        guard_move((0, 0.4, 0.20), (0.16, 0.4, 0.20), Config())
        for kwargs in ({'settle_s': 2.9}, {'safe_height_m': 0.1}, {'target': (9, 9, 0.025)}, {'trial_timeout_s': 0}):
            with self.assertRaises(ValueError): Config(**kwargs)


class ExperimentTests(unittest.TestCase):
    def run_case(self, root, scenario, recovery=True, seed=0):
        return run_trial(root/scenario, scenario, seed, Config(recovery_enabled=recovery))

    def test_nominal_records_grasp_release_and_continuous_stability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); result = self.run_case(root, 'nominal')
            self.assertTrue(result['success']); self.assertGreaterEqual(result['stability_verified_s'], 3)
            self.assertFalse(result['official_platform']); self.assertIsNone(result['official_score'])
            self.assertTrue(result['stop_request_succeeded']); self.assertFalse(result['physical_stop_verified'])
            verification = replay(root/'nominal')
            self.assertTrue(verification['trace_verified']); self.assertGreater(verification['observations_recomputed'], 30)

    def test_transient_faults_recover_from_actual_new_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in ('empty_grasp_once', 'slip_once', 'release_stuck_once', 'drift_after_release'):
                with self.subTest(scenario=scenario):
                    result = self.run_case(root, scenario)
                    self.assertTrue(result['success'], result)
                    self.assertGreater(result['recoveries'], 0)
                    self.assertTrue(replay(root/scenario)['trace_verified'])

    def test_ablation_does_not_confuse_command_complete_with_grasp_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in ('empty_grasp_once', 'slip_once', 'release_stuck_once', 'drift_after_release'):
                result = self.run_case(root, scenario, recovery=False)
                self.assertFalse(result['success'])
                self.assertIn(result['terminal_failure'], ('grasp_not_verified', 'release_not_verified', 'placement_drift'))

    def test_initial_occlusion_waits_and_never_creates_fake_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); result = self.run_case(root, 'occluded_start')
            self.assertTrue(result['success'])
            events = [json.loads(x) for x in (root/'occluded_start/events.jsonl').read_text().splitlines()]
            first_action = next(e for e in events if e['kind'] == 'action_submitted')
            self.assertGreaterEqual(first_action['time_s'], 0.9)

    def test_timeout_actuator_and_bad_observations_stop(self):
        expected = {'move_timeout': 'action_timeout', 'motion_failed': 'actuator_failure',
                    'pose_leak': 'invalid_observation', 'camera_skew': 'invalid_observation',
                    'stale_after_close': 'invalid_observation', 'duplicate_after_close': 'invalid_observation'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario, category in expected.items():
                result = self.run_case(root, scenario)
                self.assertFalse(result['success']); self.assertEqual(result['terminal_failure'], category)
                self.assertTrue(result['stop_request_succeeded'])
                if scenario in ('pose_leak', 'camera_skew'): self.assertEqual(result['action_count'], 0)

    def test_continuous_bad_visibility_reaches_deadline_without_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_trial(Path(directory)/'occluded', 'occluded_always', 0, Config(trial_timeout_s=1.0))
            self.assertFalse(result['success']); self.assertEqual(result['terminal_failure'], 'trial_timeout')
            self.assertEqual(result['action_count'], 0)

    def test_transport_and_stop_failure_are_logged_not_silent(self):
        class BrokenSensor(SyntheticBackend):
            def observe(self): raise TimeoutError('not included in public log')
        class BrokenStop(BrokenSensor):
            def stop(self): raise OSError('not included in public log')
        with tempfile.TemporaryDirectory() as directory:
            for cls in (BrokenSensor, BrokenStop):
                backend = cls(); trace = Trace(Path(directory)/cls.__name__, {'test': 'synthetic'})
                result = Controller(backend, Config(), trace).run()
                expected = 'stop_transport_failure' if cls is BrokenStop else 'sensor_transport_failure'
                self.assertEqual(result['terminal_failure'], expected)
                self.assertFalse(result['success'])

    def test_same_seed_produces_identical_sensor_action_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_trial(root/'a', 'nominal', 8)
            second = run_trial(root/'b', 'nominal', 8)
            self.assertEqual(first, second)
            def semantics(path):
                return [{k: v for k, v in json.loads(line).items()
                         if k not in ('sha256', 'previous_sha256', 'manifest_sha256')}
                        for line in path.read_text().splitlines()]
            # V2 binds the manifest, whose replay command includes each output
            # path. Sensor/action content remains deterministic across paths.
            self.assertEqual(semantics(root/'a/events.jsonl'), semantics(root/'b/events.jsonl'))
            for frame in (root/'a/frames').glob('*.ppm'):
                self.assertEqual(frame.read_bytes(), (root/'b/frames'/frame.name).read_bytes())

    def test_manifest_records_actual_checkout_and_runtime_fingerprints(self):
        identity = code_identity()
        self.assertRegex(identity['git_commit'], r'^[0-9a-f]{40}$')
        self.assertIn('closed_loop/controller.py', identity['runtime_files'])
        self.assertRegex(identity['runtime_tree_sha256'], r'^[0-9a-f]{64}$')

    def test_rgb_corruption_and_path_escape_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.run_case(root, 'nominal')
            path = next((root/'nominal/frames').glob('*.ppm'))
            path.write_bytes(path.read_bytes()[:-1]+b'\x00')
            with self.assertRaisesRegex(ValueError, 'RGB artifact hash'): replay(root/'nominal')
            with self.assertRaises(ValueError): contained_file(root, '../escape.ppm')

    def test_forged_success_without_stability_is_rejected_even_with_rehashed_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_case(root, 'empty_grasp_once', recovery=False)
            path = root/'empty_grasp_once'
            events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            result = json.loads((path/'result.json').read_text())
            result['success'] = True
            events[-1]['payload']['result'] = result
            previous = '0'*64
            for event in events:
                event.pop('sha256'); event['previous_sha256'] = previous
                event['sha256'] = digest(canonical(event).encode()); previous = event['sha256']
            (path/'events.jsonl').write_text(''.join(canonical(e)+'\n' for e in events))
            (path/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'lacks grasp/stability'): replay(path)

    def test_aborted_fault_is_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(Path(directory), 'empty_grasp_once', recovery=False)
            self.assertEqual(len(result['failures']), 1)

    def test_template_inspector_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'main.py').write_text('raise RuntimeError("do not import")\nrobot.attach("x")\narm.move_to(linear=True)\n')
            (root/'.env').write_text('dummy_test_text_not_a_secret')
            report = inspect_template(root)
            self.assertFalse(report['executed_source']); self.assertEqual(len(report['findings']), 2)
            self.assertEqual(list(report['files_sha256']), ['main.py'])


if __name__ == '__main__': unittest.main()
