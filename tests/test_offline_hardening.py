"""Portable reviewer reproductions and independent geometry/contract regressions."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from hand_competition.closed_loop.artifacts import Trace, canonical, digest, decode_json
from hand_competition.closed_loop.contracts import Config, ObservationGate, ObservationError, ActionStatus
from hand_competition.closed_loop.controller import Controller, TrialFailure
from hand_competition.closed_loop.experiments import run_trial, replay, code_identity, batch
from hand_competition.closed_loop.geometry import Camera, triangulate, distance
from hand_competition.closed_loop.synthetic import SyntheticBackend, SensorNoise


class PhaseTrace(Trace):
    def __init__(self, path, manifest, backend):
        super().__init__(path, manifest); self.backend = backend

    def event(self, kind, time_s, **payload):
        if kind == 'settle_started': self.backend.settling = True
        super().event(kind, time_s, **payload)


class SparseSettling(SyntheticBackend):
    settling = False
    jump = 3.1
    def _advance(self):
        if self.settling: self._time += self.jump
        super()._advance()


class OverDeadlineSettling(SparseSettling):
    jump = 31.0


class SlowComplete(SyntheticBackend):
    def _advance(self):
        if self._pending: self._time += 1.9
        super()._advance()


class IncompleteRetreat(SyntheticBackend):
    def _advance(self):
        pending = self._pending; action = self._actions[pending]['action'] if pending else None
        super()._advance()
        if action and action.kind == 'move' and self._moves == 7 and self._pending is None:
            self._tip = (0.16, 0.42, 0.025)


class OccludedTransit(SyntheticBackend):
    def _render(self, camera, hidden):
        return super()._render(camera, hidden or self._moves == 5)


class HugeTip(SyntheticBackend):
    def observe(self):
        raw = super().observe(); raw['robot']['tip_position_m'] = (10**400, 0.4, 0.025)
        return raw


class InvalidStatus(SyntheticBackend):
    def status(self, action_id): return None


def run_custom(path, cls, config=None):
    backend = cls(); config = config or Config()
    manifest = {'schema_version': 2, 'environment': backend.environment, 'producer': backend.producer,
                'config': asdict(config), 'cameras': [asdict(c) for c in backend.cameras], 'code': code_identity()}
    trace = PhaseTrace(path, manifest, backend)
    result = Controller(backend, config, trace).run()
    return result, backend


def rehash(path, events):
    previous = '0'*64
    for index, event in enumerate(events):
        event.pop('sha256', None); event['index'] = index; event['previous_sha256'] = previous
        event['sha256'] = digest(canonical(event).encode()); previous = event['sha256']
    (path/'events.jsonl').write_text(''.join(canonical(event)+'\n' for event in events))
    result = next(event['payload']['result'] for event in reversed(events) if event['kind'] == 'result')
    (path/'result.json').write_text(json.dumps(result))


class ReviewerRegressionTests(unittest.TestCase):
    def test_all_reported_execution_faults_terminate_with_a_result(self):
        cases = [(SparseSettling, 'trial_timeout'), (OverDeadlineSettling, 'trial_timeout'),
                 (SlowComplete, 'action_timeout'), (IncompleteRetreat, 'motion_not_verified'),
                 (OccludedTransit, 'carried_object_unobserved'), (HugeTip, 'invalid_observation'),
                 (InvalidStatus, 'action_protocol_failure')]
        with tempfile.TemporaryDirectory() as directory:
            for cls, category in cases:
                path = Path(directory)/cls.__name__
                with self.subTest(fault=cls.__name__):
                    result, backend = run_custom(path, cls)
                    self.assertFalse(result['success']); self.assertEqual(result['terminal_failure'], category)
                    self.assertEqual(backend.stop_count, 1); self.assertTrue((path/'result.json').exists())
                    self.assertTrue(replay(path)['control_record_consistent'])
                    if cls is OccludedTransit:
                        events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
                        phases = [e['payload']['phase'] for e in events if e['kind'] == 'action_submitted']
                        self.assertNotIn('target_descent', phases); self.assertNotIn('release', phases)

    def test_clock_reset_late_submit_and_nonzero_clock_origin(self):
        class ResetClock(SyntheticBackend):
            def observe(self):
                raw = super().observe()
                if self._sequence == 4: self._time = 0
                return raw
        class SlowSubmit(SyntheticBackend):
            def submit(self, action):
                self._time += 2.0
                return super().submit(action)
        class ShiftedClock(SyntheticBackend):
            def __init__(self):
                super().__init__(); self._time = 5.0
        with tempfile.TemporaryDirectory() as directory:
            for cls, category in [(ResetClock, 'backend_clock_failure'), (SlowSubmit, 'action_timeout'), (ShiftedClock, None)]:
                path = Path(directory)/cls.__name__; result, _ = run_custom(path, cls)
                self.assertEqual(result['terminal_failure'], category)
                self.assertEqual(result['success'], category is None)
                self.assertTrue(replay(path)['trace_verified'])

    def test_finalized_controller_and_log_failure_cannot_continue_actions(self):
        class BrokenEventTrace(PhaseTrace):
            def event(self, kind, stamp, **payload):
                if kind == 'grasp_verified': raise OSError('test write failure')
                return super().event(kind, stamp, **payload)
        with tempfile.TemporaryDirectory() as directory:
            backend = SyntheticBackend(); config = Config(); path = Path(directory)/'log-failure'
            manifest = {'schema_version': 2, 'environment': backend.environment, 'producer': backend.producer,
                        'config': asdict(config), 'cameras': [asdict(c) for c in backend.cameras]}
            trace = BrokenEventTrace(path, manifest, backend); controller = Controller(backend, config, trace)
            result = controller.run()
            self.assertFalse(result['success']); self.assertFalse(result['trace_complete'])
            self.assertEqual(result['terminal_failure'], 'artifact_io_failure'); self.assertEqual(backend.stop_count, 1)
            before = len(backend._actions)
            with self.assertRaises(TrialFailure): controller.command('close', 'grasp')
            with self.assertRaises(ValueError): controller.run()
            self.assertEqual(len(backend._actions), before)
            with self.assertRaises(ValueError): replay(path)

    def test_replay_requires_sensor_action_and_terminal_facts_not_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run_trial(root/'nominal')
            original = [json.loads(line) for line in (root/'nominal/events.jsonl').read_text().splitlines()]
            start = next(e['time_s'] for e in original if e['kind'] == 'settle_started')
            variants = {
                'no-actions-no-grasp-images': lambda es: [e for e in es if e['kind'] not in ('action_submitted', 'action_status')
                    and (e['kind'] != 'observation' or e['time_s'] < 0.25 or e['time_s'] >= start)],
                'wrong-grasp-coordinate': lambda es: [dict(e, payload={'position_m': [99, 99, 99]})
                    if e['kind'] == 'grasp_verified' else e for e in es],
                'missing-stop': lambda es: [e for e in es if e['kind'] != 'stop_request_succeeded'],
                'duplicate-result': lambda es: es+[es[-1]],
                'post-terminal-action': lambda es: es+[dict(es[0], kind='action_submitted', time_s=999,
                    payload={'action_id': 'after', 'phase': 'bad', 'action': {'kind': 'move', 'target_m': [999]*3}})],
                'unknown-payload-field': lambda es: [dict(e, payload={**e['payload'], 'object_pose': [0, 0, 0]})
                    if e['kind'] == 'action_status' else e for e in es],
                'unknown-event': lambda es: [dict(e, kind='unknown') if e['kind'] == 'settle_started' else e for e in es],
                'reused-action-id': lambda es: [dict(e, payload={**e['payload'], 'action_id': 'a0001'})
                    if e['kind'] == 'action_submitted' else e for e in es],
            }
            for name, edit in variants.items():
                path = root/name; shutil.copytree(root/'nominal', path)
                rehash(path, edit(json.loads(json.dumps(original))))
                with self.subTest(variant=name), self.assertRaises(ValueError): replay(path)
            path = root/'manifest-change'; shutil.copytree(root/'nominal', path)
            manifest = json.loads((path/'manifest.json').read_text()); manifest['code']['git_commit'] = '0'*40
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'manifest binding'): replay(path)

    def test_unsubstantiated_failure_is_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'failure'; run_trial(path, 'motion_failed')
            events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            kept = [e for e in events if e['kind'] in ('trial_started', 'observation', 'stop_request_succeeded', 'result')]
            kept[-1]['payload']['result']['action_count'] = 0
            rehash(path, kept)
            with self.assertRaisesRegex(ValueError, 'terminal failure evidence'): replay(path)

    def test_carry_slip_is_detected_before_descent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'slip'; result = run_trial(path, 'slip_during_transit')
            self.assertFalse(result['success']); self.assertEqual(result['terminal_failure'], 'carried_object_lost')
            self.assertTrue(replay(path)['trace_verified'])


class ContractRegressionTests(unittest.TestCase):
    def test_invalid_dimensions_numbers_names_and_config_fail_cleanly(self):
        camera = SyntheticBackend.cameras[0]
        for override in ({'width': 160.5}, {'width': True}, {'origin': None}, {'rotation': [[1, 0]]}, {'name': '../escape'}):
            with self.subTest(camera=override), self.assertRaises(ValueError): replace(camera, **override)
        for override in ({'target': [0.1, 0.4]}, {'target': [0.16, 0.42, 0.1]}, {'safe_height_m': 0.4},
                         {'stability_m': 10}, {'motion_completion_m': 0.08}, {'action_timeout_s': True},
                         {'lift_min_m': 10**400}, {'unknown': 1}):
            with self.subTest(config=override), self.assertRaises(ValueError): Config.from_dict(override)
        with self.assertRaises(ValueError): ObservationGate([], 'x')
        with self.assertRaises(ValueError): ObservationGate([camera, camera], 'x')

    def test_robot_feedback_and_envelope_have_independent_provenance_and_time(self):
        for fault in ('future-envelope', 'robot-stale', 'robot-producer', 'robot-sequence', 'robot-skew', 'huge-finite-tip'):
            backend = SyntheticBackend(); gate = ObservationGate(backend.cameras, backend.producer)
            raw = backend.observe(); gate.accept(raw, backend.now()); raw = backend.observe()
            if fault == 'future-envelope': raw['time_s'] += 0.1
            elif fault == 'robot-stale': raw['robot']['captured_s'] = 0
            elif fault == 'robot-producer': raw['robot']['producer'] = 'command_echo'
            elif fault == 'robot-sequence': raw['robot']['sequence'] = 1
            elif fault == 'robot-skew': raw['robot']['captured_s'] -= 0.05
            else: raw['robot']['tip_position_m'] = [1e200, 0.4, 0.025]
            with self.subTest(fault=fault), self.assertRaises(ObservationError): gate.accept(raw, backend.now())

    def test_strict_json_and_batch_validation_happen_before_execution(self):
        for text in ('{"target":1,"target":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError): decode_json(text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'batch'
            for scenarios, seeds in [(['nominal'], [0, 0]), ([], [0]), (['nominal', 'nominal'], [0]), (['nominal'], [])]:
                with self.assertRaises(ValueError): batch(path, scenarios, seeds)
                self.assertFalse(path.exists())

    def test_custom_configuration_is_saved_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = Config(settle_s=3.5, recovery_enabled=False)
            first = run_trial(root/'first', config=config, noise=SensorNoise(0, 0.002))
            manifest = json.loads((root/'first/manifest.json').read_text())
            self.assertIn('--config', manifest['reproduce']); self.assertIn('--calibration-shift-mm', manifest['reproduce'])
            loaded = Config.from_dict(json.loads((root/'first/config.json').read_text()))
            second = run_trial(root/'second', config=loaded, noise=SensorNoise(0, 0.002))
            self.assertEqual(first, second); self.assertTrue(replay(root/'first')['trace_verified'])

    def test_replay_can_check_an_external_runtime_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run'; run_trial(path)
            checked = replay(path, code_identity()['runtime_tree_sha256'])
            self.assertTrue(checked['runtime_identity_checked']); self.assertFalse(checked['sensor_authenticity_verified'])
            with self.assertRaisesRegex(ValueError, 'trusted expectation'): replay(path, '0'*64)

    def test_independent_pinhole_fixture_uses_closed_form_pixels(self):
        # Two cameras at x=-0.1/+0.1, optical axes coincide with base axes.
        # For P=(0.05,0.02,1), f=100,c=(80,60): pixels are (95,62),(75,62).
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        left = Camera('left', (-0.1, 0, 0), identity, fx=100, fy=100)
        right = Camera('right', (0.1, 0, 0), identity, fx=100, fy=100)
        point, _ = triangulate(left, (95, 62), right, (75, 62))
        self.assertLess(distance(point, (0.05, 0.02, 1.0)), 1e-10)
        # A consistent common extrinsic translation yields low residual but bias;
        # it must never be sold as an independent accuracy measurement.
        shifted = (replace(left, origin=(0, 0, 0)), replace(right, origin=(0.2, 0, 0)))
        biased, residual = triangulate(shifted[0], (95, 62), shifted[1], (75, 62))
        self.assertLess(residual, 1e-10); self.assertAlmostEqual(biased[0], 0.15)


if __name__ == '__main__': unittest.main()
