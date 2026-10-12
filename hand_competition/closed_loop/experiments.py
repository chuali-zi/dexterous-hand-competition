"""Run, batch-evaluate and independently replay the saved RGB evidence."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from .artifacts import Trace, canonical, digest, read_ppm
from .contracts import Config, ObservationGate, inside_xy
from .controller import Controller
from .geometry import Camera, distance
from .synthetic import SyntheticBackend
from .vision import estimate_single_block, VisionError


def runtime_sources():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted(root.rglob('*.py'))
            if '__pycache__' not in p.parts}


LOADED_SOURCE_SNAPSHOT = runtime_sources()


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = runtime_sources()
    if files != LOADED_SOURCE_SNAPSHOT:
        raise ValueError('runtime source changed during process; start a fresh experiment process')
    repo = root.parent
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True)
    state = subprocess.run(['git', 'status', '--porcelain'], cwd=repo, capture_output=True, text=True)
    return {'git_commit': revision.stdout.strip() if revision.returncode == 0 else None,
            'dirty_tree': bool(state.stdout.strip()), 'runtime_files': files,
            'runtime_tree_sha256': digest(canonical(files).encode())}


def run_trial(directory, scenario='nominal', seed=0, config=None):
    config = config or Config()
    backend = SyntheticBackend(scenario, seed)
    manifest = {'schema_version': 1, 'environment': backend.environment,
                'producer': backend.producer, 'python': platform.python_version(),
                'host': {'system': platform.system(), 'machine': platform.machine()},
                'clock': '0.1s deterministic synthetic steps; independent wall deadline also enforced',
                'config': asdict(config), 'cameras': [asdict(c) for c in backend.cameras],
                'code': code_identity(),
                'reproduce': ['python', '-m', 'hand_competition.closed_loop', 'run',
                              '--scenario', scenario, '--seed', str(seed), '--output', '<new-output-directory>']}
    if not config.recovery_enabled: manifest['reproduce'].append('--without-recovery')
    trace = Trace(directory, manifest)
    return Controller(backend, config, trace).run()


def batch(directory, scenarios, seeds, recovery_enabled=True):
    root = Path(directory); root.mkdir(parents=True, exist_ok=False)
    trials = []; grouped = {}
    for scenario in scenarios:
        for seed in seeds:
            path = root/f'{scenario}-seed-{seed}'
            result = run_trial(path, scenario, seed, Config(recovery_enabled=recovery_enabled))
            trials.append({'scenario': scenario, 'seed': seed, 'directory': path.name, **result})
            grouped.setdefault(scenario, []).append(result)
    summary = {
        'schema_version': 1, 'feedback_level': 'synthetic_rgb_kinematic',
        'official_platform': False, 'official_score': None,
        'recovery_enabled': recovery_enabled, 'trial_count': len(trials),
        'success_count': sum(t['success'] for t in trials),
        'scenarios': {}, 'trials': trials,
    }
    for scenario, results in grouped.items():
        times = sorted(r['elapsed_s'] for r in results)
        categories = {}
        for result in results:
            for failure in result['failures']:
                category = failure['category']; categories[category] = categories.get(category, 0)+1
        summary['scenarios'][scenario] = {
            'trials': len(results), 'successes': sum(r['success'] for r in results),
            'failures_by_category': categories,
            'elapsed_s_min': times[0], 'elapsed_s_max': times[-1],
            'mean_recoveries': sum(r['recoveries'] for r in results)/len(results),
        }
    (root/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return summary


def contained_file(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('artifact path escapes run directory')
    return path


def replay(directory):
    """Verify/recompute recorded sensor evidence; does not execute robot actions."""
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    cameras = tuple(Camera(**value) for value in manifest['cameras'])
    gate = ObservationGate(cameras, manifest['producer'])
    previous = '0'*64; observations = 0; rejected = 0; actions = {}; final = None
    visual_samples = []; stability_event = None; grasp_verified = False
    for index, line in enumerate((root/'events.jsonl').read_text().splitlines()):
        event = json.loads(line); recorded_hash = event.pop('sha256')
        if event['index'] != index or event['previous_sha256'] != previous:
            raise ValueError('broken event sequence/hash chain')
        if digest(canonical(event).encode()) != recorded_hash: raise ValueError('event content hash mismatch')
        previous = recorded_hash; payload = event['payload']
        if event['kind'] == 'observation':
            frames = []
            for frame in payload['frames']:
                path = contained_file(root, frame['file'])
                if digest(path.read_bytes()) != frame['file_sha256']: raise ValueError('RGB artifact hash mismatch')
                width, height, pixels = read_ppm(path)
                if width != frame['width'] or height != frame['height']: raise ValueError('RGB artifact dimension mismatch')
                frames.append({k: v for k, v in frame.items() if k not in ('file', 'file_sha256')})
                frames[-1]['rgb'] = pixels
            observation = gate.accept({'schema_version': 1, 'producer': payload['producer'],
                                       'calibration_id': payload['calibration_id'], 'time_s': event['time_s'],
                                       'robot': payload['robot'], 'frames': frames}, event['time_s'])
            try: estimate = estimate_single_block(observation, cameras)
            except VisionError:
                if payload['visual_position_m'] is not None: raise ValueError('recorded estimate not reproducible')
            else:
                if payload['visual_position_m'] is None or distance(estimate.position_m, payload['visual_position_m']) > 1e-9:
                    raise ValueError('visual estimate differs on replay')
            visual_samples.append({'time_s': event['time_s'], 'point': payload['visual_position_m'],
                                   'tip': payload['robot']['tip_position_m']})
            observations += 1
        elif event['kind'] == 'observation_rejected': rejected += 1
        elif event['kind'] == 'action_submitted': actions[payload['action_id']] = payload['action']
        elif event['kind'] == 'action_status' and payload['action_id'] not in actions:
            raise ValueError('status without submitted action')
        elif event['kind'] == 'grasp_verified': grasp_verified = True
        elif event['kind'] == 'stability_verified': stability_event = event
        elif event['kind'] == 'result': final = payload['result']
    if final is None: raise ValueError('incomplete trace: missing result')
    if final != json.loads((root/'result.json').read_text()): raise ValueError('result file differs from event log')
    if final['success']:
        config = Config(**manifest['config'])
        if not grasp_verified or stability_event is None:
            raise ValueError('successful record lacks grasp/stability evidence')
        duration = stability_event['payload']['duration_s']; end = stability_event['time_s']; start = end-duration
        if duration < config.settle_s or final['stability_verified_s'] != duration:
            raise ValueError('successful record has insufficient stability duration')
        samples = [sample for sample in visual_samples if start-1e-6 <= sample['time_s'] <= end+1e-6]
        if not samples or samples[0]['time_s'] > start+1e-6 or samples[-1]['time_s'] < end-1e-6:
            raise ValueError('stability evidence window is incomplete')
        anchor = samples[0]['point']
        for i, sample in enumerate(samples):
            point = sample['point']
            if point is None or not inside_xy(point, config.build_xy):
                raise ValueError('stability sample missing/outside configured build zone')
            if (distance(point, anchor) > config.stability_m or distance(point, config.target) > config.goal_tolerance_m
                    or distance(point, sample['tip']) < 0.08):
                raise ValueError('stability sample does not confirm motion-free placement and hand separation')
            if i and sample['time_s']-samples[i-1]['time_s'] > gate.max_age_s:
                raise ValueError('gap in continuous visual stability evidence')
    return {'feedback_level': manifest['environment']['feedback_level'],
            'official_platform': False, 'observations_recomputed': observations,
            'rejected_observation_events': rejected, 'actions_recorded': len(actions),
            'trace_verified': True, 'recorded_success': final['success']}
