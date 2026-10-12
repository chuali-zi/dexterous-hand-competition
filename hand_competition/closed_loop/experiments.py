"""Run, batch-evaluate and independently replay the saved RGB evidence."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import re

from .artifacts import Trace, canonical, digest, read_ppm, load_json, decode_json
from .contracts import Config, ObservationGate, inside_xy, finite, PHASE_ACTIONS
from .controller import Controller
from .geometry import Camera, distance, sub, add
from .synthetic import SyntheticBackend, SensorNoise, SCENARIOS
from .vision import estimate_single_block, observe_single_block_pixels, VisionError
from . import bundle_replay
from .path_replay import PathVerifier


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


def run_trial(directory, scenario='nominal', seed=0, config=None, noise=None):
    config = config or Config(robust_observations=True)
    if type(seed) is not int: raise ValueError('seed must be an integer')
    noise = noise or SensorNoise()
    backend = SyntheticBackend(scenario, seed, noise)
    manifest = {'schema_version': 4, 'environment': backend.environment,
                'producer': backend.producer, 'python': platform.python_version(),
                'host': {'system': platform.system(), 'machine': platform.machine()},
                'clock': '0.1s deterministic synthetic steps; independent wall deadline also enforced',
                'config': asdict(config), 'cameras': [asdict(c) for c in backend.cameras],
                'code': code_identity(),
                'reproduce': ['python', '-m', 'hand_competition.closed_loop', 'run',
                              '--scenario', scenario, '--seed', str(seed), '--config', str(Path(directory)/'config.json'),
                              '--pixel-jitter', str(noise.pixel_jitter_px),
                              '--calibration-shift-mm', str(noise.calibration_shift_m*1000),
                              '--output', '<new-output-directory>']}
    if not config.recovery_enabled: manifest['reproduce'].append('--without-recovery')
    trace = Trace(directory, manifest)
    return Controller(backend, config, trace).run()


def batch(directory, scenarios, seeds, recovery_enabled=None, config=None, noise=None):
    if (not isinstance(scenarios, (tuple, list)) or not scenarios or any(s not in SCENARIOS for s in scenarios)
            or len(set(scenarios)) != len(scenarios)):
        raise ValueError('batch scenarios must be nonempty, known and unique')
    if (not isinstance(seeds, (tuple, list)) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('batch seeds must be nonempty unique integers')
    config = config or Config(robust_observations=True)
    if recovery_enabled is not None:
        config = replace(config, recovery_enabled=recovery_enabled)
    root = Path(directory); root.mkdir(parents=True, exist_ok=False)
    trials = []; grouped = {}
    for scenario in scenarios:
        for seed in seeds:
            path = root/f'{scenario}-seed-{seed}'
            result = run_trial(path, scenario, seed, config, noise)
            trials.append({'scenario': scenario, 'seed': seed, 'directory': path.name, **result})
            grouped.setdefault(scenario, []).append(result)
    summary = {
        'schema_version': 1, 'feedback_level': 'synthetic_rgb_kinematic',
        'official_platform': False, 'official_score': None,
        'recovery_enabled': config.recovery_enabled, 'trial_count': len(trials),
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



PAYLOAD_FIELDS = {
    'trial_started': ({'start_s'}, set()),
    'observation': ({'producer', 'calibration_id', 'frames', 'robot', 'visual_position_m',
                     'stereo_residual_m', 'vision_error'}, {'observation_schema_version', 'sensor_time_s'}),
    'observation_rejected': ({'error'}, set()),
    'action_submitted': ({'phase', 'action_id', 'action', 'grasp_success_inferred'}, {'requested_s'}),
    'action_status': ({'action_id', 'state', 'detail'}, set()),
    'failure_detected': ({'category', 'phase', 'detail'}, set()),
    'recovery_started': ({'category', 'recovery_number'}, set()),
    'grasp_verified': ({'position_m'}, {'tip_m'}),
    'settle_started': ({'anchor_m'}, {'sensor_start_s'}),
    'settle_reset': ({'reason'}, set()),
    'stability_verified': ({'duration_s', 'position_m'}, {'sensor_start_s', 'sensor_end_s'}),
    'trial_stopped': ({'category', 'detail'}, set()),
    'backend_stopped': (set(), set()),
    'stop_request_succeeded': (set(), set()),
    'stop_request_failed': ({'error_type'}, set()),
    'robust_observation': ({'estimate_id', 'sample_indices', 'position_m', 'uncertainty_xyz_m', 'color',
                            'sensor_start_s', 'sensor_end_s', 'mean_pixels', 'effective_samples'}, set()),
    'bundle_reset': ({'reason'}, set()),
    'result': ({'result'}, set()),
}


def validate_payload(kind, payload, version, robust=False):
    required, optional = PAYLOAD_FIELDS[kind]
    if version >= 3 and robust:
        required = required | {'action_submitted': {'estimate_id'}, 'grasp_verified': {'estimate_id'},
                               'settle_started': {'estimate_id'},
                               'stability_verified': {'estimate_id', 'anchor_estimate_id'}}.get(kind, set())
    # Original v2 optional fields are mandatory in v2+, but new timing fields
    # remain optional in historical records. New writers always save both.
    if version >= 2: required = required | optional
    optional = optional | {'observation':{'packet_time_s','received_s'},'trial_stopped':{'deadline'}}.get(kind,set())
    if version >= 4: required = required | {'observation':{'packet_time_s','received_s'},'trial_stopped':{'deadline'}}.get(kind,set())
    if not required <= set(payload) or set(payload)-required-optional:
        raise ValueError('unknown/missing event payload fields')

def _replay(directory, expected_runtime_sha256=None):
    """Rebuild evidence from RGB, feedback and action ordering; no actions execute.

    This is record-consistency validation, not proof of sensor/command authenticity.
    Runtime controller checks are independently reconstructed here.
    """
    root = Path(directory)
    manifest = load_json(root/'manifest.json')
    version = manifest.get('schema_version')
    if type(version) is not int or version not in (1, 2, 3, 4): raise ValueError('unsupported trace schema')
    if (manifest['environment'].get('official_platform') is not False
            or manifest['environment'].get('feedback_level') != 'synthetic_rgb_kinematic'):
        raise ValueError('this offline verifier only supports declared synthetic records')
    if expected_runtime_sha256 is not None:
        if not isinstance(expected_runtime_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_runtime_sha256):
            raise ValueError('invalid trusted runtime fingerprint')
        if manifest['code']['runtime_tree_sha256'] != expected_runtime_sha256:
            raise ValueError('runtime fingerprint differs from trusted expectation')
    config = Config.from_dict(manifest['config'])
    robust = version >= 3 and config.robust_observations
    if config.robust_observations and not robust: raise ValueError('robust evidence requires trace v3')
    cameras = tuple(Camera(**value) for value in manifest['cameras'])
    if version >= 2 and canonical(load_json(root/'config.json')) != canonical(manifest['config']):
        raise ValueError('recorded config differs from manifest')
    manifest_hash = digest((root/'manifest.json').read_bytes())
    gate = ObservationGate(cameras, manifest['producer'], config.max_observation_age_s,
                           config.max_sensor_skew_s, allow_legacy=version == 1, tip_bounds=config.workspace)
    path_verifier = PathVerifier(config,robust,cameras)
    previous = '0'*64; previous_time = -1.0; rejected = 0; final = None; trial_start = None
    samples = []; actions = {}; active = None; stability = None; grasps = []
    bundles = {}; last_bundle = None; held_offset = None; held_color = None; settle_anchor = None
    bundle_breaks = []; carry_errors = []; held_valid = False
    stop_events = []; terminal_events = []; failed_statuses = 0; motion_mismatches = 0
    known_events = {'robust_observation', 'bundle_reset', 'trial_started', 'observation', 'observation_rejected', 'action_submitted',
                    'action_status', 'failure_detected', 'recovery_started', 'grasp_verified',
                    'settle_started', 'settle_reset', 'stability_verified', 'trial_stopped',
                    'backend_stopped', 'stop_request_succeeded', 'stop_request_failed', 'result'}
    for index, line in enumerate((root/'events.jsonl').read_text().splitlines()):
        event = decode_json(line)
        if not isinstance(event, dict): raise ValueError('event must be an object')
        if final is not None: raise ValueError('events after final result')
        required = {'index', 'kind', 'time_s', 'payload', 'previous_sha256', 'sha256'}
        if version >= 2: required.add('manifest_sha256')
        if set(event) != required: raise ValueError('unknown/missing event fields')
        if final is not None: raise ValueError('events after final result')
        recorded_hash = event.pop('sha256')
        if type(event['index']) is not int or event['index'] != index or event['previous_sha256'] != previous:
            raise ValueError('broken event sequence/hash chain')
        if digest(canonical(event).encode()) != recorded_hash: raise ValueError('event content hash mismatch')
        if version >= 2 and event['manifest_sha256'] != manifest_hash: raise ValueError('manifest binding mismatch')
        stamp = event['time_s']
        if not finite(stamp) or stamp < 0 or stamp < previous_time: raise ValueError('invalid event time ordering')
        previous = recorded_hash; previous_time = stamp; payload = event['payload']; kind = event['kind']
        if kind not in known_events or not isinstance(payload, dict): raise ValueError('unknown event kind/payload')
        validate_payload(kind, payload, version, robust)
        if kind == 'trial_started':
            if trial_start is not None or payload.get('start_s') != stamp: raise ValueError('invalid trial start')
            trial_start = stamp
        elif kind == 'observation':
            frames = []
            for frame in payload['frames']:
                path = contained_file(root, frame['file'])
                if digest(path.read_bytes()) != frame['file_sha256']: raise ValueError('RGB artifact hash mismatch')
                width, height, pixels = read_ppm(path)
                if width != frame['width'] or height != frame['height']: raise ValueError('RGB dimension mismatch')
                values = {k: v for k, v in frame.items() if k not in ('file', 'file_sha256')}
                values['rgb'] = pixels; frames.append(values)
            sensor_version = payload.get('observation_schema_version', 1)
            has_receipt = 'packet_time_s' in payload or 'received_s' in payload
            if has_receipt and (set(('packet_time_s','received_s'))-set(payload) or not finite(payload['received_s']) or payload['received_s']!=stamp):
                raise ValueError('incomplete/inconsistent receipt timestamps')
            packet_stamp=payload['packet_time_s'] if has_receipt else stamp
            observation = gate.accept({'schema_version': sensor_version, 'producer': payload['producer'],
                                       'calibration_id': payload['calibration_id'], 'time_s': packet_stamp,
                                       'robot': payload['robot'], 'frames': frames}, stamp)
            try: pixels = observe_single_block_pixels(observation)
            except VisionError: pixels = None
            try: estimate = estimate_single_block(observation, cameras, config.stereo_residual_m, pixels)
            except VisionError:
                estimate = None
                if payload['visual_position_m'] is not None: raise ValueError('recorded estimate not reproducible')
            else:
                if payload['visual_position_m'] is None or distance(estimate.position_m, payload['visual_position_m']) > 1e-9:
                    raise ValueError('visual estimate differs on replay')
            if 'sensor_time_s' in payload and payload['sensor_time_s'] != observation.sensor_time_s:
                raise ValueError('sensor time differs on replay')
            samples.append({'index':len(samples),'time_s': stamp,'packet_time_s':packet_stamp,'receipt_timed':has_receipt, 'sensor_time_s': observation.sensor_time_s,
                            'point': estimate.position_m if estimate else None,
                            'color': pixels.color if pixels else None, 'tip': observation.tip,
                            'pixels': pixels.pixels if pixels else None, 'camera_ids': pixels.camera_ids if pixels else None,
                            'robot_timed': observation.robot_captured_s is not None})
            if robust and active is not None and actions[active]['phase'] in ('target_transit', 'target_descent'):
                if pixels is None or held_offset is None or pixels.color != held_color:
                    carry_errors.append(index)
                else:
                    predicted = add(observation.tip, held_offset)
                    if any(distance(tuple(pixel)+(0,), tuple(next(c for c in cameras if c.name == name).project(predicted))+(0,)) > config.robust_tracking_pixel_px
                           for name, pixel in zip(pixels.camera_ids, pixels.pixels)):
                        carry_errors.append(index)
                if len(samples) > 1 and samples[-1]['sensor_time_s']-samples[-2]['sensor_time_s'] > config.max_observation_gap_s+1e-9:
                    carry_errors.append(index)
        elif kind == 'robust_observation':
            if not robust or type(payload['estimate_id']) is not int or payload['estimate_id'] != index:
                raise ValueError('invalid bundle id/schema')
            value = bundle_replay.reconstruct(payload, samples, cameras, config)
            value['event_index'] = index; bundles[index] = value; last_bundle = index
            latest = next(reversed(actions.values()), None) if actions else None
            if latest and latest['phase'] == 'lift' and latest['state'] == 'complete':
                close = next((r for r in reversed(list(actions.values())) if r['phase'] == 'close' and r['state'] == 'complete'
                              and r['end'] <= latest['start']), None)
                before = bundles.get(close.get('estimate_id')) if close else None
                if (before and value['sample_indices'][0] > latest['after']['index'] and before['color'] == value['color']
                        and abs(before['position_m'][2]-config.table_center_height_m)+before['uncertainty_xyz_m'][2] <= config.table_height_tolerance_m
                        and value['position_m'][2]-before['position_m'][2]-value['uncertainty_xyz_m'][2]-before['uncertainty_xyz_m'][2] >= config.lift_min_m
                        and bundle_replay.held(value, config)):
                    grasps.append({'end': stamp, 'color': value['color']})
                    held_valid = True; held_color = value['color']; held_offset = sub(value['position_m'], value['tip'])
            elif latest and latest['phase'] in ('target_transit', 'target_descent') and latest['state'] == 'complete':
                if value['sample_indices'][0] > latest['after']['index'] and bundle_replay.held(value, config) and value['color'] == held_color:
                    held_valid = True; held_offset = sub(value['position_m'], value['tip'])
            elif latest and latest['phase'] == 'retreat' and bundle_replay.held(value, config):
                # Release-stuck recovery is proved by new pixels near withdrawn hand.
                held_valid = True; held_color = value['color']; held_offset = sub(value['position_m'], value['tip'])
        elif kind == 'bundle_reset':
            if not robust: raise ValueError('bundle reset in raw trace')
            bundle_breaks.append(index)
        elif kind == 'settle_started':
            if robust:
                anchor = bundles.get(payload['estimate_id'])
                if (anchor is None or last_bundle != payload['estimate_id']
                        or not bundle_replay.placement(anchor, config)
                        or distance(payload['anchor_m'], anchor['position_m']) > 1e-9
                        or payload['sensor_start_s'] != anchor['sensor_start_s']):
                    raise ValueError('settle anchor contradicts RGB bundle')
                settle_anchor = payload['estimate_id']
        elif kind == 'settle_reset':
            bundle_breaks.append(index); settle_anchor = None
        elif kind == 'grasp_verified':
            evidence = bundles.get(payload.get('estimate_id')) if robust else samples[-1] if samples else None
            point = evidence['position_m'] if robust and evidence else evidence['point'] if evidence else None
            if point is None or distance(payload['position_m'], point) > 1e-9:
                raise ValueError('grasp label contradicts reconstructed RGB')
            if 'tip_m' in payload and distance(payload['tip_m'], samples[-1]['tip']) > 1e-9:
                raise ValueError('grasp label contradicts measured hand')
        elif kind == 'observation_rejected': rejected += 1
        elif kind == 'trial_stopped': terminal_events.append(payload.get('category'))
        elif kind == 'action_submitted':
            action_id = payload['action_id']; action = payload['action']
            if PHASE_ACTIONS.get(payload['phase']) != action.get('kind'):
                raise ValueError('unknown or mismatched action phase')
            if not isinstance(action_id, str) or not action_id.strip() or action_id in actions or active is not None:
                raise ValueError('invalid/reused/overlapping action id')
            requested = payload.get('requested_s', stamp)
            if not finite(requested) or not 0 <= requested <= stamp:
                raise ValueError('invalid action request time')
            if not samples or not 0 <= requested-samples[-1]['sensor_time_s'] <= config.max_observation_age_s:
                raise ValueError('action without fresh feedback')
            if set(action) != {'kind', 'target_m'} or action['kind'] not in ('move', 'grasp', 'release'):
                raise ValueError('invalid action contract')
            if action['kind'] == 'move':
                target = action['target_m']; tip = samples[-1]['tip']
                if not isinstance(target, (tuple, list)) or len(target) != 3 or not all(finite(v) for v in target):
                    raise ValueError('invalid action target')
                if any(not config.workspace[2*i] <= target[i] <= config.workspace[2*i+1] for i in range(3)):
                    raise ValueError('action target outside workspace')
                lateral = ((target[0]-tip[0])**2+(target[1]-tip[1])**2)**0.5
                if lateral > 1e-6 and min(target[2], tip[2]) < config.transit_min_height_m:
                    raise ValueError('low horizontal action in replay')
            elif action['target_m'] is not None: raise ValueError('non-motion action has target')
            if robust:
                reference = bundles.get(payload['estimate_id'])
                if reference is None or payload['estimate_id'] != last_bundle:
                    raise ValueError('action missing latest RGB bundle reference')
                if payload['phase'] == 'close' and (reference['sensor_end_s'] != samples[-1]['sensor_time_s'] or not bundle_replay.held(reference, config)):
                    raise ValueError('close lacks fresh conservative visual target')
                if payload['phase'] in ('target_transit', 'target_descent', 'release') and not held_valid:
                    raise ValueError('carrying/descent/release lacks reconstructed held proof')
                if payload['phase'] == 'release' and distance(samples[-1]['tip'], config.target) > config.motion_completion_m:
                    raise ValueError('release away from measured target')
            actions[action_id] = {'estimate_id': payload.get('estimate_id'), 'action': action, 'phase': payload['phase'], 'start': requested,
                                  'state': 'pending', 'before': samples[-1]}
            active = action_id
        elif kind == 'action_status':
            action_id = payload['action_id']; state = payload['state']
            if action_id not in actions or active != action_id or state not in ('pending', 'complete', 'failed'):
                raise ValueError('status without active action / invalid lifecycle')
            record = actions[action_id]
            if state == 'complete':
                if stamp-record['start'] >= config.action_timeout_s: raise ValueError('late completion bypasses action deadline')
                if not samples or samples[-1]['sensor_time_s'] < stamp-config.max_observation_age_s-1e-9:
                    raise ValueError('completion without fresh measured feedback')
                record['motion_verified'] = True
                if record['action']['kind'] == 'move':
                    record['motion_verified'] = distance(samples[-1]['tip'], record['action']['target_m']) <= config.motion_completion_m
                    if not record['motion_verified']: motion_mismatches += 1
                record['after'] = samples[-1]; record['end'] = stamp
                if robust and record['phase'] in ('target_transit', 'target_descent', 'release'):
                    held_valid = False
                if not robust and record['phase'] == 'lift' and record['action']['kind'] == 'move':
                    close = next((r for r in reversed(list(actions.values()))
                                  if r['phase'] == 'close' and r['action']['kind'] == 'grasp'
                                  and r['state'] == 'complete' and r['end'] <= record['start']), None)
                    pre = close['before'] if close else None; post = record['after']
                    if (pre and pre['point'] is not None and post['point'] is not None and pre['color'] == post['color']
                            and abs(pre['point'][2]-config.table_center_height_m) <= config.table_height_tolerance_m
                            and post['point'][2]-pre['point'][2] >= config.lift_min_m
                            and distance(post['point'], post['tip']) <= config.grasp_distance_m):
                        grasps.append({'end': stamp, 'color': post['color']})
            if state == 'failed': failed_statuses += 1
            record['state'] = state
            if state in ('complete', 'failed'): active = None
        elif kind in ('backend_stopped', 'stop_request_succeeded', 'stop_request_failed'):
            stop_events.append(kind)
            if active is not None:
                actions[active]['state'] = 'stopped'; active = None
        elif kind == 'stability_verified':
            if robust and (settle_anchor != payload['anchor_estimate_id'] or last_bundle != payload['estimate_id']):
                raise ValueError('stability references invalid/reset bundle window')
            stability = event
        elif kind == 'result': final = payload['result']
        path_verifier.accept(event,samples,bundles,actions)
        if kind=='trial_stopped': path_verifier.terminal_payload={**payload,'_time_s':stamp}
        # grasp_verified labels are deliberately ignored as proof.
    if final is None: raise ValueError('incomplete trace: missing result')
    if final != load_json(root/'result.json'): raise ValueError('result file differs from event log')
    if (type(final.get('action_count')) is not int or final['action_count'] < 0
            or type(final.get('recoveries')) is not int or final['recoveries'] < 0
            or not finite(final.get('stability_verified_s')) or final['stability_verified_s'] < 0
            or final.get('physical_stop_verified') is not False):
        raise ValueError('invalid counts/stability/physical-stop claim')
    if type(final.get('success')) is not bool or not finite(final.get('elapsed_s')) or final['elapsed_s'] < 0:
        raise ValueError('invalid result values')
    if final.get('official_platform') is not False or final.get('official_score') is not None:
        raise ValueError('offline record cannot claim an official score')
    if final.get('final_visual_position_m') is not None:
        if not samples or samples[-1]['point'] is None or distance(final['final_visual_position_m'], samples[-1]['point']) > 1e-9:
            raise ValueError('final visual result contradicts RGB evidence')
    elif samples and samples[-1]['point'] is not None:
        raise ValueError('final visual result missing despite visible final sample')
    if final.get('action_count') != len(actions): raise ValueError('action count differs from evidence')
    if len(stop_events) != 1: raise ValueError('missing/duplicate stop evidence')
    if final.get('stop_request_succeeded') is True and stop_events[0] not in ('backend_stopped', 'stop_request_succeeded'):
        raise ValueError('stop result contradicts stop event')
    if not final['success']:
        terminal = final.get('terminal_failure')
        if terminal == 'stop_transport_failure':
            if stop_events[0] != 'stop_request_failed': raise ValueError('missing failed stop evidence')
        elif terminal not in terminal_events: raise ValueError('missing matching terminal failure evidence')
        if terminal == 'motion_not_verified' and not motion_mismatches:
            raise ValueError('motion rejection has no measured arrival mismatch')
        if terminal == 'actuator_failure' and not failed_statuses:
            raise ValueError('actuator failure has no failed action evidence')
    if robust and final.get('final_robust_estimate_id') != last_bundle: raise ValueError('final bundle reference differs')
    if version >= 2:
        if trial_start is None or abs(previous_time-trial_start-final['elapsed_s']) > 1e-6:
            raise ValueError('result duration differs from event timeline')
        if final.get('trace_complete') is not True: raise ValueError('trace marked incomplete')
    if final['success']:
        first_visible = next(iter(bundles.values()), None) if robust else next((s for s in samples if s['point'] is not None), None)
        first_point = first_visible.get('position_m', first_visible.get('point')) if first_visible else None
        if (final.get('started_in_configured_supply') is not True or first_visible is None
                or not inside_xy(first_point, config.supply_xy)):
            raise ValueError('success has no initial supply-zone evidence')
        if not grasps or stability is None: raise ValueError('successful record lacks grasp/stability evidence reconstructed from sensors/actions')
        if final.get('terminal_failure') is not None or final.get('stop_request_succeeded') is not True:
            raise ValueError('successful record has failure or missing stop request')
        if any(r['state'] != 'complete' or not r.get('motion_verified', True) for r in actions.values()): raise ValueError('success has incomplete/failed actions')
        if final['elapsed_s'] >= config.trial_timeout_s: raise ValueError('success exceeds trial deadline')
        duration = stability['payload']['duration_s']; end = stability['payload'].get('sensor_end_s', stability['time_s'])
        start = stability['payload'].get('sensor_start_s', end-duration)
        if (not finite(duration) or duration < config.settle_s or abs(end-start-duration) > 1e-6
                or final['stability_verified_s'] != duration):
            raise ValueError('insufficient/inconsistent stability duration')
        release = [r for r in actions.values() if r['action']['kind'] == 'release' and r.get('after',{}).get('sensor_time_s',float('inf')) <= start]
        if not release or not any(g['end'] < release[-1]['start'] for g in grasps):
            raise ValueError('no release action after reconstructed grasp')
        window = [s for s in samples if start-1e-6 <= s['sensor_time_s'] <= end+1e-6]
        if not window or window[0]['sensor_time_s'] > start+1e-6 or window[-1]['sensor_time_s'] < end-1e-6:
            raise ValueError('stability evidence window incomplete')
        if robust:
            if carry_errors: raise ValueError('success ignores lost carried RGB evidence')
            anchor_bundle = bundles[stability['payload']['anchor_estimate_id']]
            end_bundle = bundles[stability['payload']['estimate_id']]
            if start != anchor_bundle['sensor_start_s'] or end != end_bundle['sensor_end_s']:
                raise ValueError('stability time differs from bundle capture times')
            valid_bundles = [b for b in bundles.values() if anchor_bundle['event_index'] <= b['event_index'] <= end_bundle['event_index']]
            if any(anchor_bundle['event_index'] < i < stability['index'] for i in bundle_breaks):
                raise ValueError('stability window contains a reset')
            covered = set()
            for b in valid_bundles:
                if not bundle_replay.placement(b, config) or not bundle_replay.stable(anchor_bundle, b, config):
                    raise ValueError('bundle violates conservative placement/stability bounds')
                covered.update(b['sample_indices'])
            needed = {i for i, s in enumerate(samples) if start-1e-6 <= s['sensor_time_s'] <= end+1e-6}
            if covered != needed: raise ValueError('stability bundles omit primary observations')
            if distance(stability['payload']['position_m'], end_bundle['position_m']) > 1e-9:
                raise ValueError('stability label contradicts RGB bundle')
        anchor = window[0]['point']
        for i, sample in enumerate(window):
            point = sample['point']; tip = sample['tip']
            if not robust and (point is None or not inside_xy(point, config.build_xy)
                    or abs(point[2]-config.table_center_height_m) > config.table_height_tolerance_m
                    or distance(point, config.target) > config.goal_tolerance_m
                    or distance(point, anchor) > config.stability_m
                    or distance(point, tip) < config.hand_separation_m):
                raise ValueError('stability sample violates placement/height/hand-separation criterion')
            if i and sample['sensor_time_s']-window[i-1]['sensor_time_s'] > config.max_observation_gap_s+1e-9:
                raise ValueError('gap in continuous visual stability evidence')
    failure_proof, failure_limits = path_verifier.failure_cause(final,actions,rejected,trial_start,version)
    return {'feedback_level': manifest['environment']['feedback_level'], 'official_platform': False,
            'bundles_recomputed': len(bundles), 'robust_observations': robust,
            'observations_recomputed': len(samples), 'rejected_observation_events': rejected,
            'actions_recorded': len(actions), 'grasp_evidence_reconstructed': len(grasps),
            'trace_verified': True, 'visual_evidence_recomputed': True,
            'control_record_consistent': failure_proof, 'failure_reason_reconstructed':failure_proof if not final['success'] else None,
            'failure_reason_limitations':failure_limits, 'legal_path_reconstructed':True,
            'receipt_timestamps_recorded': bool(samples) and all(s['receipt_timed'] for s in samples), 'sensor_authenticity_verified': False,
            'physical_action_execution_verified': False,
            'runtime_identity_checked': expected_runtime_sha256 is not None,
            'manifest_bound': version >= 2,
            'robot_timestamps_checked': bool(samples) and all(s['robot_timed'] for s in samples),
            'evidence_contract': f'v{version}', 'recorded_success': final['success']}


def replay(directory, expected_runtime_sha256=None):
    try:
        return _replay(directory, expected_runtime_sha256)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('malformed trace metadata/payload') from error
