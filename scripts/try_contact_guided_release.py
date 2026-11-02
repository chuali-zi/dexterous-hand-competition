"""Bounded local release using neighboring bricks as transient guides.

Keep the original strict result unchanged. Assess the final seven-brick
structure separately; this is not official scoring or native validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import try_third_yellow_release as release

g = release.g
GUIDES = ('neighbor_5', 'neighbor_3')


def assess(out, held_payload=None, placed_payload=None):
    record = json.loads((out / 'result.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    check = json.loads((out / 'precheck.json').read_text())
    summary = record['summary']
    if placed_payload is not None and (held_payload is not None or placed_payload not in ('neighbor_0', 'neighbor_7')
                                      or summary['target'] != placed_payload
                                      or tuple(plan.get('supports', ())) not in (('neighbor_5', 'neighbor_1'), ('neighbor_1', 'neighbor_3'))):
        raise ValueError('Placed blue requires its adjacent yellow support pair and hands withdrawn')
    if held_payload is not None and (held_payload not in ('neighbor_0', 'neighbor_7') or summary['target'] != held_payload):
        raise ValueError('Only the explicitly active blue payload may remain in hand')
    held = [row for row in record['rows'] if row['elapsed_s'] >= plan['hold_start_s'] - 1e-8]
    report = dict(local_seven_brick_final_structure_verified=False,
                  strict_release_verified=summary.get('local_released_bridge_verified', False),
                  diagnostic_only=True, native_verified=False, full_pyramid_verified=False,
                  whole_arm_collision_checked=False,
                  scope='Final three-second local structure after transient neighboring-brick contact; original strict result preserved')
    if placed_payload is not None:
        report.update(local_eight_brick_final_structure_verified=False, placed_blue=placed_payload,
                      scope='Final local four-red/three-yellow/one-blue structure with all hands clear; not official scoring')
    if len(held) < 2:
        report['reason'] = 'No complete final hold'
        return report
    reference = np.load(release.feed.SECOND_REFERENCE / 'integration-state.npz', allow_pickle=False)
    ref = g.mj.MjData(g.m)
    g.mj.mj_setState(g.m, ref, reference['state'], int(reference['spec']))
    g.mj.mj_forward(g.m, ref)
    structure = release.feed.current_structure(held, ref, True)
    times = np.array([row['elapsed_s'] for row in held])
    complete = bool(times[-1] - times[0] >= 2.98 and np.all(np.diff(times) > 0)
                    and np.max(np.diff(times)) <= .022)
    pairs = {'neighbor_5': ('neighbor_9', 'neighbor_6'),
             'neighbor_1': ('neighbor_6', 'neighbor_4'),
             'neighbor_3': ('neighbor_4', 'neighbor_2')}
    if placed_payload is not None:
        pairs[placed_payload] = tuple(plan['supports'])
    metrics = {}
    for name, supports in pairs.items():
        address = g.m.jnt_qposadr[g.m.joint(name + '_free').id]
        xyz = np.array([row['qpos'][address:address + 3] for row in held])
        quats = np.array([row['qpos'][address + 3:address + 7] for row in held])
        rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
        matrices = rotations.as_matrix()
        face = np.degrees(np.arccos(np.clip(np.max(np.abs(matrices[:, 2, :]), axis=1), 0, 1)))
        axes = np.take_along_axis(matrices, np.argmax(np.abs(matrices[:, 0, :]), axis=1)[:, None, None], axis=2)[:, :, 0]
        yaw = (np.degrees(np.arctan2(axes[:, 1], axes[:, 0])) + 45) % 90 - 45
        addresses = [g.m.jnt_qposadr[g.m.joint(n + '_free').id] for n in supports]
        centers = np.array([np.mean([row['qpos'][a:a + 3] for a in addresses], axis=0) for row in held])
        contacts = [row['contacts'] if name == summary['target'] else row['protected_payload_contacts']
                    if name == 'neighbor_5' else row['additional_protected_payload_contacts'][name] for row in held]
        permitted = (name, *supports)
        if placed_payload is not None and name in pairs[placed_payload]:
            permitted += (placed_payload,)
        only = all(cs and all(all(n in permitted for n in c['pair']) for c in cs) for cs in contacts)
        forces = {n: min(sum(c['force_xyz_n'][2] for c in cs if n in c['pair']) for cs in contacts) for n in supports}
        drift = float(np.linalg.norm(xyz - xyz[0], axis=1).max() * 1000)
        angular = float(np.degrees((rotations[0].inv() * rotations).magnitude().max()))
        error = float(np.linalg.norm((xyz - centers)[:, :2], axis=1).max() * 1000)
        height_error = float(np.max(np.abs(xyz[:, 2] - (g.a.g.FLOOR + (.125 if name == placed_payload else .075)))) * 1000)
        metrics[name] = dict(drift_mm=drift, angular_drift_degrees=angular,
            maximum_actual_pair_center_error_mm=error, maximum_face_tilt_degrees=float(face.max()),
            maximum_abs_yaw_degrees=float(np.abs(yaw).max()), maximum_height_error_mm=height_error,
            only_intended_support_contacts=only, minimum_support_up_force_n=forces,
            passed=bool(only and min(forces.values()) >= .1 and sum(forces.values()) >= .8 * g.m.body(name).mass * 9.81
                        and drift <= 2 and angular <= 2 and error <= 5 and face.max() <= 2 and np.abs(yaw).max() <= 10 and height_error <= 5))
    hand_contacts = []
    held_payload_contacts = []
    probe = g.mj.MjData(g.m)
    for row in held:
        probe.qpos[:] = row['qpos']
        probe.qvel[:] = row['qvel']
        g.mj.mj_forward(g.m, probe)
        for contact in probe.contact:
            names = [g.m.geom(int(i)).name for i in contact.geom]
            if any(n.startswith(('lh_', 'rh_')) for n in names):
                entry = dict(t=row['elapsed_s'], pair=names)
                if held_payload is not None and held_payload in names:
                    held_payload_contacts.append(entry)
                else:
                    hand_contacts.append(entry)
    state = np.load(out / 'integration-state.npz', allow_pickle=False)
    restored = g.mj.MjData(g.m)
    version_ok = str(state['mujoco_version']) == g.mj.__version__
    g.mj.mj_setState(g.m, restored, state['state'], int(state['spec']))
    checkpoint_ok = bool(version_ok and np.allclose(restored.qpos, held[-1]['qpos'], rtol=0, atol=1e-12)
                         and np.allclose(restored.qvel, held[-1]['qvel'], rtol=0, atol=1e-12)
                         and summary['checkpoint_next_step_maximum_difference'] <= 1e-10)
    bounds_ok = all(v <= (5 if n in GUIDES else 2) for n, v in summary['other_bricks_max_shift_mm'].items())
    report.update(held_duration_s=float(times[-1] - times[0]), complete_hold=complete,
                  six_brick_final_structure=structure, upper_brick_metrics=metrics,
                  final_hand_contacts=hand_contacts, checkpoint_verified=checkpoint_ok,
                  transient_neighbor_bounds_respected=bounds_ok,
                  maximum_neighbor_shift_mm=summary['other_bricks_max_shift_mm'])
    if held_payload is not None:
        report.update(permitted_held_payload=held_payload, held_payload_hand_contacts=held_payload_contacts,
                      scope='Seven placed bricks remain supported and clear of hands; explicitly named blue payload may be held separately')
    report['local_seven_brick_final_structure_verified'] = bool(
        complete and structure['passed'] and all(v['passed'] for n, v in metrics.items() if n != placed_payload)
        and not hand_contacts and checkpoint_ok and bounds_ok and check['passed']
        and not summary['abort'] and not summary['unintended_contacts']
        and summary['source_sequence_continuity_verified'])
    if placed_payload is not None:
        report['local_eight_brick_final_structure_verified'] = bool(
            report['local_seven_brick_final_structure_verified'] and metrics[placed_payload]['passed'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/guided-unload-115/align-level-support'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--neighbor-shift-mm', type=int, choices=(2, 5), default=2)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--assess-only', action='store_true')
    args = parser.parse_args()
    if args.run and args.assess_only:
        parser.error('--run and --assess-only are mutually exclusive')
    release.r.configure(args.source, target=release.feed.TARGET)
    if not args.assess_only:
        command, plan = release.prepare_strip_support(away_from_stop=True)
        plan.update(allowed_transient_payload_contacts=list(GUIDES), diagnostic_only=True,
                    transient_neighbor_shift_limit_mm=args.neighbor_shift_mm)
        args.out.mkdir(parents=True, exist_ok=False)
        check = g.precheck(command)
        (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
        (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
        if not check['passed'] or not args.run:
            print(json.dumps(dict(precheck_passed=check['passed'], dynamics_run=False)))
            raise SystemExit(0 if check['passed'] else 1)
        g.run(command, plan, args.out, protected_payload=GUIDES[0],
              additional_protected_payloads=(GUIDES[1], *release.feed.BASES),
              allowed_payload_contacts=GUIDES, allowed_payload_shift_m=args.neighbor_shift_mm / 1000)
    report = assess(args.out)
    (args.out / 'contact-guided-assessment.json').write_text(json.dumps(report, allow_nan=False))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['local_seven_brick_final_structure_verified'] else 1)
