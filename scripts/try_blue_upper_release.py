"""Continue the actual held blue through support conversion and upper placement."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import try_contact_guided_release as seven

g = seven.g
third = seven.release
SUPPORTS = ('neighbor_5', 'neighbor_1')


def prepare_near_support(lowering=.045, support_first=False, horizontal_strip=False, guide_clear_s=8.):
    """Lower the existing cradle before removing its lateral guide and support."""
    if not np.isfinite(lowering) or not 0 <= lowering <= .060:
        raise ValueError('Near-support lowering must be between zero and 60 mm')
    if not np.isfinite(guide_clear_s) or not 5 <= guide_clear_s <= 8:
        raise ValueError('Guide clearance time must be between five and eight seconds')
    if guide_clear_s != 8 and not (support_first and horizontal_strip):
        raise ValueError('Earlier guide clearance requires horizontal support-first stripping')
    g.restore()
    starts = {side: (g.d.mocap_pos[i].copy(), g.d.mocap_quat[i].copy())
              for side, i in [('left', g.p.t.LM), ('right', g.p.t.RM)]}
    controls = {g.m.actuator(i).name: float(g.d.ctrl[i]) for i in range(g.m.nu)}
    pair = np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS], axis=0)
    if horizontal_strip and not support_first:
        raise ValueError('Horizontal stripping requires the left guide to remain')
    withdrawal = np.array([-.08, -.08, .04])
    if horizontal_strip:
        normals = []
        for c in g.d.contact:
            names = [g.m.geom(int(i)).name for i in c.geom]
            if g.TARGET in names and any(n.startswith('lh_') for n in names):
                normals.append(c.frame[:3] * (1 if names[0] == g.TARGET else -1))
        if not normals:
            raise ValueError('Horizontal stripping requires actual left guide contact')
        normal = np.mean(normals, axis=0)
        normal[2] = 0
        if np.linalg.norm(normal) < .1:
            raise ValueError('Left guide has no reliable horizontal normal')
        withdrawal = -.08 * normal / np.linalg.norm(normal)
    g.DURATION = 13.
    def command(t):
        lower = np.array([0., 0., -lowering]) * np.clip(t / 4, 0, 1)
        poses = {}
        left_start, right_start = (guide_clear_s, 5.) if support_first else (5., 7.)
        for side, (pos, quat) in starts.items():
            if side == 'left':
                shift = np.array([.08, .08, .08]) * np.clip((t - left_start) / 2, 0, 1)
            else:
                shift = withdrawal * np.clip((t - right_start) / 3, 0, 1)
            poses[side] = (pos + lower + shift, quat)
        if t < 4:
            phase = 'lower_existing_cradle'
        elif t < 5:
            phase = 'settle_near_support'
        elif t >= 10:
            phase = 'released_hold'
        elif support_first:
            phase = 'clear_right_support' if t < guide_clear_s else 'clear_left_guide'
        else:
            phase = 'clear_left_guide' if t < 7 else 'clear_right_support'
        return poses, controls, phase
    return command, dict(source=str(g.SOURCE), target=g.TARGET, supports=g.SUPPORTS,
        pair_center_m=pair.tolist(), lowering_m=lowering, support_first=support_first,
        horizontal_strip=horizontal_strip, right_withdrawal_m=withdrawal.tolist(), duration_s=g.DURATION, hold_start_s=10.,
        guide_clear_s=guide_clear_s if support_first else 5.,
        finger_controls_unchanged=True, source_sequence_continuity_verified=True,
        online_object_feedback=False, native_verified=False, whole_arm_collision_checked=False)


def configure(source):
    record = json.loads((source / 'result.json').read_text())
    summary = record['summary']
    target = summary['target']
    if target not in ('neighbor_0', 'neighbor_7'):
        raise ValueError('Expected the actual held blue')
    assessment = seven.assess(source, held_payload=target)
    up = summary.get('held_minimum_up_force_n', 0)
    if (not any(summary.get(k, False) for k in ('local_loaded_transfer_verified', 'local_shared_transport_hold_verified', 'local_blue_shared_hold_verified'))
            or not assessment['local_seven_brick_final_structure_verified'] or not summary.get('held_only_hand_contacts', False)
            or summary['held_duration_s'] < 2.98 or summary['held_drift_mm'] > 2 or summary['held_angular_drift_degrees'] > 2
            or up < .8 * g.m.body(target).mass * 9.81 or max(summary['other_bricks_max_shift_mm'].values()) > 2):
        raise ValueError('Source is not a stable shared hold preserving seven bricks')
    state = np.load(source / 'integration-state.npz', allow_pickle=False)
    check = copy.deepcopy(json.loads((source / 'precheck.json').read_text()))
    for side, stages in check['stages'].items():
        check['stages'][side] = [stage if 'result' in stage else {'result': stage} for stage in stages]
    def restore():
        g.mj.mj_setState(g.m, g.d, state['state'], int(state['spec']))
        g.mj.mj_forward(g.m, g.d)
    g.SOURCE, g.RECORD, g.STATE, g.CHECK = source, record, state, check
    supports = tuple(json.loads((source / 'plan.json').read_text())['supports'])
    if supports not in (SUPPORTS, ('neighbor_1', 'neighbor_3')):
        raise ValueError('Expected an adjacent yellow support pair')
    g.TARGET, g.SUPPORTS, g.CONTINUOUS_SOURCE = target, list(supports), True
    g.restore = restore
    restore()


def evaluate(out, released=False):
    record = json.loads((out / 'result.json').read_text())
    summary = record['summary']
    plan = json.loads((out / 'plan.json').read_text())
    held = [row for row in record['rows'] if row['elapsed_s'] >= plan['hold_start_s'] - 1e-8]
    hands = bool(held and all(row['contacts'] and all(any(n.startswith(('lh_', 'rh_')) for n in c['pair'])
                        for c in row['contacts']) for row in held))
    up = min((sum(c['force_xyz_n'][2] for c in row['contacts']
                  if any(n.startswith(('lh_', 'rh_')) for n in c['pair'])) for row in held), default=0.)
    assessment = seven.assess(out, placed_payload=g.TARGET) if released else seven.assess(out, held_payload=g.TARGET)
    summary.update(diagnostic_only=True, held_only_hand_contacts=hands, held_minimum_up_force_n=up,
                   local_seven_brick_final_structure_verified=assessment['local_seven_brick_final_structure_verified'])
    summary['local_blue_shared_hold_verified'] = bool(not released and summary['local_seven_brick_final_structure_verified']
        and hands and up >= .8 * g.m.body(g.TARGET).mass * 9.81 and summary.get('held_duration_s', 0) >= 2.98
        and summary.get('held_drift_mm', 1e6) <= 2 and summary.get('held_angular_drift_degrees', 1e6) <= 2)
    summary['local_blue_level_hold_verified'] = bool(summary['local_blue_shared_hold_verified']
        and summary.get('final_cube_face_tilt_degrees', 1e6) <= 2)
    summary['local_blue_flat_hold_verified'] = bool(summary['local_blue_level_hold_verified']
        and abs(summary.get('final_cube_yaw_mod90_degrees', 1e6)) <= 10)
    summary['local_eight_brick_final_structure_verified'] = assessment.get('local_eight_brick_final_structure_verified', False)
    (out / 'result.json').write_text(json.dumps(record, allow_nan=False))
    (out / 'structure-assessment.json').write_text(json.dumps(assessment, allow_nan=False))
    return summary


def calibrate_release(command, plan, source):
    """One offline rigid landing correction from a genuinely released trial."""
    record = json.loads((source / 'result.json').read_text())
    prior_plan = json.loads((source / 'plan.json').read_text())
    s = record['summary']
    if (s['target'] != g.TARGET or not s.get('local_released_bridge_verified', False)
            or not s.get('held_only_base_contacts', False)
            or prior_plan['source'] != plan['source'] or prior_plan['supports'] != plan['supports']
            or any(prior_plan.get(k) != plan.get(k) for k in ('lowering_m', 'support_first', 'horizontal_strip', 'guide_clear_s'))
            or prior_plan.get('calibration_source')
            or not seven.assess(source, placed_payload=g.TARGET)['local_seven_brick_final_structure_verified']):
        raise ValueError('Calibration requires a stable actual release from this same held source')
    g.restore()
    center = g.d.xpos[g.m.body(g.TARGET).id].copy()
    landing = np.array(record['rows'][-1]['position_m'])
    pair = np.array(plan['pair_center_m'])
    yaw = -s['final_cube_yaw_mod90_degrees']
    turn = Rotation.from_euler('z', yaw, degrees=True)
    shift = pair - (turn.apply(landing - center) + center)
    shift[2] = 0
    if np.linalg.norm(shift) > .03 or abs(yaw) > 20:
        raise ValueError('Measured landing correction exceeds this local comparison')
    def calibrated(t):
        poses, ctrl, phase = command(max(t - 3, 0))
        fraction = np.clip(t / 3, 0, 1)
        rotation = Rotation.from_euler('z', yaw * fraction, degrees=True)
        poses = {side: (rotation.apply(pos - center) + center + shift * fraction,
                       (rotation * Rotation.from_quat(quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]])
                 for side, (pos, quat) in poses.items()}
        return poses, ctrl, 'offline_landing_correction' if t < 3 else phase
    g.DURATION += 3
    plan.update(calibration_source=str(source), landing_correction_m=shift.tolist(),
                landing_yaw_correction_degrees=yaw, duration_s=g.DURATION, hold_start_s=plan['hold_start_s'] + 3)
    return calibrated, plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/receiver-support-118/blue-transfer'))
    parser.add_argument('--mode', choices=('unfold', 'level', 'align', 'level-at-pair', 'right-level', 'near-support'), required=True)
    parser.add_argument('--lower-mm', type=float, default=45.)
    parser.add_argument('--support-first', action='store_true')
    parser.add_argument('--horizontal-strip', action='store_true')
    parser.add_argument('--guide-clear-s', type=float, default=8.)
    parser.add_argument('--calibration-source', type=Path)
    parser.add_argument('--pair', choices=('south', 'north'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if (args.support_first or args.horizontal_strip or args.lower_mm != 45. or args.guide_clear_s != 8.) and args.mode != 'near-support':
        parser.error('Withdrawal options and --lower-mm require --mode near-support')
    if args.calibration_source and args.mode != 'near-support':
        parser.error('--calibration-source requires --mode near-support')
    configure(args.source)
    if args.pair:
        if args.mode != 'level-at-pair':
            parser.error('--pair requires --mode level-at-pair')
        g.SUPPORTS = list(SUPPORTS if args.pair == 'south' else ('neighbor_1', 'neighbor_3'))
    if args.mode == 'unfold':
        command, plan = third.prepare_unfold_receiver(False, 2.)
    elif args.mode == 'right-level':
        command, plan = third.prepare_flatten_right(False)
    elif args.mode == 'near-support':
        command, plan = prepare_near_support(args.lower_mm / 1000, args.support_first, args.horizontal_strip, args.guide_clear_s)
    elif args.mode in ('level', 'level-at-pair'):
        command, plan = third.r.prepare()
        if args.mode == 'level-at-pair':
            base = command
            center = g.d.xpos[g.m.body(g.TARGET).id].copy()
            pair = np.mean([g.d.xpos[g.m.body(n).id] for n in g.SUPPORTS], axis=0)
            shift = np.r_[pair[:2] - center[:2], 0.]
            def command(t):
                poses, controls, phase = base(max(t - 3, 0))
                poses = {side: (pos + shift * np.clip(t / 3, 0, 1), quat) for side, (pos, quat) in poses.items()}
                return poses, controls, 'translate_to_pair' if t < 3 else phase
            g.DURATION += 3
            plan.update(common_translation_m=shift.tolist(), duration_s=g.DURATION, hold_start_s=plan['hold_start_s'] + 3)
    else:
        if not g.RECORD['summary'].get('local_blue_level_hold_verified', False):
            raise ValueError('Yaw alignment requires a verified level blue hold')
        command, plan = third.prepare_yaw(-g.RECORD['summary']['final_cube_yaw_mod90_degrees'])
    if args.calibration_source:
        command, plan = calibrate_release(command, plan, args.calibration_source)
    plan.update(mode=args.mode, diagnostic_only=True, payload_floor_m=.090)
    args.out.mkdir(parents=True, exist_ok=False)
    check = g.precheck(command)
    (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
    (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
    passed = check['passed']
    brief = dict(precheck_passed=passed, geometry_failures=len(check['geometry_failures']), stages={s: len(v) for s, v in check['stages'].items()})
    if passed and args.run:
        g.run(command, plan, args.out, protected_payload='neighbor_5',
              additional_protected_payloads=('neighbor_1', 'neighbor_3', *seven.release.feed.BASES), payload_floor_m=.090)
        brief['dynamics'] = evaluate(args.out, released=args.mode == 'near-support')
        flag = 'local_blue_shared_hold_verified' if args.mode == 'unfold' else 'local_blue_level_hold_verified' if args.mode in ('level', 'level-at-pair') else 'local_blue_flat_hold_verified'
        if args.mode == 'near-support':
            flag = 'local_eight_brick_final_structure_verified'
        passed = brief['dynamics'][flag]
    print(json.dumps(brief, indent=2))
    raise SystemExit(0 if passed else 1)
