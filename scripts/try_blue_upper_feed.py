"""Feed a blue brick from the actual local seven-brick final state.

Fixed table pushes, full integration-state continuation, original physics.
The seven-brick source remains a local diagnostic, not native validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import try_contact_guided_release as seven
import try_receiver_return as returner

g = seven.g
first = seven.release.feed.first
INITIAL = Path('runs/narrow-support-116/contact-guided-diagnostic-5mm')


def shallow_prediction_permitted(predicted, target, enabled=False):
    return bool(enabled and predicted and all(
        set(c['pair']) == {target, 'target'} and 0 <= c['depth_mm'] <= .5 for c in predicted))


def configure(source):
    record = json.loads((source / 'result.json').read_text())
    summary = record['summary']
    if source.resolve() != INITIAL.resolve() and not summary.get('local_blue_table_feed_verified', False):
        raise ValueError('Source is not a verified blue table feed')
    assessment = seven.assess(source)
    if not assessment['local_seven_brick_final_structure_verified']:
        raise ValueError('Source does not preserve the actual final seven-brick structure')
    state = np.load(source / 'integration-state.npz', allow_pickle=False)
    check = json.loads((source / 'precheck.json').read_text())
    def restore():
        g.mj.mj_setState(g.m, g.d, state['state'], int(state['spec']))
        g.mj.mj_forward(g.m, g.d)
    g.restore = restore
    g.SOURCE, g.RECORD, g.STATE, g.CHECK = source, record, state, check
    restore()
    return seven.release.feed.seeds()


def prepare(source, target, axis, destination, single_finger=False):
    g.restore()
    initial = {g.m.actuator(i).name: float(g.d.ctrl[i]) for i in range(g.m.nu)}
    targets = {n: v for n, v in g.p.t.CANDIDATE['hand_targets'].items() if n.startswith('lh_')}
    if single_finger:
        targets.update(lh_middle_mcp_pitch=1.45, lh_middle_dip=1.45)
    position = g.d.xpos[g.m.body(target).id].copy()
    goal = position[:2].copy()
    goal[0 if axis == 'west' else 1] = destination
    command, plan = first.prepare(axis, source, g.restore, target_name=target,
        goal_xy_m=goal.tolist(), hand_targets=targets, vertical_approach=axis == 'south',
        adapt_lateral_face=axis == 'west', contact_fingers=('index',) if single_finger else ('index', 'middle'))
    def fingers(t):
        return {n: initial[n] + (v - initial[n]) * np.clip(t / 2, 0, 1) for n, v in targets.items()}
    plan.update(source_sequence_continuity_verified=True, diagnostic_only=True,
                seven_brick_source=str(INITIAL), single_finger=single_finger)
    return command, fingers, plan


def payload_precheck(plan):
    g.restore()
    target = plan['target']
    address = g.m.jnt_qposadr[g.m.joint(target + '_free').id]
    start = g.d.qpos[address:address + 3].copy()
    delta = np.zeros(3)
    delta[plan['component']] = plan['destination_axis_m'] - start[plan['component']]
    bad = []
    for fraction in np.linspace(0, 1, 151):
        g.restore()
        g.d.qpos[address:address + 3] = start + fraction * delta
        g.mj.mj_forward(g.m, g.d)
        for contact in g.d.contact:
            names = [g.m.geom(int(i)).name for i in contact.geom]
            if target in names and any(n != target and (n == 'target' or n.startswith('neighbor_')) for n in names):
                bad.append(dict(fraction=float(fraction), pair=names, depth_mm=-float(contact.dist) * 1000))
    g.restore()
    return bad


def evaluate(out):
    record = json.loads((out / 'result.json').read_text())
    summary = record['summary']
    summary.update(source_sequence_continuity_verified=True, diagnostic_only=True)
    summary['blue_feed_position_verified'] = summary.pop('local_red_position_verified', False)
    (out / 'result.json').write_text(json.dumps(record, allow_nan=False))
    assessment = seven.assess(out)
    stable = bool(not summary['abort'] and not summary['unintended_contacts']
        and summary.get('held_table_only', False) and summary.get('held_duration_s', 0) >= 2.98
        and summary.get('held_drift_mm', 1e6) <= 2 and summary.get('held_angular_drift_degrees', 1e6) <= 2
        and summary.get('final_face_tilt_degrees', 1e6) <= 2
        and summary.get('minimum_table_support_n', 0) >= .8 * g.m.body(summary['target']).mass * 9.81
        and max(summary['other_bricks_max_shift_mm'].values()) <= 2)
    summary['local_blue_table_feed_verified'] = bool(stable and assessment['local_seven_brick_final_structure_verified'])
    summary['local_seven_brick_final_structure_verified'] = assessment['local_seven_brick_final_structure_verified']
    (out / 'result.json').write_text(json.dumps(record, allow_nan=False))
    (out / 'seven-brick-assessment.json').write_text(json.dumps(assessment, allow_nan=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=INITIAL)
    parser.add_argument('--target', choices=('neighbor_0', 'neighbor_7'), default='neighbor_0')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--axis', choices=('south', 'west'))
    modes.add_argument('--return-receiver', action='store_true')
    parser.add_argument('--destination-mm', type=float)
    parser.add_argument('--single-finger', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-shallow-payload-prediction', action='store_true',
                        help='Local diagnostic only: retain actual collision stops for a <=0.5 mm predicted blue/green overlap')
    args = parser.parse_args()
    if args.axis and args.destination_mm is None:
        parser.error('--axis requires --destination-mm')
    if args.return_receiver and (args.destination_mm is not None or args.single_finger):
        parser.error('Receiver return does not take feed options')
    if args.allow_shallow_payload_prediction and (args.return_receiver or args.axis != 'south'):
        parser.error('Shallow prediction diagnostic applies only to south feeding')
    left, right = configure(args.source)
    right_command = None
    if args.return_receiver:
        if g.RECORD['summary']['target'] != args.target:
            raise ValueError('Receiver return must retain the source feed target')
        returner.SOURCE, returner.RIGHT_SEED, returner.TARGET = args.source, args.source / 'precheck.json', args.target
        command, right_command, fingers, joints, plan = returner.prepare(g.restore, np.array(right))
        stages = g.CHECK['stages']
        left_stage = stages['left'][-1] if isinstance(stages, dict) else stages[-1]
        check = returner.precheck(command, right_command, fingers, joints, g.restore, left_stage)
        check['predicted_payload_geometry_failures'] = []
    else:
        command, fingers, plan = prepare(args.source, args.target, args.axis, args.destination_mm / 1000, args.single_finger)
        check = first.precheck(command, g.restore, left, target=args.target, finger_command=fingers)
        check.update(predicted_payload_geometry_failures=payload_precheck(plan), right_final_joints_sdk=right, stationary_right=True)
    args.out.mkdir(parents=True, exist_ok=False)
    predicted = check['predicted_payload_geometry_failures']
    shallow = shallow_prediction_permitted(predicted, args.target, args.allow_shallow_payload_prediction)
    check['strict_nominal_payload_geometry_passed'] = not predicted
    if args.allow_shallow_payload_prediction:
        check.update(shallow_payload_prediction_diagnostic=shallow,
                     payload_prediction_scope='Frozen-orientation translation only; actual original 0.5 mm collision and 2 mm neighbor-motion stops retained')
    check['passed'] &= not predicted or shallow
    (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
    (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
    passed = check['passed']
    brief = dict(precheck_passed=passed, geometry_failures=len(check['geometry_failures']),
                 payload_geometry_failures=len(check['predicted_payload_geometry_failures']), stages=len(check['stages']))
    if passed and args.run:
        first.run(command, plan, g.restore, args.out, finger_command=fingers, protected_payload='neighbor_5',
                  additional_protected_payloads=('neighbor_1', 'neighbor_3', *seven.release.feed.BASES), right_command=right_command)
        if args.return_receiver:
            returner.evaluate(args.out)
        brief['dynamics'] = evaluate(args.out)
        passed = brief['dynamics']['local_blue_table_feed_verified']
        if args.return_receiver:
            passed &= brief['dynamics']['local_receiver_return_verified']
    print(json.dumps(brief, indent=2))
    raise SystemExit(0 if passed else 1)
