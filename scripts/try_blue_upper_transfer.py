"""Carry the actually acquired blue above an adjacent yellow support pair."""
import argparse
import json
from pathlib import Path

import try_contact_guided_release as seven
import try_loaded_yellow_transfer as transfer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/receiver-support-118/southern-acquisition'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    record = json.loads((args.source / 'result.json').read_text())
    target = record['summary']['target']
    if not seven.assess(args.source, held_payload=target)['local_seven_brick_final_structure_verified']:
        raise ValueError('Source no longer satisfies the seven-brick structure check')
    transfer.configure_continuation(args.source, target, shared_support=True,
                                   supports=('neighbor_5', 'neighbor_1'), seven_brick_support=True)
    command, plan = transfer.prepare(.150)
    plan.update(duration_s=16., hold_start_s=13., minimum_transport_clearance_m=.105,
                diagnostic_only=True, seven_brick_source=str(args.source))
    args.out.mkdir(parents=True, exist_ok=False)
    check = transfer.precheck(command)
    (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
    (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
    passed = check['passed']
    brief = dict(precheck_passed=passed, geometry_failures=len(check['geometry_failures']),
                 stages={s: len(v) for s, v in check['stages'].items()})
    if passed and args.run:
        transfer.run(command, plan, args.out, protected_payload='neighbor_5',
                     additional_protected_payloads=('neighbor_1', 'neighbor_3', *seven.release.feed.BASES),
                     transport_clearance=.105)
        record = json.loads((args.out / 'result.json').read_text())
        summary = record['summary']
        summary.update(diagnostic_only=True, full_pyramid_verified=False, official_region_verified=False)
        assessment = seven.assess(args.out, held_payload=target)
        summary['local_seven_brick_final_structure_verified'] = assessment['local_seven_brick_final_structure_verified']
        summary['local_loaded_transfer_verified'] &= summary['local_seven_brick_final_structure_verified']
        summary['local_shared_transport_hold_verified'] = bool(summary['local_seven_brick_final_structure_verified']
            and not summary['abort'] and not summary['unintended_contacts'] and summary.get('held_duration_s', 0) >= 2.98
            and summary.get('held_drift_mm', 1e6) <= 2 and summary.get('held_angular_drift_degrees', 1e6) <= 2
            and summary.get('held_only_hand_contacts', False) and summary.get('held_minimum_bottom_mm', 0) >= 105
            and summary.get('held_minimum_up_force_n', 0) >= .8 * transfer.m.body(target).mass * 9.81
            and summary['checkpoint_next_step_maximum_difference'] <= 1e-10)
        (args.out / 'result.json').write_text(json.dumps(record, allow_nan=False))
        (args.out / 'seven-brick-assessment.json').write_text(json.dumps(assessment, allow_nan=False))
        brief['dynamics'] = summary
        passed = summary['local_shared_transport_hold_verified']
    print(json.dumps(brief, indent=2))
    raise SystemExit(0 if passed else 1)
