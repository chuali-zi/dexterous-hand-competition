"""Try the existing table-edge receiver on the actual blue supply checkpoint."""
import argparse
import json
from pathlib import Path

import try_blue_upper_feed as feed
import try_yellow_edge_acquisition as acquisition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/blue-feed-117/right-return'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--vertical-approach', action='store_true')
    closing = parser.add_mutually_exclusive_group()
    closing.add_argument('--cup-start-s', type=int, choices=(16, 20), default=20)
    closing.add_argument('--open-receiver', action='store_true')
    args = parser.parse_args()
    feed.configure(args.source)
    target = feed.g.RECORD['summary']['target']
    acquisition.configure_continuation(args.source, target, edge_blue_receiver=True)
    acquisition.DURATION = 31.
    acquisition.CUP_BEFORE_LIFT = True
    # Compare the early-edge hypothesis with the existing 20 s receiver timing.
    acquisition.CUP_START = acquisition.DURATION + 1 if args.open_receiver else float(args.cup_start_s)
    acquisition.DELAYED_PINKY = False
    acquisition.RECENTER_WIDE_FACE = False
    acquisition.four_finger_face()
    command, plan = acquisition.prepare(0. if args.vertical_approach else .020)
    plan.update(duration_s=31., hold_start_s=28., diagnostic_only=True,
                seven_brick_source=str(feed.INITIAL), receiver_fingers_kept_open=args.open_receiver)
    if args.open_receiver:
        plan.update(right_curl_start_s=None, right_curl_schedule=None)
    args.out.mkdir(parents=True, exist_ok=False)
    check = acquisition.precheck(command)
    (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
    (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
    passed = check['passed']
    brief = dict(precheck_passed=passed, geometry_failures=len(check['geometry_failures']),
                 left_stages=len(check['left_stages']))
    if passed and args.run:
        acquisition.run(command, args.out, protected_payload='neighbor_5',
                        additional_protected_payloads=('neighbor_1', 'neighbor_3', *feed.seven.release.feed.BASES))
        record = json.loads((args.out / 'result.json').read_text())
        summary = record['summary']
        summary.update(unintended_contacts=summary['unintended_hand_contacts'], diagnostic_only=True,
                       full_pyramid_verified=False, official_region_verified=False)
        (args.out / 'result.json').write_text(json.dumps(record, allow_nan=False))
        assessment = feed.seven.assess(args.out, held_payload=target)
        summary['local_seven_brick_final_structure_verified'] = assessment['local_seven_brick_final_structure_verified']
        summary['local_acquisition_verified'] &= summary['local_seven_brick_final_structure_verified']
        (args.out / 'result.json').write_text(json.dumps(record, allow_nan=False))
        (args.out / 'seven-brick-assessment.json').write_text(json.dumps(assessment, allow_nan=False))
        brief['dynamics'] = summary
        passed = summary['local_acquisition_verified']
    print(json.dumps(brief, indent=2))
    raise SystemExit(0 if passed else 1)
