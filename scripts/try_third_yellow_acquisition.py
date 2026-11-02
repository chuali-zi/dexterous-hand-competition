"""Test the established wide-face/early-cup pickup on the third actual yellow.

Preserve the six-brick structure and the complete returned-receiver state.
Shared hand support may pass acquisition; placement and native proof remain false.
"""
import argparse,json
from pathlib import Path
import try_third_yellow_feed as feed
import try_yellow_edge_acquisition as a


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/third-pickup-111/right-return'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    feed.configure(args.source)
    if not feed.g.RECORD['summary'].get('local_receiver_return_verified',False):raise ValueError('Receiver return must be physically verified')
    a.configure_continuation(args.source,feed.TARGET,southern_receiver=True)
    a.DURATION=31.;a.CUP_BEFORE_LIFT=True;a.CUP_START=20.;a.DELAYED_PINKY=False;a.RECENTER_WIDE_FACE=False;a.four_finger_face()
    command,plan=a.prepare();plan.update(structure_reference=str(feed.release.REFERENCE),second_upper_reference=str(feed.SECOND_REFERENCE),protected_upper_bricks=['neighbor_5',feed.SECOND])
    args.out.mkdir(parents=True,exist_ok=False);check=a.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        a.run(command,args.out,protected_payload='neighbor_5',additional_protected_payloads=(feed.SECOND,*feed.BASES))
        record=json.loads((args.out/'result.json').read_text());record['summary']['unintended_contacts']=record['summary']['unintended_hand_contacts']
        (args.out/'result.json').write_text(json.dumps(record,allow_nan=False));result=feed.evaluate(args.out)
        result['local_acquisition_verified'] &= result['protected_structure_verified']
        held=[r for r in record['rows'] if r['elapsed_s']>=a.DURATION-3-1e-8]
        result['held_only_right_hand_contacts']=bool(held and all(r['contacts'] and all(any(n.startswith('rh_') for n in c['pair']) for c in r['contacts']) for r in held))
        record=json.loads((args.out/'result.json').read_text());record['summary']=result;(args.out/'result.json').write_text(json.dumps(record,allow_nan=False))
        brief['dynamics']=result;passed=result['local_acquisition_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
