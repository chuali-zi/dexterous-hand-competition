"""Return the empty right receiver from the actual third-yellow supply state.

Preserve both upper bricks and all current scene state. SDK zero is reached by
the verified exterior joint route, never by resetting the hand or arm.
"""
import argparse,json
from pathlib import Path
import numpy as np
import try_third_yellow_feed as feed
import try_receiver_return as returner


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/third-pickup-111/south-pickup'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    feed.configure(args.source);left_seed,right_seed=feed.seeds();g=feed.g
    stages=g.CHECK['stages'];left_stage=stages['left'][-1] if isinstance(stages,dict) else stages[-1]
    returner.SOURCE=args.source;returner.RIGHT_SEED=args.source/'precheck.json';returner.TARGET=feed.TARGET
    command,right,fingers,joints,plan=returner.prepare(g.restore,np.array(right_seed))
    plan.update(structure_reference=str(feed.release.REFERENCE),second_upper_reference=str(feed.SECOND_REFERENCE),protected_upper_bricks=['neighbor_5',feed.SECOND])
    check=returner.precheck(command,right,fingers,joints,g.restore,left_stage)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        feed.first.run(command,plan,g.restore,args.out,finger_command=fingers,right_command=right,protected_payload='neighbor_5',additional_protected_payloads=(feed.SECOND,*feed.BASES))
        returner.evaluate(args.out);result=feed.evaluate(args.out)
        result['local_receiver_return_verified'] &= result['protected_structure_verified']
        record=json.loads((args.out/'result.json').read_text());record['summary']=result;(args.out/'result.json').write_text(json.dumps(record,allow_nan=False))
        brief['dynamics']=result;passed=result['local_receiver_return_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
