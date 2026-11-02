"""Continue the actual third-yellow shared acquisition over the middle red pair."""
import argparse,json
from pathlib import Path
import try_loaded_yellow_transfer as transfer
import try_third_yellow_feed as feed


def evaluate(out):
    result=feed.evaluate(out)
    result['local_loaded_transfer_verified'] &= result['protected_structure_verified']
    # Settling within the cupped hand may invalidate rigid-translation accuracy
    # while leaving a physically stable hold available for an alignment motion.
    result['local_shared_transport_hold_verified']=bool(result['protected_structure_verified'] and not result['abort']
        and not result['unintended_contacts'] and result.get('held_duration_s',0)>=2.98
        and result.get('held_drift_mm',float('inf'))<=2 and result.get('held_angular_drift_degrees',float('inf'))<=2
        and result.get('held_only_hand_contacts',False) and result.get('held_minimum_bottom_mm',0)>=55
        and result.get('held_minimum_up_force_n',0)>=.8*transfer.m.body(feed.TARGET).mass*9.81
        and result['checkpoint_next_step_maximum_difference']<=1e-10)
    record=json.loads((out/'result.json').read_text());record['summary']=result;(out/'result.json').write_text(json.dumps(record,allow_nan=False));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/third-pickup-111/acquisition'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--lift-mm',type=float,default=100.)
    args=parser.parse_args()
    transfer.configure_continuation(args.source,feed.TARGET,shared_support=True,supports=('neighbor_6','neighbor_4'))
    command,plan=transfer.prepare(args.lift_mm/1000)
    plan.update(structure_reference=str(feed.release.REFERENCE),second_upper_reference=str(feed.SECOND_REFERENCE))
    args.out.mkdir(parents=True,exist_ok=False);check=transfer.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(check,allow_nan=False))
    passed=check['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(check['geometry_failures']),stage_counts={s:len(v) for s,v in check['stages'].items()},precheck_wall_seconds=check['wall_seconds'])
    if passed and args.run:
        transfer.run(command,plan,args.out,protected_payload='neighbor_5',additional_protected_payloads=(feed.SECOND,*feed.BASES))
        result=evaluate(args.out)
        brief['dynamics']=result;passed=result['local_loaded_transfer_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
