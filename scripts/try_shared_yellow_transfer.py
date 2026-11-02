"""Transport the second yellow with both hands from its actual shared hold.

This local experiment preserves the four red bases and first yellow. It tests
loaded transport only; neither the payload nor any other brick is bound/reset.
"""
import argparse,json
from pathlib import Path
import try_loaded_yellow_transfer as transfer
import try_fourth_red_with_upper as fourth

REFERENCE=Path('runs/fourth-red-105/final-south')


def evaluate(out):
    record=json.loads((out/'result.json').read_text());s=record['summary']
    metrics,only,forces=fourth.third.structure_status(record['rows'],REFERENCE,extra_bases=('neighbor_4','neighbor_2'))
    safe=bool(not s['abort'] and not s['unintended_contacts'] and only and min(forces.values())>=.1
              and all(v['maximum_shift_mm']<=2 and v['maximum_angular_shift_degrees']<=2 for v in metrics.values()))
    s.update(protected_structure_metrics=metrics,protected_structure_verified=safe,upper_only_base_contacts=only,upper_minimum_support_force_n=forces)
    s['local_loaded_transfer_verified']=bool(s['local_loaded_transfer_verified'] and safe)
    (out/'result.json').write_text(json.dumps(record,allow_nan=False));return s


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/next-pickup-106/south-acquisition-early-cup'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--lift-mm',type=float,default=100.)
    args=parser.parse_args()
    transfer.configure_continuation(args.source,'neighbor_3',shared_support=True,supports=('neighbor_6','neighbor_4'))
    command,plan=transfer.prepare(args.lift_mm/1000);plan.update(structure_reference=str(REFERENCE),protected_structure=[*fourth.BASES,fourth.TARGET,fourth.UPPER])
    args.out.mkdir(parents=True,exist_ok=False);checked=transfer.precheck(command)
    (args.out/'plan.json').write_text(json.dumps(plan,allow_nan=False));(args.out/'precheck.json').write_text(json.dumps(checked,allow_nan=False))
    passed=checked['passed'];brief=dict(precheck_passed=passed,geometry_failure_count=len(checked['geometry_failures']),
                                       stage_counts={s:len(v) for s,v in checked['stages'].items()},precheck_wall_seconds=checked['wall_seconds'])
    if args.run and passed:
        transfer.run(command,plan,args.out,protected_payload=fourth.UPPER);brief['dynamics']=evaluate(args.out)
        passed=brief['dynamics']['local_loaded_transfer_verified']
    print(json.dumps(brief,indent=2));raise SystemExit(0 if passed else 1)
