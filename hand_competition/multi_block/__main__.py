import argparse,json
from .records import replay
from .planning import PlanConfig,build_slots
from ..closed_loop.artifacts import load_json

def main():
    parser=argparse.ArgumentParser(description='Offline RGB multi-block planner; no robot execution')
    sub=parser.add_subparsers(dest='command',required=True)
    rp=sub.add_parser('replay');rp.add_argument('directory')
    cfg=sub.add_parser('blueprint');cfg.add_argument('config')
    args=parser.parse_args()
    try:
        if args.command=='replay':result=replay(args.directory)
        else:
            from dataclasses import asdict
            config=PlanConfig.from_dict(load_json(args.config)['planning'])
            result={'planning_only':True,'official_scene_confirmed':False,'slots':[asdict(s) for s in build_slots(config)]}
    except (ValueError,OSError) as error:parser.error(str(error))
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
