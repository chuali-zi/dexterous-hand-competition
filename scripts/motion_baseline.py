#!/usr/bin/env python3
"""Run trusted archived local code with a current full-window truth evaluator."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkout',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args();checkout=Path(args.checkout).resolve();root=Path(args.output).resolve()
sys.path.insert(0,str(checkout))
from hand_competition.closed_loop.contracts import Config
from hand_competition.closed_loop.controller import Controller
from hand_competition.closed_loop.artifacts import Trace
from hand_competition.closed_loop.experiments import replay,code_identity
from motion_fixtures import (AlternatingPlacement,Alternating16,PeriodicPlacement,SlowDriftPlacement,
                            evaluated_backend,evaluate_window)
root.mkdir(parents=True,exist_ok=False)
report={'schema_version':1,'runtime':code_identity(),'evaluator_only':True,'official_platform':False,'trials':[]}
for cls in (Alternating16,AlternatingPlacement,PeriodicPlacement,SlowDriftPlacement):
    for robust in (False,True):
        cfg=Config(robust_observations=robust);backend=evaluated_backend(cls);path=root/(cls.__name__+str(robust))
        manifest={'schema_version':4,'environment':backend.environment,'producer':backend.producer,'config':asdict(cfg),
                  'cameras':[asdict(c) for c in backend.cameras],'code':code_identity()}
        result=Controller(backend,cfg,Trace(path,manifest)).run()
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        evaluated=evaluate_window(backend.evaluation_history,result,events,cfg)
        (path/'evaluation-trajectory.json').write_text(json.dumps(backend.evaluation_history,indent=2)+'\n')
        checked=replay(path)
        report['trials'].append({'backend':cls.__name__,'robust':robust,'result':result,'replay':checked,'evaluator':evaluated})
        print(cls.__name__,robust,result['success'],evaluated['violation_categories'],flush=True)
report['false_accepts']=sum(r['evaluator']['false_accept_under_evaluator_checks'] for r in report['trials'])
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
