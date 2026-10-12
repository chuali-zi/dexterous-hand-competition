#!/usr/bin/env python3
"""Small offline matched/held-out/negative scan. No robot or learned models.

Synthetic object truth is used only in the evaluator's safety report, never in
observations, estimator or controller. Hold-outs are first run after parameters
are frozen; matched seeds 0/1/2 were already inspected and are not hold-outs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from hand_competition.closed_loop.artifacts import Trace
from hand_competition.closed_loop.contracts import Config
from hand_competition.closed_loop.controller import Controller
from hand_competition.closed_loop.experiments import code_identity, replay
from hand_competition.closed_loop.geometry import look_at, distance
from hand_competition.closed_loop.synthetic import SyntheticBackend, SensorNoise
from motion_fixtures import (AlternatingPlacement,Alternating16,PeriodicPlacement,SlowDriftPlacement,
                             evaluated_backend,evaluate_window)
from test_offline_hardening import (SparseSettling, OverDeadlineSettling, SlowComplete,
    IncompleteRetreat, OccludedTransit, HugeTip, InvalidStatus)


class WiderPixels(SyntheticBackend):
    cameras = tuple(replace(c, width=240,height=180,fx=270,fy=270,cx=120,cy=90) for c in SyntheticBackend.cameras)


class NarrowerBaseline(SyntheticBackend):
    cameras = (look_at('camera_left',(-.30,-.50,.65),(0,.4,.05)),
               look_at('camera_right',(.30,-.50,.65),(0,.4,.05)))


class PersistentPixels:
    """Piecewise constant sensor jitter, four consecutive frames correlated."""
    def __init__(self, backend, seed): self.backend=backend; self.seed=seed; self.calls=0
    def randint(self, low, high):
        coordinate=self.calls % 8; self.calls+=1
        return random.Random(self.seed*1000+(self.backend._sequence//4)*8+coordinate).randint(low,high)


class CorrelatedPixels(SyntheticBackend):
    def __init__(self, scenario='nominal',seed=0,noise=None):
        super().__init__(scenario,seed,noise); self._pixel_rng=PersistentPixels(self,seed)


class LateStep(SyntheticBackend):
    settling=False; settling_start=None; changed_s=None
    def _advance(self):
        super()._advance()
        if self.settling and self.changed_s is None and self._time-self.settling_start >= 2.6:
            self._block=(self._block[0]+.020,*self._block[1:]); self.changed_s=self._time


class EvaluationTrace(Trace):
    def __init__(self,path,manifest,backend): super().__init__(path,manifest); self.backend=backend
    def event(self,kind,stamp,**payload):
        if kind=='settle_started':
            self.backend.settling=True
            if getattr(self.backend,'settling_start',None) is None: self.backend.settling_start=stamp
        super().event(kind,stamp,**payload)


BACKENDS={c.__name__:c for c in (SyntheticBackend,WiderPixels,NarrowerBaseline,CorrelatedPixels,LateStep,
                               SparseSettling,OverDeadlineSettling,SlowComplete,IncompleteRetreat,OccludedTransit,HugeTip,InvalidStatus,AlternatingPlacement,Alternating16,PeriodicPlacement,SlowDriftPlacement)}


def execute(args):
    root,row=args; config=Config(robust_observations=row['mode']=='robust')
    cls=BACKENDS[row.get('backend','SyntheticBackend')]
    backend=evaluated_backend(cls,row.get('scenario','nominal'),row['seed'],SensorNoise(row['jitter'],row['shift']))
    path=Path(root)/row['name']
    manifest={'schema_version':5,'environment':backend.environment,'producer':backend.producer,
              'config':asdict(config),'cameras':[asdict(c) for c in backend.cameras],'code':code_identity(),
              'evaluation_fixture':row,'fixture_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'motion_fixture_source_sha256':hashlib.sha256((ROOT/'scripts/motion_fixtures.py').read_bytes()).hexdigest()}
    result=Controller(backend,config,EvaluationTrace(path,manifest,backend)).run()
    try: verification=replay(path)
    except ValueError as error: verification={'trace_verified':False,'error':str(error)}
    # Read truth only after Controller.run ended; replay never reads this file.
    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    safety=evaluate_window(backend.evaluation_history,result,events,config)
    (path/'evaluation-trajectory.json').write_text(json.dumps(backend.evaluation_history,indent=2)+'\n')
    safety['trajectory_file']='evaluation-trajectory.json'
    safety['trajectory_sha256']=hashlib.sha256((path/'evaluation-trajectory.json').read_bytes()).hexdigest()
    safety['evaluator_only_final_goal_error_m']=distance(backend._block,config.target)
    phases=[e['payload']['phase'] for e in events if e['kind']=='action_submitted']
    resets={}
    for event in events:
        if event['kind']=='settle_reset':
            reason=event['payload']['reason'];resets[reason]=resets.get(reason,0)+1

    return {**row,'result':result,'replay':verification,'evaluator':safety,'phases':phases,'stability_resets':resets}


def cases(suites):
    rows=[]
    def add(suite,mode,seed,jitter=0,shift=0,backend='SyntheticBackend',scenario='nominal'):
        name=f'{suite}-{mode}-{backend}-{scenario}-j{jitter}-x{int(shift*1000)}-s{seed}'
        rows.append(dict(name=name,suite=suite,mode=mode,seed=seed,jitter=jitter,shift=shift,backend=backend,scenario=scenario))
    if 'matched' in suites:
        for mode in ('raw','robust'):
            for jitter in (0,1,2):
                for shift in (0,.010):
                    for seed in (0,1,2): add('matched',mode,seed,jitter,shift)
    if 'holdout' in suites:
        for backend in ('SyntheticBackend','WiderPixels','NarrowerBaseline','CorrelatedPixels'):
            for seed in (100,101,102): add('holdout','robust',seed,2 if backend!='CorrelatedPixels' else 1,backend=backend)
    if 'negative' in suites:
        for mode in ('raw','robust'):
            for backend in ('SparseSettling','OverDeadlineSettling','SlowComplete','IncompleteRetreat','OccludedTransit','HugeTip','InvalidStatus'):
                add('negative',mode,0,backend=backend)
            for scenario in ('slip_during_transit','continuous_drift','false_move_complete','pose_leak','camera_skew','stale_after_close','duplicate_after_close'):
                add('negative',mode,0,scenario=scenario)
        for jitter in (0,1,2): add('negative','robust',101,jitter,backend='LateStep')
    if 'motion' in suites:
        for mode in ('raw','robust'):
            for backend in ('Alternating16','AlternatingPlacement','PeriodicPlacement','SlowDriftPlacement'):
                for jitter in (0,1): add('motion',mode,0,jitter,backend=backend)
    if len({r['name'] for r in rows}) != len(rows): raise ValueError('duplicate case names')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True); parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--suites',nargs='+',choices=['matched','holdout','negative','motion'],default=['matched','holdout','negative','motion'])
    args=parser.parse_args()
    if not 1<=args.workers<=3: parser.error('workers must be 1..3')
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    report={'schema_version':2,'official_platform':False,'official_score':None,'runtime':code_identity(),
            'fixture_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'motion_fixture_source_sha256':hashlib.sha256((ROOT/'scripts/motion_fixtures.py').read_bytes()).hexdigest(),
            'evaluation_scope':'entire successful window at every generated tick; no controller access to truth',
            'holdout_status':'previously observed configurations; regression, not new holdout',
            'policy':asdict(Config(robust_observations=True)),'historical_development_seeds':[20,21],
            'matched_seeds_already_seen':[0,1,2],'previous_holdout_regression_seeds':[100,101,102],'trials':[]}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(execute,[(str(root),row) for row in cases(args.suites)]):
            report['trials'].append(row)
            print(row['name'],row['result']['success'],row['result']['terminal_failure'],row['replay'].get('trace_verified'),flush=True)
    report['groups']={}
    for row in report['trials']:
        key=row['suite']+'-'+row['mode']; group=report['groups'].setdefault(key,dict(trials=0,successes=0,replay_failures=0,false_accepts=0))
        group['trials']+=1; group['successes']+=row['result']['success']
        group['replay_failures']+=not row['replay'].get('trace_verified',False)
        group['false_accepts']+=row['evaluator']['false_accept_under_evaluator_checks']
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['groups'],indent=2),flush=True)
    if any(g['replay_failures'] or g['false_accepts'] for g in report['groups'].values()): raise SystemExit(1)


if __name__=='__main__': main()
