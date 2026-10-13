#!/usr/bin/env python3
"""Record independent reviewer RGB regressions; no robot or author generator."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_multi_block_review import calibration,reviewer_packet,COLORS
from hand_competition.multi_block.planning import PlanConfig
from hand_competition.multi_block.perception import VisionConfig
from hand_competition.multi_block.session import PlanningSession
from hand_competition.multi_block.records import PlanningTrace,replay
from hand_competition.closed_loop.experiments import code_identity


def cases():
    source=[('red',(-.2,-.12,.025))]
    yellow=[('yellow',(-.2,-.12,.025))]
    different=[('red',(-.1+(216-256)*.94/390,-50*.94/390,.06)),
               ('red',(.1+(136-256)*.82/390,-50*.82/390,.18))]
    return [
      ('MB1-unmatched-blue',source+[('blue',(.08,0.,.15))],{}, {'visibility':((True,True),(True,False))},'reject'),
      ('MB2-footprint-50percent',[('red',(.0725,0.,.025)),('red',(.1475,0.,.025))]+yellow,{}, {},'reject'),
      ('MB2-elevated-support',[('red',(.08,0.,.0325)),('red',(.14,0.,.0325))]+yellow,{'half_baseline':.16}, {},'reject'),
      ('MB3-asymmetric-clipping',[('red',(-.2,-.12,.06))],{}, {'crop_left':((2,),(8,))},'reject'),
      ('MB3-symmetric-clipping',source,{}, {'crop_left':((2,),(2,))},'reject'),
      ('tiny-unmatched-obstacle',source,{}, {'tiny_blue_pixel':True},'reject'),
      ('normal-source',source,{}, {},'layer-0-slot-0'),
      ('normal-supports',[('red',(.08,0.,.025)),('red',(.14,0.,.025))]+yellow,{}, {},'layer-1-slot-0'),
      ('indistinguishable-different-objects',different,{}, {'visibility':((True,False),(False,True))},'unauthenticated_conditional_hypothesis')]


def run_case(path,name,scene,cam_kwargs,render_kwargs,expected):
    cameras=calibration(**cam_kwargs);config=PlanConfig();vision=VisionConfig()
    session=PlanningSession(cameras,'reviewer_rgb',config,vision)
    manifest={'schema_version':2,'feedback_level':'offline_multi_block_planning','official_platform':False,
      'planning':asdict(config),'vision':asdict(vision),'cameras':[asdict(c) for c in cameras],'producer':'reviewer_rgb',
      'code':code_identity(),'fixture_kind':'independent reviewer equations from test_multi_block_review.py',
      'fixture_sha256':hashlib.sha256((ROOT/'tests/test_multi_block_review.py').read_bytes()).hexdigest(),
      'effect_model':'none; fixed reviewer RGB observations, no contact or robot transport'}
    trace=PlanningTrace(path,manifest);decisions=[];tiny=render_kwargs.get('tiny_blue_pixel',False)
    kwargs={k:v for k,v in render_kwargs.items() if k!='tiny_blue_pixel'}
    for sequence in range(1,33):
        raw=reviewer_packet(scene,sequence,cameras,**kwargs)
        if tiny:
            rgb=bytearray(raw['frames'][0]['rgb']);offset=3*(192*512+328)
            rgb[offset:offset+3]=bytes(COLORS['blue']);raw['frames'][0]['rgb']=bytes(rgb)
        _,snapshot=session.observe(raw,raw['time_s']);trace.observation(raw,raw['time_s'],snapshot)
        decisions.append(snapshot['decision'])
    candidate=decisions[-1]['candidate']
    if expected=='reject':assert all(d['candidate'] is None for d in decisions),name
    elif expected=='unauthenticated_conditional_hypothesis':
        assert candidate and not decisions[-1]['correspondence_authenticity_verified'] and not decisions[-1]['execution_available'],name
    else:assert candidate and candidate['target_slot_id']==expected,name
    session.stop()
    result={'feedback_level':'offline_multi_block_planning','official_platform':False,'official_score':None,'robot_actions_executed':0,
      'observations':32,'stop_reason':'fixture_complete','geometric_goal_count':len(decisions[-1]['geometrically_supported_slots']),
      'physical_support_verified':False,'sensor_authenticity_verified':False}
    trace.finish(3.2,result)
    return {'case':name,'expectation':expected,'assertions_passed':True,'result':result,'last_snapshot':snapshot,
      'candidate_observations':sum(d['candidate'] is not None for d in decisions),'replay':replay(path),
      'evaluator_only_scene':scene,'evaluator_visibility_and_crop':render_kwargs,
      'artifact_sha256':{p:hashlib.sha256((path/p).read_bytes()).hexdigest() for p in ('manifest.json','events.jsonl','result.json')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name,scene,cam_kwargs,kwargs,expected in cases():
        row=run_case(root/name,name,scene,cam_kwargs,kwargs,expected);rows.append(row)
        print(name,row['candidate_observations'],row['last_snapshot']['decision']['kind'],flush=True)
    report={'schema_version':1,'code':code_identity(),'official_platform':False,'official_score':None,
      'robot_actions_executed':0,'observations':sum(r['result']['observations'] for r in rows),'trials':rows}
    (root/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print({'trials':len(rows),'observations':report['observations'],'all_replayed':all(r['replay']['record_verified'] for r in rows)})


if __name__=='__main__':main()
