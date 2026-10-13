#!/usr/bin/env python3
"""Small RGB association/geometry planning scan; no robot actions or scoring."""
import argparse,json,sys,hashlib
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hand_competition.multi_block.planning import PlanConfig
from hand_competition.multi_block.perception import VisionConfig
from hand_competition.multi_block.session import PlanningSession
from hand_competition.multi_block.records import PlanningTrace,replay
from hand_competition.closed_loop.experiments import code_identity
from multi_block_fixtures import CAMERAS,packet,staged_scene,goal_scene,calibrated_cameras


def run_case(path,stages,seed=0,jitter=0,cameras=CAMERAS):
    config=PlanConfig();vision=VisionConfig();session=PlanningSession(cameras,'offline_multi_rgb_fixture_v1',config,vision)
    manifest={'schema_version':2,'feedback_level':'offline_multi_block_planning','official_platform':False,
       'planning':asdict(config),'vision':asdict(vision),'cameras':[asdict(c) for c in cameras],'producer':session.producer,
       'code':code_identity(),'fixture_source_sha256':hashlib.sha256((ROOT/'scripts/multi_block_fixtures.py').read_bytes()).hexdigest(),
       'effect_model':'none; predeclared RGB snapshots, no executed robot transport or contact physics','seed':seed,'pixel_jitter':jitter}
    trace=PlanningTrace(path,manifest);sequence=0;stage_results=[];last=None
    for stage in stages:
        scene=stage['scene'];hidden=stage.get('hidden_colors',())
        for _ in range(32):
            sequence+=1;stamp=round(sequence*.1,6);raw=packet(scene,sequence,stamp,hidden,jitter,seed,cameras)
            _,last=session.observe(raw,stamp);trace.observation(raw,stamp,last)
        stage_results.append({'stage':stage['name'],'tracks':len(last['tracks']),'snapshot':last})
    session.stop()
    result={'feedback_level':'offline_multi_block_planning','official_platform':False,'official_score':None,'robot_actions_executed':0,
        'observations':sequence,'stop_reason':'fixture_complete','geometric_goal_count':len(last['decision']['geometrically_supported_slots']),
        'physical_support_verified':False,'sensor_authenticity_verified':False}
    trace.finish(stamp,result)
    return {'result':result,'stages':stage_results,'replay':replay(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    full=goal_scene();collapsed=[item for item in full if item['hidden_label'] not in ('goal-eval-0-0','goal-eval-1-0','goal-eval-2-0','goal-eval-3-0')]
    fallen=[dict(item,center=(-.28,-.15+i*.10,.025)) for i,item in enumerate(full) if item['hidden_label'] in ('goal-eval-0-0','goal-eval-1-0','goal-eval-2-0','goal-eval-3-0')]
    # Evaluator chooses snapshots. Hidden labels never appear in sensor packets.
    for i,item in enumerate(fallen):item['center']=(-.28,(-.15,-.05,.05,.15)[i],.025)
    report={'schema_version':1,'code':code_identity(),'official_platform':False,'robot_actions_executed':0,'trials':[]}
    for seed in (0,17,101):
        stages=[{'name':'supply','scene':staged_scene(0,seed)},{'name':'base','scene':staged_scene(1,seed)},
                {'name':'two_layers','scene':staged_scene(2,seed)},{'name':'all_geometric_slots','scene':full},
                {'name':'branch_removed','scene':collapsed+fallen}]
        row=run_case(root/f'progress-seed-{seed}',stages,seed);report['trials'].append({'case':f'progress-seed-{seed}',**row})
        print(f'progress-seed-{seed}',[(s['stage'],s['tracks'],s['snapshot']['decision']['kind']) for s in row['stages']],flush=True)
    report['trials'].append({'case':'occlusion',**run_case(root/'occlusion',[{'name':'supply','scene':staged_scene()},
       {'name':'red_hidden_in_one_view','scene':staged_scene(),'hidden_colors':('red',)},{'name':'reacquired','scene':staged_scene()}])})
    report['trials'].append({'case':'jitter-1px',**run_case(root/'jitter-1px',[{'name':'noisy_supply','scene':staged_scene()}],jitter=1)})
    # Chosen after the five development cases, before inspecting these results.
    # Parameters stay frozen; these are supply-only checks, not stack success.
    for seed,half_baseline,focal in ((207,.16,340.),(509,.06,450.)):
        name=f'heldout-supply-{seed}'
        report['trials'].append({'case':name,'split':'prospective_heldout_supply_only',
          **run_case(root/name,[{'name':'heldout_supply','scene':staged_scene(0,seed)}],seed,
                     cameras=calibrated_cameras(half_baseline,focal))})
    (root/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print({'planning_trials':len(report['trials']),'robot_actions':0,'all_replayed':all(t['replay']['record_verified'] for t in report['trials'])})
if __name__=='__main__':main()
