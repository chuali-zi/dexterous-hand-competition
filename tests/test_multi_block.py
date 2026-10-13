"""Independent analytic correspondence, all dependency subsets and failures."""
from dataclasses import replace,asdict
from itertools import combinations,permutations
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from hand_competition.closed_loop.contracts import ObservationGate
from hand_competition.closed_loop.vision import Detection
from hand_competition.closed_loop.geometry import distance
from hand_competition.multi_block.perception import VisionConfig,BlockObservation,associate,observe_many
from hand_competition.multi_block.tracking import Tracker,TrackView
from hand_competition.multi_block.planning import PlanConfig,build_slots,plan
from hand_competition.multi_block.session import PlanningSession
from hand_competition.multi_block.records import PlanningTrace,replay
from hand_competition.closed_loop.artifacts import canonical,digest
from scripts.multi_block_fixtures import CAMERAS,packet,initial_scene,pixels_for,goal_scene


def block(x=0.,y=0.,color='red',pixels=((1.,2.),(3.,4.))):
    return BlockObservation(color,(x,y,.025),pixels,('camera_left','camera_right'),0.)


def view(identity,color,center,stable=3.1):
    return TrackView(identity,color,center,((1.,2.),(3.,4.)),('camera_left','camera_right'),10.,4,10.-stable,True)


def recanonicalize(path,events):
    previous='0'*64
    for i,event in enumerate(events):
        event.pop('sha256',None);event['index']=i;event['previous_sha256']=previous
        event['sha256']=digest(canonical(event).encode());previous=event['sha256']
    (path/'events.jsonl').write_text(''.join(canonical(e)+'\n' for e in events))
    result=next(e['payload'] for e in reversed(events) if e['kind']=='result')
    (path/'result.json').write_text(canonical(result)+'\n')


class MultiPerceptionTests(unittest.TestCase):
    def test_closed_form_pixels_recover_independent_points(self):
        points=[(-.22,-.12,.025),(-.14,.07,.025),(.17,0.,.175)]
        colors=('red','red','green')
        detections=tuple(tuple(Detection(c,pixels_for(p,i),121) for p,c in zip(points,colors)) for i in range(2))
        result=associate(detections,CAMERAS,VisionConfig())
        self.assertFalse(result.issues);self.assertEqual(len(result.blocks),3)
        for expected,c in zip(points,colors):
            self.assertTrue(any(b.color==c and distance(b.center_m,expected)<1e-10 for b in result.blocks))

    def test_bijection_is_order_independent_and_ambiguous_color_is_discarded(self):
        points=[(-.05,0.,.025),(.05,0.,.025)]
        detections=tuple(tuple(Detection('red',pixels_for(p,i),121) for p in points) for i in range(2))
        narrow=associate(detections,CAMERAS,VisionConfig())
        reordered=associate(tuple(tuple(reversed(v)) for v in detections),CAMERAS,VisionConfig())
        self.assertEqual(narrow,reordered);self.assertEqual(len(narrow.blocks),2)
        broad=associate(detections,CAMERAS,VisionConfig(workspace=(-.4,.4,-.25,.25,-1.,.9)))
        self.assertEqual(broad.blocks,());self.assertIn('red:stereo_ambiguous',broad.issues)

    def test_one_view_occlusion_blocks_only_affected_color(self):
        detections=((Detection('red',pixels_for((-.2,0,.025),0),121),Detection('green',pixels_for((.1,.1,.025),0),121)),
                    (Detection('green',pixels_for((.1,.1,.025),1),121),))
        result=associate(detections,CAMERAS,VisionConfig())
        self.assertEqual([b.color for b in result.blocks],['green']);self.assertIn('red:view_count_mismatch',result.issues)

    def test_ten_block_rgb_has_no_hidden_ids_or_pose_inputs(self):
        raw=packet(initial_scene(91),1,.1);obs=ObservationGate(CAMERAS,raw['producer'],tip_bounds=(-.4,.4,-.25,.25,0,.4)).accept(raw,.1)
        found=observe_many(obs,CAMERAS)
        self.assertEqual(len(found.blocks),10);self.assertFalse(found.issues)
        self.assertEqual(set(raw),{'schema_version','producer','calibration_id','time_s','frames','robot'})
        for forbidden in ('object_id','object_pose','hidden_label'):
            self.assertNotIn(forbidden,raw)
        self.assertEqual(sorted(b.color for b in found.blocks),sorted(['red']*4+['yellow']*3+['blue']*2+['green']))

    def test_invalid_parameters_and_limits_fail_closed(self):
        for kwargs in ({'max_same_color':5},{'max_objects':True},{'max_residual_m':float('nan')}):
            with self.assertRaises(ValueError):VisionConfig(**kwargs)
        many=tuple(Detection('red',(float(i),1.),20) for i in range(11))
        self.assertEqual(associate((many,many),CAMERAS,VisionConfig()).blocks,())


class LocalTrackTests(unittest.TestCase):
    def test_ids_follow_small_observed_motion_but_witness_resets(self):
        tracker=Tracker();first,_,_=tracker.update((block(-.2),),.1,(0,.2,.3))
        current,_,_=tracker.update((block(-.19,pixels=((2.,2.),(4.,4.))),),.2,(0,.2,.3))
        self.assertEqual(first[0].track_id,current[0].track_id);self.assertEqual(current[0].unchanged_for_s,0.)
        # Same centroids with a different supplied pose still cannot extend witness.
        current,_,_=tracker.update((block(-.185,pixels=((2.,2.),(4.,4.))),),.3,(0,.2,.3))
        self.assertEqual(current[0].unchanged_for_s,0.)

    def test_missing_reappearing_and_gap_never_inherit_identity(self):
        tracker=Tracker();first,_,_=tracker.update((block(-.2),),.1,(0,.2,.3))
        empty,_,retired=tracker.update((),.2,(0,.2,.3));self.assertIn(first[0].track_id,retired)
        next_frame,_,_=tracker.update((block(-.2),),.3,(0,.2,.3));self.assertNotEqual(first[0].track_id,next_frame[0].track_id)
        after_gap,_,_=tracker.update((block(-.2),),1.,(0,.2,.3));self.assertNotEqual(next_frame[0].track_id,after_gap[0].track_id)
        with self.assertRaises(ValueError):tracker.update((),1.,(0,.2,.3))

    def test_temporal_ambiguity_retires_instead_of_swapping_ids(self):
        tracker=Tracker();first,_,_=tracker.update((block(-.02),block(.02)),.1,(0,.2,.3))
        current,issues,retired=tracker.update((block(0.),),.2,(0,.2,.3))
        self.assertEqual(current,());self.assertIn('red:temporal_ambiguity',issues);self.assertEqual(set(retired),{t.track_id for t in first})

    def test_order_invariance_and_hand_clearance(self):
        a=Tracker();b=Tracker();items=(block(-.2),block(-.1,color='blue'))
        self.assertEqual(a.update(items,.1,(0,.2,.3)),b.update(tuple(reversed(items)),.1,(0,.2,.3)))
        near=Tracker();values,_,_=near.update((block(),),.1,(0.,0.,.025));self.assertFalse(values[0].robot_clear)


class DependencyPlannerTests(unittest.TestCase):
    def test_independent_pyramid_coordinates_and_color_inventory(self):
        slots=build_slots(PlanConfig());truth=goal_scene()
        self.assertEqual(len(slots),10)
        for slot,item in zip(slots,truth):self.assertEqual(slot.color,item['color']);self.assertLess(distance(slot.center_m,item['center']),1e-12)
        self.assertEqual(slots[4].supports,('layer-0-slot-0','layer-0-slot-1'))
        self.assertEqual(slots[-1].supports,('layer-2-slot-0','layer-2-slot-1'))

    def test_all_1024_presence_subsets_have_exact_dependency_closure(self):
        cfg=PlanConfig();slots=build_slots(cfg)
        # Oracle uses recursive ancestor membership rather than production scan.
        def ancestors(i):
            if slots[i].layer==0:return set()
            parents={next(j for j,s in enumerate(slots) if s.slot_id==name) for name in slots[i].supports}
            return parents|set().union(*(ancestors(j) for j in parents))
        for mask in range(1<<10):
            present={i for i in range(10) if mask&(1<<i)}
            tracks=[view('unit-'+str(i),s.color,s.center_m) for i,s in enumerate(slots) if i in present]
            result=plan(tracks,cfg)
            expected={slots[i].slot_id for i in present if ancestors(i)<=present}
            self.assertEqual(set(result['geometrically_supported_slots']),expected)
            self.assertIsNone(result['official_score']);self.assertEqual(result['robot_actions_executed'],0)

    def test_base_color_selection_and_support_failures_replan(self):
        cfg=PlanConfig();slots=build_slots(cfg)
        supply=[view('source-red','red',(-.2,-.1,.025)),view('source-yellow','yellow',(-.2,.1,.025))]
        first=plan(supply,cfg);self.assertEqual(first['candidate']['target_slot_id'],slots[0].slot_id)
        base=[view(str(i),s.color,s.center_m) for i,s in enumerate(slots[:4])]
        next_task=plan(base+supply,cfg);self.assertEqual(next_task['candidate']['target_slot_id'],slots[4].slot_id)
        all_tracks=[view(str(i),s.color,s.center_m) for i,s in enumerate(slots)]
        complete=plan(all_tracks,cfg);self.assertEqual(complete['kind'],'geometric_goal_observed')
        broken=plan(all_tracks[1:]+supply,cfg)
        self.assertNotIn(slots[-1].slot_id,broken['geometrically_supported_slots'])
        self.assertIsNone(broken['candidate']);self.assertEqual(broken['blocked'][slots[0].slot_id],'occupied_target_or_descent_column')
        # Clearing all descendants opens a repair candidate while preserving others.
        removed={0,4,7,9};remaining=[t for i,t in enumerate(all_tracks) if i not in removed]
        repair=plan(remaining+supply,cfg);self.assertEqual(repair['candidate']['target_slot_id'],slots[0].slot_id)

    def test_wrong_color_boundaries_and_close_supplies_never_get_candidates(self):
        cfg=PlanConfig();slots=build_slots(cfg)
        wrong=plan([view('wrong','blue',slots[0].center_m)],cfg)
        self.assertEqual(wrong['blocked'][slots[0].slot_id],'wrong_color_occupied')
        for source in (view('boundary','red',(-.34,0,.025)),view('unstable','red',(-.2,0,.025),.5)):
            self.assertIsNone(plan([source],cfg)['candidate'])
        crowd=[view('a','red',(-.2,0,.025)),view('b','blue',(-.18,0,.025))]
        self.assertIsNone(plan(crowd,cfg)['candidate'])

    def test_invalid_blueprint_assumptions_rejected(self):
        for kwargs in ({'official_scene_confirmed':True},{'pitch_x_m':.04},{'layer_counts':(4,2,1)},
                       {'layer_counts':None},{'target_xy_tolerance_m':.04},{'base_anchor_m':(.33,0,.025)},
                       {'build_xy':(-.3,.34,-.2,.2)},{'min_support_fraction':.99}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):PlanConfig(**kwargs)


class PlanningLifecycleTests(unittest.TestCase):
    def test_invalid_observation_and_deadline_are_irreversible(self):
        for fault in ('truth','repeat','deadline'):
            cfg=PlanConfig(session_budget_s=.2 if fault=='deadline' else 300.)
            session=PlanningSession(CAMERAS,'offline_multi_rgb_fixture_v1',cfg)
            raw=packet(initial_scene(),1,.1);session.observe(raw,.1)
            bad=packet(initial_scene(),2,.3 if fault=='deadline' else .2)
            if fault=='truth':bad['object_id']='fixture-truth'
            elif fault=='repeat':bad['frames'][0]['sequence']=1
            with self.assertRaises(ValueError):session.observe(bad,bad['time_s'])
            with self.assertRaises(ValueError):session.observe(packet(initial_scene(),3,.4),.4)
            with self.assertRaises(NotImplementedError):session.execute_candidate({})

    def test_planning_record_recompute_and_terminal_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'valid';cfg=PlanConfig();vc=VisionConfig()
            manifest={'schema_version':1,'feedback_level':'offline_multi_block_planning','official_platform':False,
                      'planning':asdict(cfg),'vision':asdict(vc),'cameras':[asdict(c) for c in CAMERAS],'producer':'offline_multi_rgb_fixture_v1'}
            session=PlanningSession(CAMERAS,manifest['producer'],cfg,vc);trace=PlanningTrace(base,manifest)
            for i in (1,2):
                raw=packet(initial_scene(),i,i*.1);_,snapshot=session.observe(raw,i*.1);trace.observation(raw,i*.1,snapshot)
            session.stop();result={'feedback_level':'offline_multi_block_planning','official_platform':False,'official_score':None,
              'robot_actions_executed':0,'observations':2,'stop_reason':'fixture_complete','geometric_goal_count':0,
              'physical_support_verified':False,'sensor_authenticity_verified':False};trace.finish(.2,result)
            self.assertTrue(replay(base)['record_verified'])
            original=[json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
            for fault in ('derived','early-stop','trailing','score','action','missing-result'):
                path=root/fault;shutil.copytree(base,path);events=json.loads(json.dumps(original))
                if fault=='derived':events[0]['payload']['derived']['decision']['official_score']=50
                elif fault=='early-stop':events.insert(0,events.pop(-2));events[0]['time_s']=.1
                elif fault=='trailing':events.append(events[-1])
                elif fault=='score':events[-1]['payload']['official_score']=50
                elif fault=='action':events[0]['kind']='robot_action'
                else:events=events[:-1]
                if fault=='missing-result':
                    previous='0'*64
                    for i,e in enumerate(events):
                        e.pop('sha256');e['index']=i;e['previous_sha256']=previous;e['sha256']=digest(canonical(e).encode());previous=e['sha256']
                    (path/'events.jsonl').write_text(''.join(canonical(e)+'\n' for e in events))
                else:recanonicalize(path,events)
                with self.subTest(fault=fault),self.assertRaises(ValueError):replay(path)
