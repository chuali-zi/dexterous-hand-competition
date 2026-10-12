"""M1: true trajectories, time-resolved observations and no mean certificates."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from hand_competition.closed_loop.contracts import Config,Frame,Observation
from hand_competition.closed_loop.artifacts import read_ppm,write_ppm,digest
from hand_competition.closed_loop.geometry import Camera
from hand_competition.closed_loop.vision import estimate_single_block
from hand_competition.closed_loop.experiments import run_trial,replay
from scripts.motion_fixtures import (AlternatingPlacement,Alternating16,PeriodicPlacement,SlowDriftPlacement,
    evaluated_backend,evaluate_window)
from test_replay_state_and_time import custom
from replay_revision_fixtures import normalize


class MotionStabilityTests(unittest.TestCase):
    def test_all_visible_motion_patterns_fail_both_modes_and_full_records_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            for cls in (Alternating16,AlternatingPlacement,PeriodicPlacement,SlowDriftPlacement):
                for robust in (False,True):
                    with self.subTest(motion=cls.__name__,robust=robust):
                        backend=evaluated_backend(cls);cfg=Config(robust_observations=robust)
                        path=Path(directory)/(cls.__name__+str(robust))
                        result=custom(path,backend,cfg,version=5)
                        self.assertFalse(result['success']);self.assertEqual(result['stability_verified_s'],0.)
                        self.assertFalse(result['physical_stability_verified'])
                        self.assertTrue(replay(path)['trace_verified'])
                        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
                        checked=evaluate_window(backend.evaluation_history,result,events,cfg)
                        self.assertGreater(checked['post_release_withdrawn_samples'],0)
                        self.assertGreater(checked['post_release_withdrawn_diameter_m'],.015)
                        self.assertFalse(checked['false_accept_under_evaluator_checks'])

    def test_full_window_truth_rejects_motion_even_when_last_point_at_goal(self):
        cfg=Config();history=[]
        for i in range(31):
            offset=(.020 if i%2 else -.020) if i<30 else 0.
            history.append({'time_s':i*.1,'position_m':(.16+offset,.42,.025),
                            'tip_m':(.16,.42,.2),'held':False,'released_at_s':0.})
        events=[{'kind':'stability_verified','payload':{'sensor_start_s':0.,'sensor_end_s':3.}}]
        checked=evaluate_window(history,{'success':True},events,cfg)
        self.assertTrue(checked['successful_window_checked']);self.assertTrue(checked['false_accept_under_evaluator_checks'])
        self.assertIn('whole_window_motion',checked['violation_categories'])
        self.assertAlmostEqual(checked['window']['diameter_m'],.040)
        self.assertLess(checked['window']['max_goal_error_m'],cfg.goal_tolerance_m)
        self.assertTrue(evaluate_window([],{'success':True},events,cfg)['false_accept_under_evaluator_checks'])

    def test_truth_window_checks_interior_support_height_and_clearance(self):
        cfg=Config();events=[{'kind':'stability_verified','payload':{'sensor_start_s':0.,'sensor_end_s':3.}}]
        base=[{'time_s':i*.1,'position_m':cfg.target,'tip_m':(.16,.42,.2),'held':False,'released_at_s':0.} for i in range(31)]
        self.assertFalse(evaluate_window(base,{'success':True},events,cfg)['false_accept_under_evaluator_checks'])
        for fault in ('held','height','clearance','gap'):
            history=json.loads(json.dumps(base))
            if fault=='held':history[15]['held']=True
            elif fault=='height':history[15]['position_m'][2]=.20
            elif fault=='clearance':history[15]['tip_m']=history[15]['position_m']
            else:history=history[:10]+history[15:]
            checked=evaluate_window(history,{'success':True},events,cfg)
            with self.subTest(fault=fault):self.assertTrue(checked['false_accept_under_evaluator_checks'])

    def test_evaluator_wrapper_preserves_the_control_input_whitelist(self):
        backend=evaluated_backend(AlternatingPlacement);raw=backend.observe()
        self.assertEqual(set(raw),{'schema_version','producer','calibration_id','time_s','frames','robot'})
        self.assertGreater(len(backend.evaluation_history),0)
        self.assertNotIn('held',raw['robot']);self.assertNotIn('position_m',raw)

    def test_rehashed_primary_window_motion_is_rejected_for_both_modes(self):
        """Preserve all action/grasp/location evidence; only change stable RGB."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for robust in (False,True):
                path=root/str(robust);result=run_trial(path,config=Config(robust_observations=robust))
                self.assertTrue(result['success'])
                manifest=json.loads((path/'manifest.json').read_text());cameras=tuple(Camera(**c) for c in manifest['cameras'])
                events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
                stable=next(e for e in events if e['kind']=='stability_verified');start=stable['payload']['sensor_start_s']
                for index,event in enumerate(events):
                    if event['kind']!='observation' or event['payload']['sensor_time_s']<=start:continue
                    point=(.16+(.020 if index%2 else -.020),.42,.025);frames=[]
                    for f in event['payload']['frames']:
                        camera=next(c for c in cameras if c.name==f['camera_id'])
                        pixels=bytearray((35,35,35))*camera.width*camera.height
                        u,v=map(round,camera.project(point))
                        for y in range(v-4,v+5):
                            for x in range(u-5,u+6):pixels[3*(y*camera.width+x):3*(y*camera.width+x)+3]=bytes((230,30,30))
                        file=path/f['file'];write_ppm(file,camera.width,camera.height,bytes(pixels));f['file_sha256']=digest(file.read_bytes())
                        frames.append(Frame(camera.name,f['frame_id'],f['sequence'],f['captured_s'],camera.width,camera.height,bytes(pixels)))
                    observation=Observation(event['payload']['packet_time_s'],tuple(frames),tuple(event['payload']['robot']['tip_position_m']),manifest['cameras'][0]['calibration_id'])
                    estimate=estimate_single_block(observation,cameras)
                    event['payload'].update(visual_position_m=estimate.position_m,stereo_residual_m=estimate.stereo_residual_m,vision_error=None)
                last=next(e for e in reversed(events) if e['kind']=='observation')
                stable['payload']['position_m']=last['payload']['visual_position_m']
                events[-1]['payload']['result']['final_visual_position_m']=last['payload']['visual_position_m']
                normalize(path,events)
                with self.assertRaisesRegex(ValueError,'stability sample|raw RGB change'):replay(path)


if __name__=='__main__':unittest.main()
