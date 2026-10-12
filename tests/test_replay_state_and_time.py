"""R4/N1: legal paths, irreversible terminals and genuine cached RGB packets."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from hand_competition.closed_loop.experiments import run_trial,replay,code_identity
from hand_competition.closed_loop.contracts import Config
from hand_competition.closed_loop.controller import Controller
from hand_competition.closed_loop.artifacts import Trace,write_ppm,digest
from hand_competition.closed_loop.synthetic import SyntheticBackend
from replay_revision_fixtures import BufferedPackets,normalize


def custom(path,backend,config,version=4):
    manifest={'schema_version':version,'environment':backend.environment,'producer':backend.producer,
              'config':asdict(config),'cameras':[asdict(c) for c in backend.cameras],'code':code_identity()}
    return Controller(backend,config,Trace(path,manifest)).run()


class StateAndTimeTests(unittest.TestCase):
    def test_rehashed_phase_terminal_and_missing_carry_variants_fail_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for robust in (False,True):
                mode='robust' if robust else 'raw';base=root/mode
                run_trial(base,config=Config(robust_observations=robust))
                original=[json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
                for fault in ('missing-phases','wrong-release','early-stop','early-terminal','carried-image-hidden'):
                    path=root/(mode+'-'+fault);shutil.copytree(base,path);es=json.loads(json.dumps(original))
                    if fault=='missing-phases':
                        ids={e['payload']['action_id'] for e in es if e['kind']=='action_submitted' and e['payload']['phase'] in ('target_transit','target_descent','retreat')}
                        es=[e for e in es if not(e['kind'] in ('action_submitted','action_status') and e['payload']['action_id'] in ids)]
                        es[-1]['payload']['result']['action_count']-=len(ids)
                    elif fault=='wrong-release':
                        for e in es:
                            if e['kind']=='action_submitted' and e['payload']['phase']=='release':e['payload']['phase']='reset_empty_grasp'
                    elif fault=='early-stop':
                        stop=next(e for e in es if e['kind']=='stop_request_succeeded');es.remove(stop);stop['time_s']=es[0]['time_s'];es.insert(1,stop)
                    elif fault=='early-terminal':
                        es.insert(1,dict(es[0],kind='trial_stopped',payload={'category':'actuator_failure','detail':'injected terminal','deadline':None}))
                    else:
                        carrying=False
                        for e in es:
                            if e['kind']=='action_submitted' and e['payload']['phase']=='target_transit':carrying=True
                            elif carrying and e['kind']=='observation':
                                for f in e['payload']['frames']:
                                    file=path/f['file'];write_ppm(file,f['width'],f['height'],bytes((35,35,35))*f['width']*f['height']);f['file_sha256']=digest(file.read_bytes())
                                e['payload'].update(visual_position_m=None,stereo_residual_m=None,vision_error='missing/ambiguous colored block; no motion target produced');break
                    normalize(path,es) # preserve bundle/manifest validity, not just old event hashes
                    with self.subTest(mode=mode,fault=fault),self.assertRaises(ValueError):
                        replay(path,code_identity()['runtime_tree_sha256'])

    def test_stability_cannot_reuse_window_before_latest_reset_or_lie_about_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'valid';run_trial(base,config=Config())
            original=[json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
            for fault in ('new-window-old-success','wrong-anchor','wrong-stability-point'):
                path=root/fault;shutil.copytree(base,path);es=json.loads(json.dumps(original))
                if fault=='new-window-old-success':
                    index=next(i for i,e in enumerate(es) if e['kind']=='stability_verified')
                    obs=next(e for e in reversed(es[:index]) if e['kind']=='observation')
                    es[index:index]=[dict(obs,kind='settle_reset',payload={'reason':'sensor_gap'}),
                      dict(obs,kind='settle_started',payload={'anchor_m':obs['payload']['visual_position_m'],
                                                           'sensor_start_s':obs['payload']['sensor_time_s']})]
                elif fault=='wrong-anchor':next(e for e in es if e['kind']=='settle_started')['payload']['anchor_m'][0]+=.1
                else:next(e for e in es if e['kind']=='stability_verified')['payload']['position_m'][0]+=.1
                normalize(path,es)
                with self.subTest(fault=fault),self.assertRaises(ValueError):replay(path)

    def test_real_cached_packets_preserve_capture_times_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            for robust,version in ((False,2),(False,4),(True,4)):
                path=Path(directory)/f'{robust}-{version}'
                result=custom(path,BufferedPackets(),Config(robust_observations=robust),version)
                self.assertTrue(result['success'],result)
                checked=replay(path);self.assertTrue(checked['control_record_consistent']);self.assertTrue(checked['receipt_timestamps_recorded'])
                es=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
                obs=next(e for e in es if e['kind']=='observation');p=obs['payload']
                self.assertAlmostEqual(obs['time_s']-p['packet_time_s'],.2)
                self.assertLess(p['packet_time_s'],es[0]['time_s'])
                self.assertEqual(p['packet_time_s'],p['frames'][0]['captured_s'])
                self.assertEqual(p['received_s'],obs['time_s'])
                self.assertEqual(p['robot']['captured_s'],p['packet_time_s'])
                # Rehashing a plausible-looking receipt time cannot erase source age.
                bad=Path(directory)/f'bad-age-{robust}-{version}';shutil.copytree(path,bad)
                forged=json.loads(json.dumps(es));event=next(e for e in forged if e['kind']=='observation')
                event['payload']['frames'][0]['captured_s']-=.1
                normalize(bad,forged)
                with self.assertRaises(ValueError):replay(bad)

    def test_wall_deadline_has_recorded_basis_and_missing_basis_lowers_claim(self):
        class WallBackend(SyntheticBackend):
            wall=0.
            def observe(self):
                self.wall+=.1
                return super().observe()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);backend=WallBackend();config=Config(wall_timeout_s=.05)
            with patch('hand_competition.closed_loop.controller.time.monotonic',side_effect=lambda:backend.wall):
                result=custom(root/'valid',backend,config)
            self.assertEqual(result['terminal_failure'],'trial_timeout')
            self.assertLess(result['elapsed_s'],config.trial_timeout_s)
            self.assertTrue(replay(root/'valid')['failure_reason_reconstructed'])
            es=[json.loads(line) for line in (root/'valid/events.jsonl').read_text().splitlines()]
            bad=root/'unproved';shutil.copytree(root/'valid',bad)
            next(e for e in es if e['kind']=='trial_stopped')['payload']['deadline']=None
            normalize(bad,es)
            checked=replay(bad)
            self.assertTrue(checked['trace_verified']);self.assertFalse(checked['control_record_consistent'])
            self.assertFalse(checked['failure_reason_reconstructed']);self.assertTrue(checked['failure_reason_limitations'])
            contradictory=root/'late-footer';shutil.copytree(root/'valid',contradictory)
            events=[json.loads(line) for line in (contradictory/'events.jsonl').read_text().splitlines()]
            terminal=next(e for e in events if e['kind']=='trial_stopped')
            terminal['payload']['deadline'].update(backend_elapsed_s=30.,wall_elapsed_s=0.)
            for event in events:
                if event['kind'] in ('stop_request_succeeded','result'):event['time_s']=30.
            events[-1]['payload']['result']['elapsed_s']=30.
            normalize(contradictory,events)
            with self.assertRaisesRegex(ValueError,'matching budget'):replay(contradictory)

    def test_text_only_failure_never_claims_control_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'minimal';backend=SyntheticBackend();cfg=Config()
            manifest={'schema_version':4,'environment':backend.environment,'producer':backend.producer,
                      'config':asdict(cfg),'cameras':[asdict(c) for c in backend.cameras],'code':code_identity()}
            trace=Trace(path,manifest)
            trace.event('trial_started',0.,start_s=0.)
            trace.event('trial_stopped',.1,category='trial_timeout',detail='unsupported wall deadline',deadline=None)
            trace.event('stop_request_succeeded',.1)
            result={'success':False,'elapsed_s':.1,'action_count':0,'recoveries':0,'terminal_failure':'trial_timeout',
                    'stability_verified_s':0.,'physical_stop_verified':False,'stop_request_succeeded':True,
                    'official_platform':False,'official_score':None,'final_visual_position_m':None,'trace_complete':True}
            trace.finish(result,.1)
            checked=replay(path)
            self.assertTrue(checked['trace_verified']);self.assertTrue(checked['legal_path_reconstructed'])
            self.assertFalse(checked['control_record_consistent']);self.assertFalse(checked['failure_reason_reconstructed'])

    def test_action_wall_timeout_and_status_aged_feedback_are_recorded(self):
        class WallAction(SyntheticBackend):
            wall=0.
            def observe(self):
                if self._pending:self.wall+=2.
                return super().observe()
        class SlowStatus(SyntheticBackend):
            def status(self,action_id):
                self._time+=.3
                return super().status(action_id)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);backend=WallAction()
            with patch('hand_competition.closed_loop.controller.time.monotonic',side_effect=lambda:backend.wall):
                result=custom(root/'wall-action',backend,Config())
            self.assertEqual(result['terminal_failure'],'action_timeout')
            self.assertTrue(replay(root/'wall-action')['failure_reason_reconstructed'])
            result=custom(root/'slow-status',SlowStatus(),Config())
            self.assertEqual(result['terminal_failure'],'invalid_observation')
            self.assertEqual(result['action_count'],1)
            self.assertTrue(replay(root/'slow-status')['control_record_consistent'])

    def test_approach_displacement_uses_supported_recovery_path(self):
        class ApproachMoved(SyntheticBackend):
            shifted=False
            def _advance(self):
                pending=self._pending
                super()._advance()
                if pending and self._pending is None and self._moves==2 and not self.shifted:
                    self._block=(self._block[0]+.04,*self._block[1:]);self.shifted=True
        with tempfile.TemporaryDirectory() as directory:
            for robust in (False,True):
                path=Path(directory)/str(robust)
                result=custom(path,ApproachMoved(),Config(robust_observations=robust))
                self.assertTrue(result['success'],result);self.assertGreater(result['recoveries'],0)
                self.assertTrue(replay(path)['control_record_consistent'])

    def test_retry_cannot_reuse_an_old_grasp_as_current_attempt_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            for robust in (False,True):
                path=Path(directory)/str(robust)
                result=run_trial(path,'empty_grasp_once',config=Config(robust_observations=robust));self.assertTrue(result['success'])
                es=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
                closed=[e['payload']['action_id'] for e in es if e['kind']=='action_submitted' and e['payload']['phase'] in ('close','lift')]
                removed=set(closed[-2:])
                es=[e for e in es if not(e['kind'] in ('action_submitted','action_status') and e['payload']['action_id'] in removed)]
                es[-1]['payload']['result']['action_count']-=2;normalize(path,es)
                with self.assertRaises(ValueError):replay(path)


if __name__=='__main__':unittest.main()
