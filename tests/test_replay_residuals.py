"""C1/C2: all status ages and failure time rather than cleanup footer time."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from hand_competition.closed_loop.artifacts import Trace
from hand_competition.closed_loop.contracts import Config,ObservationGate,ActionStatus
from hand_competition.closed_loop.experiments import run_trial,replay,code_identity
from hand_competition.closed_loop.synthetic import SyntheticBackend
from hand_competition.closed_loop.vision import estimate_single_block
from test_replay_state_and_time import custom
from replay_revision_fixtures import normalize


def delay_first_pending(events,age,state='pending'):
    at=next(i for i,e in enumerate(events) if e['kind']=='action_status' and e['payload']['state']=='pending')
    # Keep source/capture times; later receipts and requests move coherently.
    events[at]['time_s']+=age;events[at]['payload']['state']=state
    shift=age-.1
    for e in events[at+1:]:
        e['time_s']+=shift
        if e['kind']=='observation':e['payload']['received_s']+=shift
        if e['kind']=='action_submitted':e['payload']['requested_s']+=shift
    events[-1]['payload']['result']['elapsed_s']+=shift


class ResidualReplayTests(unittest.TestCase):
    def test_every_status_rejects_stale_source_and_boundary_pending_remains_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for robust in (False,True):
                original=root/str(robust);run_trial(original,config=Config(robust_observations=robust))
                base=[json.loads(line) for line in (original/'events.jsonl').read_text().splitlines()]
                for state in ('pending','failed','complete'):
                    path=root/(str(robust)+state);shutil.copytree(original,path);events=json.loads(json.dumps(base))
                    delay_first_pending(events,.3,state);normalize(path,events)
                    with self.subTest(robust=robust,state=state),self.assertRaisesRegex(ValueError,'status without fresh measured feedback'):
                        replay(path,code_identity()['runtime_tree_sha256'])
                path=root/(str(robust)+'boundary');shutil.copytree(original,path);events=json.loads(json.dumps(base))
                delay_first_pending(events,.25);normalize(path,events)
                checked=replay(path);self.assertTrue(checked['recorded_success']);self.assertTrue(checked['control_record_consistent'])

    def test_online_all_late_states_stop_before_status_is_logged(self):
        class LateStatus(SyntheticBackend):
            returned_state='pending'
            def status(self,action_id):
                self._time+=.3
                return ActionStatus(self.returned_state)
        with tempfile.TemporaryDirectory() as directory:
            for robust in (False,True):
                for state in ('pending','failed','complete'):
                    backend=LateStatus();backend.returned_state=state;path=Path(directory)/(str(robust)+state)
                    result=custom(path,backend,Config(robust_observations=robust))
                    self.assertFalse(result['success']);self.assertEqual(result['terminal_failure'],'invalid_observation')
                    self.assertEqual(result['action_count'],1);self.assertTrue(replay(path)['trace_verified'])
                    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
                    self.assertFalse(any(e['kind']=='action_status' for e in events))

    def test_cleanup_footer_cannot_prove_early_timeout_but_real_failure_time_can(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for footer in (.1,30.):
                backend=SyntheticBackend();cfg=Config();path=root/str(footer)
                manifest={'schema_version':4,'environment':backend.environment,'producer':backend.producer,
                          'config':asdict(cfg),'cameras':[asdict(c) for c in backend.cameras],'code':code_identity()}
                trace=Trace(path,manifest);trace.producer=backend.producer;trace.event('trial_started',0.,start_s=0.)
                raw=backend.observe();observation=ObservationGate(backend.cameras,backend.producer).accept(raw,backend.now())
                visual=estimate_single_block(observation,backend.cameras);trace.observation(observation,visual,received_s=backend.now())
                trace.event('trial_stopped',.1,category='trial_timeout',detail='no recorded timeout basis',deadline=None)
                trace.event('stop_request_succeeded',footer)
                result={'success':False,'elapsed_s':footer,'action_count':0,'recoveries':0,'terminal_failure':'trial_timeout',
                        'stability_verified_s':0.,'physical_stop_verified':False,'stop_request_succeeded':True,
                        'official_platform':False,'official_score':None,'final_visual_position_m':visual.position_m,'trace_complete':True}
                trace.finish(result,footer);checked=replay(path)
                self.assertTrue(checked['trace_verified']);self.assertFalse(checked['failure_reason_reconstructed'])
                self.assertFalse(checked['control_record_consistent'])
            path=root/'real-timeout';run_trial(path,config=Config(trial_timeout_s=.2))
            events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
            next(e for e in events if e['kind']=='trial_stopped')['payload']['deadline']=None
            for e in events:
                if e['kind'] in ('stop_request_succeeded','result'):e['time_s']+=.1
            events[-1]['payload']['result']['elapsed_s']+=.1
            normalize(path,events)
            checked=replay(path);self.assertTrue(checked['failure_reason_reconstructed']);self.assertTrue(checked['control_record_consistent'])


if __name__=='__main__':unittest.main()
