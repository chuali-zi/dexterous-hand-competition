"""Reviewer-supplied buffered packet fixture and evidence-preserving mutations."""
import json,sys,shutil
from pathlib import Path
from dataclasses import asdict

from hand_competition.closed_loop.experiments import run_trial,replay,code_identity
from hand_competition.closed_loop.contracts import Config,ActionStatus
from hand_competition.closed_loop.artifacts import Trace,canonical,digest,write_ppm
from hand_competition.closed_loop.controller import Controller
from hand_competition.closed_loop.synthetic import SyntheticBackend

class BufferedPackets(SyntheticBackend):
 def __init__(self):
  super().__init__();self._buffer=[];self._delivered=-1
  self._buffer.append(super().observe());self._buffer.append(super().observe())
 def _advance(self):
  pending=self._pending;super()._advance()
  if pending and self._pending is None:self._actions[pending]['actual_completion_s']=self._time
 def observe(self):
  self._buffer.append(super().observe());raw=self._buffer.pop(0);self._delivered=raw['time_s'];return raw
 def status(self,action_id):
  record=self._actions[action_id]
  if record['status'].state=='complete' and self._delivered<record['actual_completion_s']:return ActionStatus('pending')
  return record['status']

def normalize(path,events):
 mapping={e['index']:i for i,e in enumerate(events) if e['kind']=='robust_observation'}
 previous='0'*64
 for i,e in enumerate(events):
  e.pop('sha256',None);e['index']=i;e['previous_sha256']=previous
  p=e['payload']
  for k in ('estimate_id','anchor_estimate_id'):
   if k in p:p[k]=mapping.get(p[k],p[k])
  if e['kind']=='result' and 'final_robust_estimate_id' in p['result']:
   p['result']['final_robust_estimate_id']=mapping.get(p['result']['final_robust_estimate_id'],p['result']['final_robust_estimate_id'])
  e['sha256']=digest(canonical(e).encode());previous=e['sha256']
 (path/'events.jsonl').write_text(''.join(canonical(e)+'\n' for e in events))
 (path/'result.json').write_text(json.dumps(events[-1]['payload']['result']))

