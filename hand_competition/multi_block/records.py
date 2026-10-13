"""Hash-bound RGB planning records; no action or scoring events are accepted."""
from dataclasses import asdict
from pathlib import Path
from ..closed_loop.artifacts import canonical,digest,load_json,decode_json,write_ppm,read_ppm
from ..closed_loop.geometry import Camera,finite
from .planning import PlanConfig
from .perception import VisionConfig
from .session import PlanningSession


class PlanningTrace:
    def __init__(self,directory,manifest):
        if type(manifest.get('schema_version')) is not int or manifest.get('schema_version')!=2:
            raise ValueError('planning trace requires post-MB schema2')
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=False);(self.root/'frames').mkdir()
        (self.root/'manifest.json').write_text(canonical(manifest)+'\n')
        self.bound=digest((self.root/'manifest.json').read_bytes());self.previous='0'*64;self.index=0;self.stopped=False
        self.stream=(self.root/'events.jsonl').open('w');self.frame_count=0
    def event(self,kind,time_s,payload):
        if self.stopped:raise ValueError('record finalized')
        e={'index':self.index,'kind':kind,'time_s':time_s,'payload':payload,'previous_sha256':self.previous,'manifest_sha256':self.bound}
        e['sha256']=digest(canonical(e).encode());self.stream.write(canonical(e)+'\n');self.stream.flush()
        self.previous=e['sha256'];self.index+=1
    def observation(self,raw,received_s,snapshot):
        packet={k:v for k,v in raw.items() if k!='frames'};packet['frames']=[]
        for frame in raw['frames']:
            name=f'frames/{self.frame_count:05}-{frame["camera_id"]}.ppm';path=self.root/name
            write_ppm(path,frame['width'],frame['height'],frame['rgb'])
            saved={k:v for k,v in frame.items() if k!='rgb'};saved.update(file=name,file_sha256=digest(path.read_bytes()))
            packet['frames'].append(saved)
        self.event('scene_observation',received_s,{'packet':packet,'derived':snapshot});self.frame_count+=1
    def finish(self,time_s,result):
        self.event('session_stop',time_s,{'reason':result['stop_reason']})
        self.event('result',time_s,result)
        (self.root/'result.json').write_text(canonical(result)+'\n');self.stopped=True;self.stream.close()


def replay(directory):
    root=Path(directory);manifest=load_json(root/'manifest.json')
    if type(manifest.get('schema_version')) is not int or manifest.get('schema_version')!=2 or manifest.get('feedback_level')!='offline_multi_block_planning' or manifest.get('official_platform') is not False:
        raise ValueError('unsupported planning record')
    config=PlanConfig.from_dict(manifest['planning']);vision=VisionConfig(**manifest['vision'])
    session=PlanningSession(tuple(Camera(**c) for c in manifest['cameras']),manifest['producer'],config,vision)
    bound=digest((root/'manifest.json').read_bytes());previous='0'*64;last=-1.;stopped=False;final=None;observations=0;rejects=0;stop_reason=None
    for i,line in enumerate((root/'events.jsonl').read_text().splitlines()):
        e=decode_json(line)
        if final is not None:raise ValueError('event after result')
        if not isinstance(e,dict):raise ValueError('planning event object')
        if set(e)!={'index','kind','time_s','payload','previous_sha256','manifest_sha256','sha256'}:raise ValueError('planning event fields')
        sha=e.pop('sha256')
        if type(e['index']) is not int or e['index']!=i or e['previous_sha256']!=previous or e['manifest_sha256']!=bound or digest(canonical(e).encode())!=sha:raise ValueError('planning chain/binding mismatch')
        previous=sha;stamp=e['time_s'];kind=e['kind'];p=e['payload']
        if not finite(stamp) or stamp<0 or stamp<last:raise ValueError('planning time ordering')
        last=stamp
        if session.terminated and not stopped and kind!='session_stop':raise ValueError('event after planning rejection terminal')
        if stopped and kind!='result':raise ValueError('event after irreversible planning stop')
        if kind=='scene_observation':
            if set(p)!={'packet','derived'}:raise ValueError('scene payload fields')
            raw={**p['packet'],'frames':[]}
            for frame in p['packet']['frames']:
                path=(root/frame['file']).resolve()
                if not path.is_relative_to(root.resolve()) or digest(path.read_bytes())!=frame['file_sha256']:raise ValueError('RGB path/hash')
                width,height,rgb=read_ppm(path)
                if (width,height)!=(frame['width'],frame['height']):raise ValueError('RGB size')
                restored={k:v for k,v in frame.items() if k not in ('file','file_sha256')};restored['rgb']=rgb;raw['frames'].append(restored)
            _,snapshot=session.observe(raw,stamp)
            if canonical(snapshot)!=canonical(p['derived']):raise ValueError('derived planning record differs from RGB recomputation')
            observations+=1
        elif kind=='observation_rejected':
            if set(p)!={'error_type'} or not isinstance(p['error_type'],str):raise ValueError('rejection payload')
            # Rejected raw fields are not saved; this cannot reconstruct cause.
            session.terminated=True;rejects+=1
        elif kind=='session_stop':
            if set(p)!={'reason'} or p['reason'] not in ('fixture_complete','invalid_observation','planning_deadline'):raise ValueError('stop fields')
            if not session.terminated:session.stop()
            stopped=True;stop_reason=p['reason']
        elif kind=='result':
            if not stopped or p!=load_json(root/'result.json'):raise ValueError('result/stop mismatch')
            if set(p)!={'feedback_level','official_platform','official_score','robot_actions_executed','observations','stop_reason','geometric_goal_count','physical_support_verified','sensor_authenticity_verified'}:raise ValueError('planning result fields')
            if (p['feedback_level']!='offline_multi_block_planning' or p['official_platform'] is not False or p['official_score'] is not None
                or type(p['robot_actions_executed']) is not int or p['robot_actions_executed']!=0 or p['physical_support_verified'] is not False or p['sensor_authenticity_verified'] is not False
                or type(p['observations']) is not int or p['observations']!=observations or p['stop_reason']!=stop_reason or (stop_reason!='fixture_complete' and not rejects)):raise ValueError('invalid planning result claim')
            current=0 if not observations else len(snapshot['decision']['geometrically_supported_slots'])
            if type(p['geometric_goal_count']) is not int or p['geometric_goal_count']!=current:raise ValueError('final geometry count')
            final=p
        else:raise ValueError('unknown planning event / robot action not supported')
    if final is None:raise ValueError('missing final planning result')
    return {'record_verified':True,'rgb_planning_recomputed':observations,'deterministic_production_recompute':True,
            'independent_algorithm_correctness_verified':False,'rejection_reason_reconstructed':False if rejects else None,
            'robot_actions_executed':0,'official_score':None,'physical_support_verified':False,'sensor_authenticity_verified':False}
