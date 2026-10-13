#!/usr/bin/env python3
"""Audit existing denied planning execution and local assets; no SDK execution."""
from pathlib import Path
import concurrent.futures
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,sys,urllib.request,urllib.error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from hand_competition.closed_loop.experiments import code_identity
from hand_competition.closed_loop.preflight import environment_report
from hand_competition.multi_block.session import PlanningSession
from test_multi_block_review import calibration,reviewer_packet
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--check-public-sources',action='store_true')
args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
identity=code_identity()
assert identity['runtime_tree_sha256']=='113ce5c0792124237cffaa52067358c91c7e13e346d8626536bda9e192ca274f'
callbacks=[];denials=[]
def fake_executor(*args,**kwargs): callbacks.append('unexpected_callback')
def attempt(name,session,payload):
    before=len(callbacks)
    try:session.execute_candidate(payload,executor=fake_executor)
    except NotImplementedError as error:
        denials.append({'case':name,'outcome':'denied','exception':'NotImplementedError','reason':str(error),
                        'local_callback_invocations':len(callbacks)-before,'session_terminated':session.terminated})
    else:raise AssertionError('execution request was not denied: '+name)
    assert len(callbacks)==before
cameras=calibration();session=PlanningSession(cameras,'reviewer_rgb')
attempt('before_any_observation',session,None)
for seq in range(1,33):
    raw=reviewer_packet([('red',(-.2,-.12,.025))],seq,cameras)
    _,snapshot=session.observe(raw,raw['time_s'])
assert snapshot['decision']['candidate'] is not None
assert snapshot['decision']['correspondence_authenticity_verified'] is False
assert snapshot['decision']['physical_component_completeness_verified'] is False
assert snapshot['decision']['execution_available'] is False
attempt('live_conditional_candidate',session,snapshot['decision'])
forged=json.loads(json.dumps(snapshot['decision']))
for flag in ('correspondence_authenticity_verified','physical_component_completeness_verified','execution_available','physical_support_verified','robot_collision_verified','transport_provenance_verified'):forged[flag]=True
attempt('forged_all_verification_flags_true',session,forged)
session.stop();attempt('after_explicit_stop',session,forged)
bad_session=PlanningSession(cameras,'reviewer_rgb');raw=reviewer_packet([('red',(-.2,-.12,.025))],1,cameras);bad_session.observe(raw,.1)
try:bad_session.observe(raw,.2)
except ValueError:pass
else:raise AssertionError('repeated source sequence accepted')
assert bad_session.terminated
attempt('after_input_rejection_terminal',bad_session,forged)
assert not callbacks
sdk_names=('rabo_robolib','rabo_robocap','rclpy')
assert all(name not in sys.modules for name in sdk_names)
paths=subprocess.check_output(['git','ls-tree','-r','--name-only','86e12ae'],cwd=ROOT,text=True).splitlines()
all_reviewed=[p for p in paths if p.split('/')[0] in ('hand_competition','configs','scripts','tests')]
for p in all_reviewed:assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show','86e12ae:'+p],cwd=ROOT)
search=subprocess.run(['rg','--files',str(ROOT.parent),'-g','*agent_system*','-g','*rabo*','-g','*.urdf','-g','*.xacro','-g','*.bag','-g','*.mcap','-g','camera_info*','-g','package.xml','-g','metadata.yaml'],cwd=ROOT,text=True,capture_output=True)
assert search.returncode in (0,1)
urls=['https://robot.bistu.edu.cn/tzgg/index.html','https://robot.bistu.edu.cn/tzgg/3d95cef5dc594ecbbc11333770b5cbe0.html','http://www.rabokit.com/v1/api/manual/docs/lbot_dual_arm']
def fetch(url):
    try:
        with urllib.request.urlopen(url,timeout=8) as response:
            data=response.read(4096)
            return {'url':url,'transport':'urllib default environment network/proxy configuration','http_status':response.status,
                    'bytes_read':len(data),'response_sample_sha256':hashlib.sha256(data).hexdigest(),'contents_independently_read':False}
    except urllib.error.HTTPError as error:
        return {'url':url,'transport':'urllib default environment network/proxy configuration','http_status':error.code,'error':str(error),'contents_independently_read':False}
    except (urllib.error.URLError,TimeoutError,OSError) as error:
        return {'url':url,'transport':'urllib default environment network/proxy configuration','error_type':type(error).__name__,'error':str(error),'contents_independently_read':False}
access=[]
if args.check_public_sources:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:access=list(pool.map(fetch,urls))
report={'schema_version':1,'recorded_utc':datetime.now(timezone.utc).isoformat(),'runtime_code':identity,
 'audit_kind':'existing read-only planning boundary and local readiness; no new controller/import protocol',
 'denials':denials,'denial_cases':len(denials),'local_callback_invocations':0,'observations_for_candidate':32,
 'candidate_observed':snapshot['decision'],'environment':environment_report(),
 'sdk_modules_imported':{name:name in sys.modules for name in sdk_names},
 'all_reviewed_code_config_script_test_files_unchanged':len(all_reviewed),
 'local_asset_search':{'scope':'rg non-hidden, non-ignored workspace files; no credential/session inspection',
   'patterns':['*agent_system*','*rabo*','*.urdf','*.xacro','*.bag','*.mcap','camera_info*','package.xml','metadata.yaml'],
   'matches':search.stdout.splitlines(),'limits':'filename search is not an origin/permission audit; no actual official template/data supplied'},
 'public_access_attempts':access,'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'official_platform_opened':False,'sdk_initialized':False,'robot_actions_executed':0,
 'physical_stop_verified':False,'official_score':None,
 'scope_limit':'Only this public planning-session API is denied; no sandbox/security/physical-stop proof. Fake callback is local instrumentation, not an SDK.'}
(out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps({'denials':len(denials),'callbacks':0,'reviewed_files_unchanged':len(all_reviewed),'local_asset_matches':search.stdout.splitlines(),
 'access':access,'runtime_fingerprint':identity['runtime_tree_sha256']},ensure_ascii=False,indent=2))
