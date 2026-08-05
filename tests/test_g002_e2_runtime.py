from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import json
import numpy as np
import pytest
from PIL import Image

from fine_defect_ad.g002_e2_runtime import (COMMAND, INITIAL_BORDER, TILE, _apply_probe, collect_e2, decode_rgb01, per_origin_probe_evidence, select_e1_or_e2, stream_tiled_map, summarize_probe_cases)
from fine_defect_ad.g002_evaluate import AdmittedCheckpoint
from fine_defect_ad.storage import PreflightProof, READY

class Tensor:
 def __init__(self,value): self.value=value
 @property
 def shape(self): return self.value.shape
 def unsqueeze(self,dim): return Tensor(np.expand_dims(self.value,dim))
 def to(self,_): return self
class Torch:
 @staticmethod
 def from_numpy(x): return Tensor(x)

def admitted(root,size=(480,480)):
 leaf=root/'sheet_metal'/'validation'/'good';leaf.mkdir(parents=True);rows=[]
 for i in range(19):
  p=leaf/f'{i:02}.png';Image.fromarray(np.full((*size,3),i,dtype=np.uint8)).save(p);rows.append((f'validation/good/{p.name}',sha256(p.read_bytes()).hexdigest()))
 return AdmittedCheckpoint(root/'x.ckpt','c','s','m','a','i','p','run',root,tuple(rows))

def mapper(tile):
 assert tile.shape==(1,3,TILE,TILE); score=tile.value[0].mean(axis=0); return score[None,None],(score+.25)[None,None]

def test_streaming_tiling_covers_edges_and_uses_one_exact_tile():
 raw,plan,boxes,lo,hi,seam=stream_tiled_map(np.zeros((480,480,3),np.float32),mapper,Torch,border=INITIAL_BORDER)
 assert raw.shape==(480,480) and raw.dtype.str=='<f4' and plan.stride==(224,224) and lo>=1 and hi>=lo and len(boxes)==4 and seam==0

def test_endpoint_probe_and_decode():
 x=np.asarray([[[.1,.5,.9]]],np.float32);assert _apply_probe(x,{'point':[0,0]},'black_endpoint').tolist()==[[[0.,0.,0.]]]
 with TemporaryDirectory() as d:
  p=Path(d)/'x.png';Image.fromarray(np.zeros((2,2),np.uint8)).save(p);assert decode_rgb01(p).shape==(2,2,3)

def test_per_origin_runs_four_forwards_per_origin_both_families_and_polarities():
 calls=[]
 def counted(tile): calls.append(tile.shape);return mapper(tile)
 rgb=np.zeros((480,480,3),np.float32); evidence=per_origin_probe_evidence(rgb,counted,Torch,identity='validation/good/a.png',source_sha256='a'*64,border=16)
 assert {r['family'] for r in evidence['records']}=={'impulse','seam_crossing_line'} and {r['polarity'] for r in evidence['records']}=={'black_endpoint','white_endpoint'}
 assert len(calls)==4*sum(1 for _ in evidence['records']) # every retained origin record is exactly four calls
 assert all('cross_origin_normal_disagreement' in r and 'cross_origin_response_disagreement' in r for r in evidence['records']) and len(evidence['output_sha256'])==64

def test_collect_writes_map_immediately_and_retains_no_raw_corpus():
 with TemporaryDirectory() as d:
  written=[]
  record=collect_e2(admitted(Path(d)),mapper,Torch,map_sink=lambda row: written.append(row['_bytes']) or {'path':'sink'})
 assert len(written)==19 and all('_bytes' not in row for row in record['maps']) and len(record['probe_summary']['cases'])>0

def measured(cases,delta=0):
 updated=[]
 for c in cases:
  value=dict(c);value.setdefault('normal_repeatability',0.);value.setdefault('response_repeatability',.1);value.setdefault('cross_origin_normal_disagreement',0.);value.setdefault('cross_origin_response_disagreement',0.);value.setdefault('recipe_sha256','r'*64);value.setdefault('probe_content_sha256','p'*64);value['response_interval']=[c['response_interval'][0]+delta,c['response_interval'][1]+delta];updated.append(value)
 maps=[{'image_identity':f'validation/good/{i}.png','map_sha256':'a'*64,'coverage_min':1} for i in range(19)]
 return {'maps':maps,'cases':updated,'latency_seconds':1.0,'vram':{'allocated_bytes':1,'reserved_bytes':2}}

def test_selection_is_measured_hierarchical_positive_tie_and_prohibited_inputs():
 rows=[]
 for family in ('impulse','seam_crossing_line'):
  rows.append({'image_identity':'validation/good/a.png','family':family,'case':family,'polarity':'black_endpoint','response_interval':[1.,1.], 'normal_repeatability':0.,'response_repeatability':.1,'cross_origin_normal_disagreement':0.,'cross_origin_response_disagreement':0.,'recipe_sha256':'r'*64,'probe_content_sha256':'p'*64})
 base=measured(rows); improved=measured(rows,1.)
 assert select_e1_or_e2(e1=base,e2=improved)['selected']=='E2'
 assert select_e1_or_e2(e1=base,e2=base)['selected']=='E1'
 with pytest.raises(ValueError):select_e1_or_e2(e1={**base,'test':1},e2=base)

def test_e2_ready_integration_uses_e2_lease_command_and_writes_each_map():
 from fine_defect_ad.g002_eval_runtime import EvaluationArgs
 from fine_defect_ad.g002_e2_runtime import run_e2_evaluation
 import fine_defect_ad.g002_evaluate as admission_module
 with TemporaryDirectory() as d:
  root=Path(d); gate=admitted(root,size=(256,256)); checkpoint=root/'g002-last-run-0.ckpt';checkpoint.write_bytes(b'x');gate=replace(gate,path=checkpoint,checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest());sidecar=checkpoint.with_suffix('.ckpt.json');sidecar.write_text('{}');metrics=root/'g002-metrics-run.json';metrics.write_text('[]');final=root/'g002-attempt-run-x.json';final.write_text('{}')
  identity={'data':{'validation':[{'path':p,'sha256':h} for p,h in gate.validation_identities]}}; raw=json.dumps(identity,sort_keys=True,separators=(',',':')).encode();ip=root/f'g002-training-identity-run-{sha256(raw).hexdigest()}.json';ip.write_bytes(raw)
  class LiveTorch:
   class serialization:
    @staticmethod
    def safe_globals(_):return nullcontext()
   class cuda:
    @staticmethod
    def is_available():return True
    @staticmethod
    def max_memory_allocated():return 10
    @staticmethod
    def max_memory_reserved():return 20
   @staticmethod
   def load(*_,**__):return {'state_dict':{},'global_step':70000}
   @staticmethod
   def device(_):return 'cuda:0'
   @staticmethod
   def from_numpy(x):return Tensor(x)
   @staticmethod
   def inference_mode():return nullcontext()
  class Lease:
   def __init__(self,*_):pass
   def __enter__(self):return self
   def __exit__(self,*_):return False
  class Model:
   class Inner:
    def get_maps(self,tile,*,normalize):assert normalize is False;return mapper(tile)
   model=Inner()
   def load_state_dict(self,_):pass
   def eval(self):return self
   def to(self,_):return self
  proof=PreflightProof('e2',{'artifact':str(root)},'x','2000-01-01T00:00:00+00:00',{},[],{})
  def write(path,payload,**_):Path(path).write_bytes(payload);return {'status':READY}
  original=admission_module.admit_completed_checkpoint;admission_module.admit_completed_checkpoint=lambda *_:gate
  try:
   got=run_e2_evaluation(EvaluationArgs(root,checkpoint,sidecar,metrics,final,ip,root,root/'teacher',root/'imagenette','e2',root/'leases'),runtime_factory=lambda *_,**__:(Model(),None,None,None),lease_factory=Lease,torch_module=LiveTorch,admit=lambda **_:proof,writer=write,lease_event_loader=lambda *_:[{'state':'acquired','run_id':'e2','command':COMMAND,'pid':__import__('os').getpid(),'timestamp':'2026-01-01T00:00:00+00:00'},{'state':'released','run_id':'e2','command':COMMAND,'pid':__import__('os').getpid(),'timestamp':'2026-01-01T00:00:01+00:00','outcome':'normal'}])
  finally:admission_module.admit_completed_checkpoint=original
  assert got['status']==READY, got['limitations']
  assert got['lease_events'][-1]['outcome']=='normal' and len(got['raw_maps']['map_paths'])==19
  admission_module.admit_completed_checkpoint=lambda *_:gate
  try:
   failed=run_e2_evaluation(EvaluationArgs(root,checkpoint,sidecar,metrics,final,ip,root,root/'teacher',root/'imagenette','e2',root/'leases'),runtime_factory=lambda *_,**__:(_ for _ in ()).throw(RuntimeError('RUNNER_CAUSE')),lease_factory=Lease,torch_module=LiveTorch,admit=lambda **_:proof,writer=write,lease_event_loader=lambda *_:[])
  finally: admission_module.admit_completed_checkpoint=original
  assert 'RUNNER:RuntimeError:RUNNER_CAUSE' in failed['limitations'][0] and 'LEASE_EVIDENCE:' in failed['limitations'][0]

def test_one_revision_preserves_initial_and_never_third(monkeypatch):
 import fine_defect_ad.g002_e2_runtime as module
 calls=[]; notices=[]
 def fake(*_,border,**__):
  calls.append(border);return {'geometry':{'empirical_border':24 if len(calls)==1 else 28},'maps':[],'probe_summary':{}}
 monkeypatch.setattr(module,'collect_e2',fake)
 got=module.collect_e2_with_one_revision(object(),None,None,map_sink=lambda _:None,initial_failure_sink=notices.append)
 assert calls==[16,24] and len(notices)==1 and got['revision']['status']=='REVISION_UNSTABLE_RETAIN_E1'

def test_freeze_binds_identity_checkpoint_hardware_and_blocks_test_terms():
 from fine_defect_ad.g002_e2_runtime import freeze_pretest_selection
 with TemporaryDirectory() as d:
  gate=admitted(Path(d),size=(256,256));cases=[]
  for family in ('impulse','seam_crossing_line'):cases.append({'image_identity':'validation/good/a.png','family':family,'case':family,'polarity':'black_endpoint','response_interval':[1.,1.],'normal_repeatability':0.,'response_repeatability':.1,'cross_origin_normal_disagreement':0.,'cross_origin_response_disagreement':0.,'recipe_sha256':'r'*64,'probe_content_sha256':'p'*64})
  value=measured(cases);value['probe_recipe_sha256']='r'*64
  frozen=freeze_pretest_selection(e1=value,e2=value,admitted=gate,hardware={'allocated_bytes':1,'reserved_bytes':2})
  assert frozen['selection']['selected']=='E1' and len(frozen['validation_identities'])==19 and len(frozen['freeze_sha256'])==64

def test_module_has_one_public_selection_and_collection_definition():
 import ast
 import importlib
 source=Path(importlib.import_module('fine_defect_ad.g002_e2_runtime').__file__).read_text()
 names=[node.name for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)]
 assert names.count('collect_e2')==1 and names.count('select_e1_or_e2')==1

def test_cli_help_is_available():
 import subprocess,sys,os
 got=subprocess.run([sys.executable,'-m','fine_defect_ad.g002_e2_runtime','--help'],env={**os.environ,'PYTHONPATH':f'{Path.cwd()/"src"}:{Path.cwd()/".internal/venv/r1-overlay"}'},text=True,capture_output=True)
 assert got.returncode==0 and '--training-identity' in got.stdout

def test_freeze_summary_hash_tamper_is_rejected():
 from fine_defect_ad.g002_e2_runtime import freeze_pretest_selection,verify_pretest_freeze
 with TemporaryDirectory() as d:
  gate=admitted(Path(d),size=(256,256)); cases=[]
  for family in ('impulse','seam_crossing_line'): cases.append({'image_identity':'validation/good/a.png','family':family,'case':family,'polarity':'black_endpoint','response_interval':[1.,1.],'normal_repeatability':0.,'response_repeatability':.1,'cross_origin_normal_disagreement':0.,'cross_origin_response_disagreement':0.,'recipe_sha256':'r'*64,'probe_content_sha256':'p'*64})
  value=measured(cases); value['geometry']={'empirical_border':16};value['revision']={'e2_eligible':True}; frozen=freeze_pretest_selection(e1=value,e2=value,admitted=gate,hardware={'gpu':'fake'})
  frozen['e2_measurement']['cases'][0]['response_interval'][0]=9.
  with pytest.raises(ValueError):verify_pretest_freeze(frozen)


def test_freeze_rejects_rehashed_forged_selection_and_revision():
 from copy import deepcopy
 from fine_defect_ad.g002_e2_runtime import _canonical, freeze_pretest_selection, verify_pretest_freeze
 with TemporaryDirectory() as d:
  gate=admitted(Path(d),size=(256,256)); cases=[]
  for family in ('impulse','seam_crossing_line'): cases.append({'image_identity':'validation/good/a.png','family':family,'case':family,'polarity':'black_endpoint','response_interval':[1.,1.],'normal_repeatability':0.,'response_repeatability':.1,'cross_origin_normal_disagreement':0.,'cross_origin_response_disagreement':0.,'recipe_sha256':'r'*64,'probe_content_sha256':'p'*64})
  value=measured(cases); value['geometry']={'empirical_border':16}; value['revision']={'e2_eligible':True}
  frozen=freeze_pretest_selection(e1=value,e2=value,admitted=gate,hardware={})
  for key, forged in (("selection", {**frozen["selection"], "selected":"E2"}), ("revision", {"e2_eligible":False})):
   changed=deepcopy(frozen); changed[key]=forged; changed['freeze_sha256']=sha256(_canonical({name:item for name,item in changed.items() if name!='freeze_sha256'})).hexdigest()
   with pytest.raises(ValueError): verify_pretest_freeze(changed)



def test_revision_uses_actual_uniform_map_border_and_retains_e1(monkeypatch):
 import fine_defect_ad.g002_e2_runtime as module
 assert module._collected_border({"maps":[{"border":80}] * 19, "geometry":{"empirical_border":96}}) == 80
 with pytest.raises(ValueError): module._collected_border({"maps":[{"border":80}, {"border":96}]})
 captured=[]; original=module.bounded_tiles
 monkeypatch.setattr(module, "bounded_tiles", lambda shape, tile, *, invalid_border: captured.append(invalid_border) or original(shape, tile, invalid_border=invalid_border))
 with TemporaryDirectory() as directory: module.collect_e1_comparison(admitted(Path(directory), size=(256,256)), mapper, Torch, border=80)
 assert captured and captured[0] == (80,80)
 rows=[{'image_identity':'validation/good/a.png','family':f,'case':f,'polarity':'black_endpoint','response_interval':[1.,1.], 'normal_repeatability':0.,'response_repeatability':.1,'cross_origin_normal_disagreement':0.,'cross_origin_response_disagreement':0.,'recipe_sha256':'r'*64,'probe_content_sha256':'p'*64} for f in ('impulse','seam_crossing_line')]
 assert module.select_e1_or_e2(e1=measured(rows), e2={**measured(rows), 'revision':{'e2_eligible':False}})['selected'] == 'E1'


def test_split_branch_hann_terminal_coverage_and_fresh_quantiles():
 from fine_defect_ad.g002_e2_runtime import _split_boxes, periodic_hann_weights, split_quantiles
 shape=(480, 560); boxes=_split_boxes(shape)
 assert boxes[-1] == (224, 304, 480, 560)
 coverage=np.zeros(shape, np.float64)
 for box in boxes:
  weight=periodic_hann_weights(box, shape)
  assert weight.shape == (256,256) and weight.min() > 0
  y,x,y2,x2=box;coverage[y:y2,x:x2] += weight
 assert coverage.min() > 0
 q=split_quantiles([np.arange(100,dtype=np.float32)], [np.arange(100,dtype=np.float32)+.5])
 assert q['qb_st'] > q['qa_st'] and q['qb_stae'] > q['qa_stae']


def test_split_branch_local_matches_get_maps_st_and_never_calls_ae_per_tile():
 torch=pytest.importorskip('torch')
 import torch.nn.functional as F
 from fine_defect_ad.g002_e2_runtime import local_st_map
 class Core:
  teacher_out_channels=2;pad_maps=True;mean_std={'mean':torch.zeros(1,2,1,1),'std':torch.ones(1,2,1,1)}
  def __init__(self): self.ae_calls=0
  def is_set(self, _): return True
  def teacher(self,x): return torch.cat((x[:,:1],x[:,1:2]),1)
  def student(self,x): return torch.cat((x[:,:1]+1,x[:,1:2]-1,x[:,:1],x[:,1:2]),1)
  def ae(self,*_): self.ae_calls += 1; raise AssertionError('AE must not run for local ST')
  def get_maps(self,x,normalize=False):
   t=self.teacher(x);s=self.student(x);st=(t-s[:,:2]).square().mean(1,keepdim=True)
   return F.interpolate(F.pad(st,(4,4,4,4)),size=x.shape[-2:],mode='bilinear'), torch.zeros_like(x[:,:1])
 core=Core();tile=torch.rand(1,3,256,256)
 assert torch.equal(local_st_map(tile, core, torch), core.get_maps(tile, normalize=False)[0]) and core.ae_calls == 0


def test_split_freeze_binds_fresh_maps_geometry_and_new_decision():
 from fine_defect_ad.g002_e2_runtime import freeze_split_validation, split_quantiles, verify_split_freeze
 with TemporaryDirectory() as d:
  gate=admitted(Path(d), size=(256,256)); local=[np.arange(100,dtype=np.float32)+i for i in range(19)]; global_=[x+.25 for x in local]
  rows=[]
  for (identity, source), st, stae in zip(gate.validation_identities, local, global_):
   rows.append({'image_identity':identity,'source_sha256':source,'local_st_sha256':sha256(st.tobytes()).hexdigest(),'global_stae_sha256':sha256(stae.tobytes()).hexdigest(),'_local_st':st,'_global_stae':stae})
  frozen=freeze_split_validation(admitted=gate,quantiles=split_quantiles(local,global_),map_rows=rows,geometry={'tile':256,'stride':128,'weight_min':.1})
  verify_split_freeze(frozen)
  assert frozen['decision_id'] != 'DEC-GEO-002' and frozen['status'] == 'READY'
