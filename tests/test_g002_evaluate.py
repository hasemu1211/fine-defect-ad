import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pytest
from fine_defect_ad.g002_evaluate import VALIDATION_GOOD_COUNT, admit_completed_checkpoint, collect_validation_maps, persist_validation_maps, raw_map
from fine_defect_ad.g002_training import PILOT_SHA256
from fine_defect_ad.storage import PreflightProof, READY
class Model:
 def get_maps(self,image,normalize=False): assert normalize is False; return np.asarray(image),np.asarray(image)+2
def digest(p): return sha256(Path(p).read_bytes()).hexdigest()
def inputs(dataset):
 rows=[]; loader=[]; leaf=dataset/'sheet_metal'/'validation'/'good';leaf.mkdir(parents=True)
 for i in range(19):
  p=leaf/f'{i:02}.png';p.write_bytes(f'image-{i}'.encode()); key=f'validation/good/{p.name}';rows.append({'path':key,'sha256':digest(p)});loader.append({'image':[i,i+1],'image_identity':key,'source_path':p})
 return {'data':{'validation':rows},'other':'bound'},loader
def fixture(root,identity):
 c=root/'g002-last-run-0.ckpt';m=root/'g002-metrics-run.json';c.write_bytes(b'checkpoint');m.write_bytes(b'[]'); ih=sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest();s={'checkpoint_name':c.name,'checkpoint_sha256':digest(c),'identity_sha256':ih,'pilot_sha256':PILOT_SHA256,'global_step':70000,'lineage':'run'};sp=c.with_suffix('.ckpt.json');sp.write_text(json.dumps(s));a={'run_id':'run','status':READY,'lease_outcome':'normal','artifacts':{'checkpoint':digest(c),'sidecar':digest(sp),'metrics':digest(m)}};payload=json.dumps(a,sort_keys=True).encode();f=root/f'g002-attempt-run-{sha256(payload).hexdigest()}.json';f.write_bytes(payload);return c,m,f
def gate(root,identity,dataset):
 c,m,f=fixture(root,identity);return admit_completed_checkpoint(c,root,identity,dataset,f,m)
@pytest.mark.parametrize('target',['checkpoint','sidecar','metrics','final'])
def test_admission_rejects_tampered_artifacts(target):
 with TemporaryDirectory() as d:
  root=Path(d);identity,unused=inputs(root/'data');c,m,f=fixture(root,identity);{'checkpoint':c,'sidecar':c.with_suffix('.ckpt.json'),'metrics':m,'final':f}[target].write_bytes(b'x')
  with pytest.raises(ValueError):admit_completed_checkpoint(c,root,identity,root/'data',f,m)
def test_admission_binds_identity_lineage_and_names():
 with TemporaryDirectory() as d:
  root=Path(d);identity,unused=inputs(root/'data');c,m,f=fixture(root,identity)
  forged={**identity,'other':'forged'}
  with pytest.raises(ValueError):admit_completed_checkpoint(c,root,forged,root/'data',f,m)
  with pytest.raises(ValueError):admit_completed_checkpoint(c,root,identity,root/'bad',f,m)
def test_collection_only_uses_admitted_identity_and_exact_paths():
 with TemporaryDirectory() as d:
  root=Path(d);identity,loader=inputs(root/'data');result=collect_validation_maps(Model(),loader,gate(root,identity,root/'data'));assert len(result['maps'])==19 and result['maps'][0]['dtype']=='<f4'
  loader[0]['source_path']=root/'data'/'sheet_metal'/'test'/'x.png'
  with pytest.raises(ValueError):collect_validation_maps(Model(),loader,gate(root,identity,root/'data'))
  loader[0]['source_path']=root/'data'/'sheet_metal'/'validation'/'good'/'00.png';loader[0]['mask']=1
  with pytest.raises(ValueError):collect_validation_maps(Model(),loader,gate(root,identity,root/'data'))
def test_raw_map_converts_before_addition_and_rejects_shape():
 assert raw_map(np.array([2**31-1],dtype=np.int32),np.array([2**31-1],dtype=np.int32)).dtype.str=='<f4'
 with pytest.raises(ValueError):raw_map(np.ones(2),np.ones(3))
def test_lossless_roundtrip_and_exact_capacity_proof():
 with TemporaryDirectory() as d:
  root=Path(d);identity,loader=inputs(root/'data');collection=collect_validation_maps(Model(),loader,gate(root,identity,root/'data'));seen={}
  proof=PreflightProof('run',{'artifact':str(root)},'x','2000-01-01T00:00:00+00:00',{},[],{})
  def admit(**kwargs):seen.update(kwargs);return proof
  def write(path,payload,**kwargs):Path(path).write_bytes(payload);return {'status':READY}
  result=persist_validation_maps(collection,root,'run',{'normalize':False,'resize':256},admit=admit,writer=write);persistent=sum(len(row['_bytes']) for row in collection['maps'])+Path(result['manifest']).stat().st_size;assert seen['allocations'][0].bytes==persistent;assert seen['allocations'][1].bytes==seen['reserve_bytes']==max([Path(result['manifest']).stat().st_size,*[len(x['_bytes']) for x in collection['maps']]])
  assert Path(result['map_paths'][0]).read_bytes()==collection['maps'][0]['_bytes']
  with pytest.raises(ValueError):persist_validation_maps(collection,root,'x',{'normalize':False,'path':'/bad'},admit=admit,writer=write)
  with pytest.raises(ValueError,match='write failed'):persist_validation_maps(collection,root,'x',{'normalize':False},admit=admit,writer=lambda *a,**k:{'status':'STOPPED'})
