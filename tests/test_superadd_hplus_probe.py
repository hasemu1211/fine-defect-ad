import hashlib,json
from pathlib import Path
from fine_defect_ad import superadd_hplus_probe as subject
from fine_defect_ad.superadd_preflight import *
from test_superadd_preflight import _setup,_provenance,_plan,_Proof,_Lease,_writer

def test_producer_injected_real_step_writes_consumer_schema(tmp_path,monkeypatch):
    dataset,artifacts,identity,entries=_setup(tmp_path);plan=_plan(entries,_provenance(artifacts)); source=tmp_path/'source';source.mkdir()
    seen=[]
    def step(**kwargs):
        seen.extend(item['path'] for item in kwargs['fixture']['entries']); return {'resource':{'peak_vram_bytes':1,'peak_host_ram_bytes':2,'seconds_per_image':.1,'index_growth_bytes':3}}
    result=subject.produce_probe(plan,{"run_id":"offline-r1"},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/'lease',anomalib_source=source,admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,step=step,runtime_binding=lambda _:'d'*64,source_verifier=lambda _:None)
    assert seen == [item['path'] for item in entries]
    assert result['producer_module']==PROBE_PRODUCER_MODULE and Path(result['artifact']).is_file()

def test_default_cli_dispatches_producer(tmp_path,monkeypatch,capsys):
    for n in ('p','s','i','d','l','a'): (tmp_path/n).write_text('{}')
    monkeypatch.setattr(subject,'produce_probe',lambda *a,**k:{'status':'READY','artifact':'/private'})
    assert subject.main(['--plan',str(tmp_path/'p'),'--storage-plan',str(tmp_path/'s'),'--training-identity',str(tmp_path/'i'),'--dataset-root',str(tmp_path/'d'),'--lease-directory',str(tmp_path/'l'),'--anomalib-source',str(tmp_path/'a')])==0
    assert 'private' not in capsys.readouterr().out

def test_actual_cuda_oom_is_resource_failure_and_integrates_to_fallback(tmp_path, monkeypatch):
    import sys, types
    dataset,artifacts,identity,entries=_setup(tmp_path); plan=_plan(entries,_provenance(artifacts)); source=tmp_path/'source';source.mkdir()
    class OOM(Exception): pass
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=OOM)))
    def oom_step(**_): raise OOM('allocator exhausted')
    result=subject.produce_probe(plan,{'run_id':'offline-r1'},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/'lease',anomalib_source=source,admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,step=oom_step,runtime_binding=lambda _:'d'*64,source_verifier=lambda _:None)
    assert result['status']=='RESOURCE_FAILURE'
    from fine_defect_ad.superadd_preflight import run_preflight
    def trusted(*_,**__): return result
    selected=run_preflight(plan,{'run_id':'offline-r1'},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/'lease',admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,producer=trusted,anomalib_source=source)
    assert selected['workflow_status']=='PINNED_VITS_ADMITTED'
