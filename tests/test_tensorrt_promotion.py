from pathlib import Path
import pytest
from fine_defect_ad.tensorrt_promotion import MODEL_NAME, config_pbtxt, parse_args, path_free_evidence


def test_config_is_fixed_dynamic_batch_tensorrt_plan():
    value = config_pbtxt()
    assert f'name: "{MODEL_NAME}"' in value and 'platform: "tensorrt_plan"' in value
    assert 'max_batch_size: 8' in value and 'fp16' not in value.lower()


def test_parser_reuses_g002_and_candidate_arguments(tmp_path):
    args = parse_args(['--artifact-root', str(tmp_path), '--checkpoint', 'c', '--metrics', 'm', '--final-attempt', 'f', '--training-identity', 'i', '--dataset-root', str(tmp_path), '--teacher-small', 't', '--imagenette-root', str(tmp_path), '--lease-directory', str(tmp_path), '--run-id', 'r', '--plan', 'p', '--split-freeze', 's', '--parity-manifest', 'pm', '--calibration-artifact', 'ca', '--source-image', 'x', '--perf-analyzer', 'pa', '--perf-wheel-version', '2.60'])
    assert args.run_id == 'r' and args.calibration_artifact == Path('ca')


def test_evidence_removes_path_bearing_fields():
    assert path_free_evidence({'path': '/secret', 'nested': {'source':'x','ok':1}, 'plan': '/x', 'sha256': 'a'}) == {'nested': {'ok': 1}, 'sha256': 'a'}


def test_launcher_runs_live_path_but_never_promotes(monkeypatch, tmp_path):
    import sys
    import fine_defect_ad.tensorrt_promotion as subject
    from types import SimpleNamespace
    plan=tmp_path/'model.plan'; freeze=tmp_path/'freeze.json'; manifest=tmp_path/'manifest.json'; calibration=tmp_path/'calibration.json'; source=tmp_path/'source.png'
    for item in (plan,freeze,manifest,calibration,source): item.write_bytes(b'x')
    args=subject.PromotionArgs(tmp_path,tmp_path/'c',tmp_path/'m',tmp_path/'f',tmp_path/'i',tmp_path,tmp_path/'t',tmp_path,tmp_path,'run',plan,freeze,manifest,calibration,source,18000,tmp_path/'pa','2.60')
    events=[]
    class Lease:
        def __enter__(self): events.append('enter'); return self
        def __exit__(self,*_): events.append('exit')
    proof=SimpleNamespace(roots={'artifact':str(tmp_path)})
    monkeypatch.setattr(subject,'_admit_source',lambda _: source)
    monkeypatch.setattr(subject,'parity_manifest_entries',lambda *_: [{'path':'train/a.png','sha256':'a'}]*3)
    monkeypatch.setattr(subject,'_preflight',lambda _: proof)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(subject,'_live',lambda *_args,**_kw: {'status':'READY','promotion_eligible':False,'source_e2e':{'raw_map_sha256':'r','total_seconds':1.0,'tile_count':256,'tile_batch_size':4,'triton_call_count':65,'triton_transport':'tritonclient.http.binary'}})
    monkeypatch.setattr('fine_defect_ad.storage.atomic_write',lambda *_a,**_kw: {'status':'READY'})
    result=subject.run_promotion(args,lease_factory=lambda *_: Lease())
    assert events == ['enter','exit']
    assert result['status']=='READY' and result['promotion_eligible'] is False, result
    assert result['reason']=='TESTPUB_METRICS_DEFERRED' and 'artifact_sha256' in result
    assert result['steps']['source_e2e']['total_seconds'] == 1.0
    assert result['steps']['source_e2e']['tile_batch_size'] == 4
    assert result['steps']['source_e2e']['triton_call_count'] == 65


def test_launcher_failure_is_fail_closed(monkeypatch, tmp_path):
    import sys
    import fine_defect_ad.tensorrt_promotion as subject
    from types import SimpleNamespace
    plan=tmp_path/'model.plan'; freeze=tmp_path/'freeze.json'; manifest=tmp_path/'manifest.json'; calibration=tmp_path/'calibration.json'; source=tmp_path/'source.png'
    for item in (plan,freeze,manifest,calibration,source): item.write_bytes(b'x')
    args=subject.PromotionArgs(tmp_path,tmp_path/'c',tmp_path/'m',tmp_path/'f',tmp_path/'i',tmp_path,tmp_path/'t',tmp_path,tmp_path,'run',plan,freeze,manifest,calibration,source,18000,tmp_path/'pa','2.60')
    monkeypatch.setattr(subject,'_admit_source',lambda _: source); monkeypatch.setattr(subject,'parity_manifest_entries',lambda *_: [])
    monkeypatch.setattr(subject,'_preflight',lambda _: SimpleNamespace(roots={'artifact':str(tmp_path)})); monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(subject,'_live',lambda *_a,**_kw: (_ for _ in ()).throw(RuntimeError('boom')))
    monkeypatch.setattr('fine_defect_ad.storage.atomic_write',lambda *_a,**_kw: {'status':'READY'})
    class Lease:
        def __enter__(self): return self
        def __exit__(self,*_): pass
    result=subject.run_promotion(args,lease_factory=lambda *_: Lease())
    assert result['status']==subject.UNAVAILABLE and result['promotion_eligible'] is False

def test_model_repository_is_hash_qualified_atomic_and_cleanup(monkeypatch, tmp_path):
    import fine_defect_ad.tensorrt_promotion as subject
    from types import SimpleNamespace
    plan=tmp_path/'model.plan'; plan.write_bytes(b'plan')
    monkeypatch.setattr('fine_defect_ad.storage.atomic_write', lambda path,data,**_: (path.parent.mkdir(parents=True,exist_ok=True),path.write_bytes(data),{'status':'READY'})[-1])
    repo, plan_hash, config_hash=subject._prepare_model_repository(tmp_path,plan,proof=SimpleNamespace(),run_id='run',prefix='trt')
    assert plan_hash[:16] in repo.name and config_hash[:16] in repo.name
    assert (repo/subject.MODEL_NAME/'1'/'model.plan').read_bytes()==b'plan'
    subject._cleanup_model_repository(repo)
    assert not repo.exists()


def test_model_repository_does_not_replace_existing(monkeypatch, tmp_path):
    import fine_defect_ad.tensorrt_promotion as subject
    from types import SimpleNamespace
    plan=tmp_path/'model.plan'; plan.write_bytes(b'plan')
    monkeypatch.setattr('fine_defect_ad.storage.atomic_write', lambda path,data,**_: (path.parent.mkdir(parents=True,exist_ok=True),path.write_bytes(data),{'status':'READY'})[-1])
    repo, *_=subject._prepare_model_repository(tmp_path,plan,proof=SimpleNamespace(),run_id='run',prefix='trt')
    import pytest
    with pytest.raises(RuntimeError,match='ALREADY_EXISTS'):
        subject._prepare_model_repository(tmp_path,plan,proof=SimpleNamespace(),run_id='run',prefix='trt')
    subject._cleanup_model_repository(repo)
