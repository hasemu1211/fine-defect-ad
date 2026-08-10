from fine_defect_ad.tensorrt_testpub import Args, parse_args

def test_parser_accepts_backend_ab_inputs(tmp_path):
    values=['--artifact-root',str(tmp_path),'--checkpoint','c','--metrics','m','--final-attempt','f','--training-identity','i','--dataset-root',str(tmp_path),'--teacher-small','t','--imagenette-root',str(tmp_path),'--lease-directory',str(tmp_path),'--run-id','r','--plan','p','--split-freeze','s','--evaluator','e']
    args=parse_args(values)
    assert args.run_id=='r' and args.http_port==18000

def test_args_preserve_metric_wiring_contract(tmp_path):
    args=Args(tmp_path,*([tmp_path]*8),'run',tmp_path/'plan',tmp_path/'freeze',tmp_path/'evaluator')
    assert args.plan.name=='plan' and args.evaluator.name=='evaluator'

def test_manifest_has_exact_public_count_and_no_host_paths():
    from fine_defect_ad.tensorrt_testpub import raw_manifest
    rows=[{'image_identity':f'test_public/good/{i}.png','label':'good','source_sha256':'a','mask_sha256':None,'map_sha256':'b','dtype':'<f4','byte_order':'<','shape':[528,2112],'seconds':.1} for i in range(114)]
    manifest=raw_manifest(run_id='r',checkpoint_sha256='c',split_freeze_sha256='f',plan_sha256='p',rows=rows,total_seconds=12.0)
    assert len(manifest['maps']) == 114 and all('source' not in row for row in manifest['maps'])

def test_attempt_latch_rejects_alternate_binding(tmp_path):
    from fine_defect_ad.tensorrt_testpub import Args, _establish_attempt_latch
    from types import SimpleNamespace
    args=Args(tmp_path,*([tmp_path]*8),'run',tmp_path/'plan',tmp_path/'freeze',tmp_path/'evaluator')
    def writer(path, data, **_): path.write_bytes(data); return {'status':'READY'}
    _establish_attempt_latch(tmp_path,args,{'checkpoint_sha256':'c','split_freeze_sha256':'f','plan_sha256':'p'},SimpleNamespace(),writer)
    import pytest
    with pytest.raises(RuntimeError, match='BINDING_MISMATCH'):
        _establish_attempt_latch(tmp_path,args,{'checkpoint_sha256':'other','split_freeze_sha256':'f','plan_sha256':'p'},SimpleNamespace(),writer)


def test_legacy_authorized_recovery_reads_maps_without_sources(tmp_path):
    import json
    from hashlib import sha256
    from fine_defect_ad.tensorrt_testpub import Args, _existing_testpub_binding, _recover_legacy_authorized
    run='tensorrt-testpub-ab-20260810a'; rows=[]
    for index in range(114):
        payload=f'map-{index}'.encode(); digest=sha256(payload).hexdigest()
        (tmp_path/f'tensorrt-testpub-raw-{index:03d}-{digest}.bin').write_bytes(payload)
        rows.append({'map_sha256':digest})
    manifest={'binding':{'checkpoint_sha256':'c','split_freeze_sha256':'f','plan_sha256':'p'},'maps':rows}
    path=tmp_path/f'tensorrt-testpub-manifest-{run}-x.json'; path.write_text(json.dumps(manifest))
    args=Args(tmp_path,*([tmp_path]*8),run,tmp_path/'plan',tmp_path/'freeze',tmp_path/'evaluator')
    binding,evidence=_existing_testpub_binding(tmp_path,run)
    result=_recover_legacy_authorized(tmp_path,args,binding,evidence)
    assert result['initial_attempt_latch']=='NOT_AVAILABLE_LEGACY_AUTHORIZED_COMPARISON'
    assert result['recovery']=='READ_ONLY_PERSISTED_MAPS_MANIFEST'

def test_same_binding_latch_is_explicitly_blocked(tmp_path):
    from fine_defect_ad.tensorrt_testpub import Args, _establish_attempt_latch
    from types import SimpleNamespace
    args=Args(tmp_path,*([tmp_path]*8),'run',tmp_path/'plan',tmp_path/'freeze',tmp_path/'evaluator')
    def writer(path, data, **_): path.write_bytes(data); return {'status':'READY'}
    binding={'checkpoint_sha256':'c','split_freeze_sha256':'f','plan_sha256':'p'}
    assert _establish_attempt_latch(tmp_path,args,binding,SimpleNamespace(),writer)=='NEW'
    assert _establish_attempt_latch(tmp_path,args,binding,SimpleNamespace(),writer)=='EXISTING_BLOCKED'
