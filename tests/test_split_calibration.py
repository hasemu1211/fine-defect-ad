import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import fine_defect_ad.split_calibration as module
from fine_defect_ad.g002_e2_runtime import SPLIT_DECISION_ID, SPLIT_TARGET_SHAPE
from fine_defect_ad.storage import PreflightProof, READY


def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def digest(value): return sha256(value if isinstance(value, bytes) else Path(value).read_bytes()).hexdigest()

def fixture(root, *, duplicate=False):
    rows=[]; identities=[]
    for index in range(19):
        identity=f"validation/good/{index:02d}.png"; source=sha256(identity.encode()).hexdigest(); identities.append({"path":identity,"sha256":source})
        for branch in ('st','stae'):
            raw=np.array([index, index + (2 if branch == 'stae' else 1)],dtype='<f4').tobytes(); h=digest(raw)
            (root/f"g002-e2-split-validation-run-{branch}-{index:02d}-{h}.bin").write_bytes(raw)
            if duplicate and index == 0 and branch == 'st': (root/f"g002-e2-split-validation-short-{h}.bin").write_bytes(raw)
            if branch == 'st': local=h
            else: global_=h
        rows.append({"image_identity":identity,"source_sha256":source,"local_st_sha256":local,"global_stae_sha256":global_,"local_st_shape":[2],"global_stae_shape":[2]})
    freeze={"stage":"PRE_TEST_FREEZE","status":"READY","decision_id":SPLIT_DECISION_ID,"checkpoint_sha256":"a"*64,"validation_identities":identities,"geometry":{},"quantiles":{"qa_st":0,"qb_st":1,"qa_stae":0,"qb_stae":1},"maps":rows,"code_sha256":"b"*64}
    freeze['freeze_sha256']=digest(canon(freeze)); path=root/f"g002-e2-split-pretest-freeze-run-{freeze['freeze_sha256']}.json"; path.write_bytes(canon(freeze))
    return module.SplitCalibrationInput(root,path,'run'), rows

def admit(**_): return PreflightProof('run',{},'x','2000-01-01T00:00:00+00:00',{},[],{})
def writer(path,payload,**_): Path(path).write_bytes(payload); return {'status':READY}

def combine(local, global_, *_):
    return np.full(SPLIT_TARGET_SHAPE, float(local.mean()+global_.mean()), dtype='<f4')

def test_exact_stats_and_tie_metadata(monkeypatch):
    with TemporaryDirectory() as temp:
        args, rows=fixture(Path(temp)); monkeypatch.setattr(module,'combine_split_maps',combine)
        result=module.calibrate(args, torch=object(), admit=admit, writer=writer); proof=json.loads(Path(result['artifact']).read_text())
        values=[index + .5 + index + 1 for index in range(19)]
        assert proof['pixel_count'] == 19 * 528 * 2112
        assert proof['population_mean'] == pytest.approx(sum(values)/19)
        assert proof['decision'] == {'comparator':'>','image_rule':'any pixel > threshold','provenance':'project decision; not claimed official MVTec comparator'}
        assert proof['test_access'] == 'NONE' and proof['combined_map_persistence'] == 'NONE'

def test_duplicate_identical_artifacts_pass_and_record_provenance(monkeypatch):
    with TemporaryDirectory() as temp:
        args, _ = fixture(Path(temp), duplicate=True); monkeypatch.setattr(module, 'combine_split_maps', combine)
        result = module.calibrate(args, torch=object(), admit=admit, writer=writer)
        proof = json.loads(Path(result['artifact']).read_text())
        artifact = proof['validation_maps'][0]['local_st_artifact']
        assert artifact['artifact_duplicate_count'] == 2
        assert len(artifact['artifact_basenames']) == 2


@pytest.mark.parametrize('case', ['mismatch', 'zero'])
def test_mismatched_or_missing_hash_artifact_fails(monkeypatch, case):
    with TemporaryDirectory() as temp:
        root = Path(temp); args, rows = fixture(root); monkeypatch.setattr(module, 'combine_split_maps', combine)
        digest = rows[0]['local_st_sha256']
        if case == 'mismatch':
            (root / f'bad-{digest}.bin').write_bytes(b'not-the-digested-map')
        else:
            next(root.glob(f'*-{digest}.bin')).unlink()
        with pytest.raises(ValueError, match='missing or mismatched'):
            module.calibrate(args, torch=object(), admit=admit, writer=writer)

def test_preflight_before_write(monkeypatch):
    with TemporaryDirectory() as temp:
        args,_=fixture(Path(temp)); monkeypatch.setattr(module,'combine_split_maps',combine); calls=[]
        def reject(**kwargs): calls.append(kwargs); raise RuntimeError('preflight')
        with pytest.raises(RuntimeError,match='preflight'): module.calibrate(args,torch=object(),admit=reject,writer=writer)
        assert calls and not list(Path(temp).glob('g002-e2-split-calibration-*'))
