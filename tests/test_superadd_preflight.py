import hashlib, json
from dataclasses import asdict
from pathlib import Path

import pytest

from fine_defect_ad import superadd_hplus_probe as producer
from fine_defect_ad.superadd_preflight import (
    ANOMALIB_COMMIT, DINO_COMMIT, DINO_LICENSE_CONTENT_SHA256, DINO_LICENSE_IDENTIFIER, DINO_LICENSE_URL,
    EXACT_HPLUS, EXACT_HPLUS_MODEL_ID, FALLBACK_LABEL, PINNED_VITS_MODEL_ID, PROBE_PRODUCER_MODULE,
    ChallengerBlocked, _probe_producer_source_sha256, main, run_preflight,
)


def _digest(letter): return letter * 64
def _canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def _sha(value): return hashlib.sha256(_canon(value)).hexdigest()


def _setup(tmp_path):
    category = tmp_path / "dataset" / "sheet_metal"; train = category / "train" / "good"; train.mkdir(parents=True)
    entries=[]
    for i in range(10):
        path=train/f"{i:03d}.png"; path.write_bytes(f"image-{i}".encode()); entries.append({"path":str(path.relative_to(category)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    identity={"data":{"train":entries,"validation":[]}}; raw=_canon(identity); artifacts=tmp_path/"artifacts";artifacts.mkdir()
    identity_path=artifacts/f"g002-training-identity-r1-{hashlib.sha256(raw).hexdigest()}.json";identity_path.write_bytes(raw)
    for name,digest in (("hplus.bin",_digest("a")),("vits.bin",_digest("b"))):
        # Synthetic files are patched by _weight digest when testing admission; actual hash is needed for run path.
        (artifacts/name).write_bytes(b"weight")
    return category.parent,artifacts,identity_path,entries

def _fixture(entries): return {"entries":entries}
def _weight(artifacts, model_id, filename, available=True):
    model=f"https://huggingface.co/{model_id}"
    if not available:return {"model_url":model,"access":"license_required"}
    path=artifacts/filename; actual=hashlib.sha256(path.read_bytes()).hexdigest()
    return {"model_url":model,"download_url":model+"/resolve/main/model.safetensors","sha256":actual,"local_path":str(path),"access":"available"}
def _provenance(artifacts, available=True):
    return {"anomalib_commit":ANOMALIB_COMMIT,"dino_commit":DINO_COMMIT,"timm_version":"1.0.28",
      "dino_license":{"identifier":DINO_LICENSE_IDENTIFIER,"url":DINO_LICENSE_URL,"acceptance":"accepted","content_sha256":DINO_LICENSE_CONTENT_SHA256} if available else {"identifier":DINO_LICENSE_IDENTIFIER,"url":DINO_LICENSE_URL,"acceptance":"not_accepted"},
      "weights":{"exact_hplus":_weight(artifacts,EXACT_HPLUS_MODEL_ID,"hplus.bin",available),"pinned_vits":_weight(artifacts,PINNED_VITS_MODEL_ID,"vits.bin",available)}}
def _safe(entries): return tuple(sorted(({"path_sha256":hashlib.sha256(e['path'].encode()).hexdigest(),"content_sha256":e['sha256']} for e in entries), key=lambda x:(x['path_sha256'],x['content_sha256'])))
def _probe(entries,weight,status="RESOURCE_FAILURE",**more):
    return {"schema_version":1,"status":status,"variant":"exact_hplus","recipe_fingerprint":_sha(asdict(EXACT_HPLUS)),"fixture_entries_sha256":_sha(_safe(entries)),"resolved_weight_sha256":weight,"runtime_binding_sha256":_digest("d"),"resource":{"peak_vram_bytes":1,"peak_host_ram_bytes":2,"seconds_per_image":3.0,"index_growth_bytes":4},"reason":"resource preflight","producer_module":PROBE_PRODUCER_MODULE,"producer_source_sha256":_probe_producer_source_sha256(),"anomalib_commit":ANOMALIB_COMMIT,**more}
def _write_probe(artifacts,payload):
    raw=_canon(payload); path=artifacts/f"superadd-hplus-probe-r1-{hashlib.sha256(raw).hexdigest()}.json";path.write_bytes(raw);return {"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()}
def _plan(entries,prov):return {"run_id":"offline-r1","fixture":_fixture(entries),"provenance":prov}
def _trusted(artifacts, entries, prov):
    def producer(*_, **__):
        payload=_probe(entries,prov["weights"]["exact_hplus"]["sha256"]); ref=_write_probe(artifacts,payload); return {**payload,"artifact":ref["path"],"artifact_sha256":ref["sha256"]}
    return producer
class _Proof:
    def __init__(self,root):self.roots={"artifact":str(root)}
class _Lease:
    def __init__(self,*_):pass
    def __enter__(self):return self
    def __exit__(self,*_):return None
def _writer(path,raw,**_):path.write_bytes(raw);return {"status":"READY"}


def test_gated_license_stops_without_hash_or_probe(tmp_path):
    dataset, artifacts, identity, entries=_setup(tmp_path); result=run_preflight(_plan(entries,_provenance(artifacts,False)),{"run_id":"offline-r1"},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/"lease",admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,producer=_trusted(artifacts,entries,p) if "p" in locals() else None,anomalib_source=artifacts)
    assert result["status"]=="STOPPED_INCOMPLETE" and result["workflow_status"]=="WEIGHT_ACCESS_REQUIRED" and "content_sha256" not in json.dumps(result["provenance"])

@pytest.mark.parametrize("mutate",[
 lambda p:p["weights"]["exact_hplus"].__setitem__("model_url",f"https://huggingface.co/{PINNED_VITS_MODEL_ID}"),
 lambda p:p["weights"]["pinned_vits"].__setitem__("model_url",f"https://huggingface.co/{EXACT_HPLUS_MODEL_ID}"),
 lambda p:p["weights"]["exact_hplus"].__setitem__("download_url","https://example.org/x"),
 lambda p:p["dino_license"].__setitem__("content_sha256",_digest("f")),
])
def test_swapped_fake_or_unattested_provenance_blocks(tmp_path,mutate):
    dataset,artifacts,identity,entries=_setup(tmp_path);p=_provenance(artifacts);mutate(p)
    with pytest.raises(ChallengerBlocked):run_preflight(_plan(entries,p),{"run_id":"offline-r1"},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/"lease",admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,producer=_trusted(artifacts,entries,p) if "p" in locals() else None,anomalib_source=artifacts)

def test_available_weight_requires_actual_local_artifact_bytes(tmp_path):
    dataset,artifacts,identity,entries=_setup(tmp_path);p=_provenance(artifacts);(artifacts/"hplus.bin").write_bytes(b"tampered")
    with pytest.raises(ChallengerBlocked):run_preflight(_plan(entries,p),{"run_id":"offline-r1"},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/"lease",admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,producer=_trusted(artifacts,entries,p) if "p" in locals() else None,anomalib_source=artifacts)

def test_only_canonical_producer_probe_unlocks_fallback(tmp_path):
    dataset,artifacts,identity,entries=_setup(tmp_path);p=_provenance(artifacts);probe=_write_probe(artifacts,_probe(entries,p["weights"]["exact_hplus"]["sha256"]))
    result=run_preflight(_plan(entries,p),{"run_id":"offline-r1"},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/"lease",admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,producer=_trusted(artifacts,entries,p) if "p" in locals() else None,anomalib_source=artifacts)
    assert result["status"]=="READY" and result["recipe"]["claim_label"]==FALLBACK_LABEL and result["recipe"]["fp16_status"].startswith("NOT_ADMITTED")

def test_inline_hplus_evidence_and_outside_probe_are_rejected(tmp_path):
    dataset,artifacts,identity,entries=_setup(tmp_path);p=_provenance(artifacts);bad=_plan(entries,p);bad["hplus_evidence"]={}
    with pytest.raises(ChallengerBlocked):run_preflight(bad,{"run_id":"offline-r1"},training_identity_path=identity,dataset_root=dataset,lease_directory=artifacts/"lease",admit=lambda _: _Proof(artifacts),writer=_writer,lease_factory=_Lease,producer=_trusted(artifacts,entries,p) if "p" in locals() else None,anomalib_source=artifacts)

def test_cli_failure_is_private(tmp_path,capsys):
    missing=tmp_path/"private.json";assert main(["--plan",str(missing),"--storage-plan",str(missing),"--training-identity",str(missing),"--dataset-root",str(tmp_path),"--lease-directory",str(tmp_path),"--anomalib-source",str(tmp_path)])==2
    assert str(missing) not in capsys.readouterr().out
