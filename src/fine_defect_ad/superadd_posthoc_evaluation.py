"""Resumable posthoc evaluation of immutable full-resolution SuperADD raw maps."""
from __future__ import annotations
import argparse, json, subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from PIL import Image
from .g002_e2_runtime import SPLIT_TARGET_SHAPE
from .g002_testpub_runtime import GOOD_COUNT, BAD_COUNT, test_public_entries
from .evaluation_history import _image_auroc
from .mvtec_aupro import local_au_pro_0_05
from .storage import Allocation, READY, atomic_write, preflight
from .superadd_comparison import MAP_PREFIX, LATCH_PREFIX, _canonical, _hash, _path_free, _read_canonical, _anon, ChallengerBlocked

DERIVED_PREFIX = "superadd-vits-posthoc-raw"
COMMAND = "superadd-vits-posthoc-evaluation"

def _git_source(commit: str) -> str:
    value=subprocess.run(["git","-C",str(Path(__file__).resolve().parents[2]),"show",f"{commit}:src/fine_defect_ad/superadd_comparison.py"],check=True,capture_output=True).stdout
    return sha256(value).hexdigest()

def _resize_map(raw: bytes, shape: Sequence[int]) -> np.ndarray:
    if list(shape) != [1056, 4224] or len(raw) != 1056 * 4224 * 4: raise ChallengerBlocked("historical raw map must be exact 1056x4224 <f4")
    value=np.frombuffer(raw,dtype="<f4").reshape(shape)
    if not np.isfinite(value).all(): raise ChallengerBlocked("historical raw map non-finite")
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        output=F.interpolate(torch.from_numpy(value.copy()).reshape(1,1,*shape),size=SPLIT_TARGET_SHAPE,mode="bilinear",align_corners=False)[0,0].cpu().numpy()
    return np.asarray(output,dtype="<f4")

def _write(root:Path,run_id:str,path:Path,payload:bytes,*,admit:Callable[...,Any],writer:Callable[...,Any],component:str)->None:
    source=f"exact {component} bytes={len(payload)}"; proof=admit(run_id=run_id,allocations=[Allocation("artifact",len(payload),"persistent",source,component),Allocation("artifact",len(payload),"transient",source,component+"-incoming")],reserve_bytes=len(payload),reserve_evidence={"max_pending_atomic_write_bytes":len(payload),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ChallengerBlocked("posthoc proof artifact root changed")
    if writer(path,payload,proof=proof,run_id=run_id,overwrite=False).get("status")!=READY or path.read_bytes()!=payload: raise ChallengerBlocked("posthoc write failed")

def evaluate(*,artifact_root:Path,raw_manifest:Path,latch:Path,dataset_root:Path,evaluator:Path,run_id:str,admit:Callable[...,Any]=preflight,writer:Callable[...,Any]=atomic_write,evaluator_fn:Callable[...,Any]=local_au_pro_0_05)->dict[str,Any]:
    root=Path(artifact_root).resolve(); manifest, manifest_sha=_read_canonical(raw_manifest,root,f"{MAP_PREFIX}-{run_id}-"); latched,latch_sha=_read_canonical(latch,root,LATCH_PREFIX+"-")
    rows,lineage=manifest.get("maps"),manifest.get("lineage")
    latch_keys={"status","run_id","command","preflight_sha256","recipe_sha256","weight_sha256","train_bank_sha256","validation_parity_sha256","evaluator_sha256","runner_source_sha256","code_git_commit"}
    lineage_keys={"preflight_sha256","weight_sha256","recipe_sha256","coreset_seed","coreset_seed_derivation","train_bank_sha256","frozen_train_sha256","frozen_validation_sha256","evaluator_sha256","runner_source_sha256","code_git_commit"}
    if manifest.get("run_id") != run_id or manifest.get("status")!="SUPERADD_TEST_PUBLIC_RAW_MAPS" or not isinstance(rows,list) or len(rows)!=GOOD_COUNT+BAD_COUNT or not isinstance(lineage,Mapping) or set(latched)!=latch_keys or set(lineage)!=lineage_keys or latched.get("status")!="TEST_PUBLIC_INITIAL_ATTEMPT_LATCH" or latched.get("command")!="superadd-pinned-vits-evidence-comparison" or latched.get("run_id")!=run_id: raise ChallengerBlocked("historical raw manifest/latch schema mismatch")
    for key in ("preflight_sha256","recipe_sha256","weight_sha256","train_bank_sha256","evaluator_sha256","runner_source_sha256","code_git_commit"):
        if latched.get(key)!=lineage.get(key): raise ChallengerBlocked("historical latch/lineage binding mismatch")
    if _hash(evaluator)!=latched.get("evaluator_sha256") or _git_source(str(lineage.get("code_git_commit"))) != lineage.get("runner_source_sha256"): raise ChallengerBlocked("historical evaluator/runner source binding mismatch")
    entries=test_public_entries(dataset_root)
    if len(entries)!=len(rows): raise ChallengerBlocked("TESTpub entry count mismatch")
    derived=[]; maps=[]; masks=[]; stats=[]
    for i,(row,entry) in enumerate(zip(rows,entries)):
        if row.get("id_sha256")!=_anon(entry["image_identity"]) or row.get("label")!=entry["label"] or row.get("source_sha256")!=entry["source_sha256"] or row.get("mask_sha256")!=entry["mask_sha256"]: raise ChallengerBlocked("historical entry binding mismatch")
        digest=row.get("map_sha256")
        if row.get("shape") != [1056,4224] or row.get("dtype") != "<f4" or row.get("byte_order") != "<" or not isinstance(digest,str): raise ChallengerBlocked("historical raw schema mismatch")
        raw=(root/f"{MAP_PREFIX}-{i:03d}-{digest}.bin").read_bytes()
        if sha256(raw).hexdigest()!=digest or len(raw) != 1056*4224*4: raise ChallengerBlocked("historical raw hash mismatch")
        mapped=_resize_map(raw,row["shape"]); body=mapped.tobytes(); derived_digest=sha256(body).hexdigest(); target=root/f"{DERIVED_PREFIX}-{run_id}-{i:03d}-{derived_digest}.bin"
        if not target.exists(): _write(root,run_id,target,body,admit=admit,writer=writer,component="superadd-posthoc-derived-map")
        elif _hash(target)!=derived_digest or target.read_bytes()!=body: raise ChallengerBlocked("derived map hash mismatch")
        derived.append({"id_sha256":row["id_sha256"],"label":row["label"],"source_sha256":row["source_sha256"],"mask_sha256":row["mask_sha256"],"map_sha256":derived_digest,"shape":list(SPLIT_TARGET_SHAPE),"dtype":"<f4","byte_order":"<","original_latency_seconds":row.get("latency_seconds")})
        maps.append(mapped); masks.append(None if entry["mask"] is None else np.asarray(Image.open(entry["mask"]).resize((SPLIT_TARGET_SHAPE[1],SPLIT_TARGET_SHAPE[0]),Image.Resampling.NEAREST))); stats.append({"label":row["label"],"max":float(mapped.max())})
    payload=_canonical({"status":"SUPERADD_POSTHOC_DERIVED_RAW_MAPS","run_id":run_id,"original_manifest_sha256":manifest_sha,"latch_sha256":latch_sha,"lineage":dict(lineage),"maps":derived})
    derived_manifest=root/f"{DERIVED_PREFIX}-{run_id}-{sha256(payload).hexdigest()}.json"
    if not derived_manifest.exists(): _write(root,run_id,derived_manifest,payload,admit=admit,writer=writer,component="superadd-posthoc-derived-manifest")
    elif derived_manifest.read_bytes() != payload: raise ChallengerBlocked("derived manifest bytes mismatch")
    record={"status":READY,"command":COMMAND,"original_manifest_sha256":manifest_sha,"derived_manifest_sha256":_hash(derived_manifest),"latch_sha256":latch_sha,"lineage":dict(lineage),"local_au_pro_0_05":evaluator_fn(maps,masks,evaluator,include_curve=False),"image_auroc_tie_aware":_image_auroc({str(i):v for i,v in enumerate(stats)},normal_label="good",anomaly_label="bad"),"threshold_metrics":"NONE","protocol":"POSTHOC_RAW_MAP_ONLY_BILINEAR_528x2112_NO_SOURCE_INFERENCE","target_shape":list(SPLIT_TARGET_SHAPE),"posthoc_evaluator_sha256":_hash(evaluator),"posthoc_source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),"posthoc_git_commit":subprocess.run(["git","-C",str(Path(__file__).resolve().parents[2]),"rev-parse","HEAD"],check=True,text=True,capture_output=True).stdout.strip(),"original_latency_seconds":[row.get("latency_seconds") for row in rows],"limitation":"Inference completed once; only evaluation geometry normalized posthoc."}
    if not _path_free(record): raise ChallengerBlocked("posthoc evidence privacy violation")
    final=_canonical(record); target=root/f"superadd-vits-posthoc-evidence-{run_id}-{sha256(final).hexdigest()}.json"
    if not target.exists(): _write(root,run_id,target,final,admit=admit,writer=writer,component="superadd-posthoc-final")
    elif target.read_bytes() != final: raise ChallengerBlocked("posthoc final bytes mismatch")
    return {**record,"artifact_sha256":sha256(final).hexdigest()}

def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("artifact-root","raw-manifest","latch","dataset-root","evaluator"): p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--run-id",required=True); a=p.parse_args(argv)
    try:
        result=evaluate(**{k.replace("-","_"):v for k,v in vars(a).items()})
        print(json.dumps({"status":result["status"],"artifact_sha256":result["artifact_sha256"]},sort_keys=True)); return 0
    except Exception as exc:
        print(json.dumps({"status":"STOPPED_INCOMPLETE","exception_type":type(exc).__name__,"exception_fingerprint":sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()[:16]},sort_keys=True)); return 2
if __name__=="__main__": raise SystemExit(main())
