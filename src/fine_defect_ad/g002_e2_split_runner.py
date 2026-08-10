"""DEC-SPLIT-003: validation freeze and one-shot post-hoc TESTpub correction."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from .g002_e2_runtime import (SPLIT_DECISION_ID, SPLIT_TARGET_SHAPE, combine_split_maps, decode_rgb01,
                               freeze_split_validation, split_branch_raw_maps, split_quantiles, verify_split_freeze)
from .g002_evaluate import admit_completed_checkpoint, _hash
from .g002_eval_runtime import load_training_identity, safe_load_checkpoint
from .g002_pilot import G002Args, _lazy_runtime
from .g002_testpub_runtime import _canon, evaluate_persisted_split_test_public, test_public_entries
from .gpu_lock import GpuLease
from .pilot import PilotEvidence
from .storage import Allocation, READY, atomic_write, preflight

VALIDATION_COMMAND = "g002-e2-split-validation-freeze"
TEST_COMMAND = "g002-e2-split-test-public-once"


def verify_split_lineage(freeze: dict[str, Any], admitted: Any) -> None:
    for field in ("checkpoint_sha256", "sidecar_sha256", "metrics_sha256", "final_attempt_sha256", "identity_sha256", "pilot_sha256"):
        if field in freeze and freeze[field] != getattr(admitted, field):
            raise ValueError(f"split freeze/model lineage mismatch: {field}")


def safe_failure_fields(exc: Exception) -> dict[str, str]:
    """Correlate failures without serializing possibly path-bearing exception text."""
    kind = type(exc).__name__
    return {"exception_type": kind, "exception_fingerprint_sha256": sha256(f"{kind}:{exc}".encode()).hexdigest(),
            "exception_message": "Execution failed; the original exception was re-raised unchanged."}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    for key in ("artifact-root", "checkpoint", "metrics", "final-attempt", "training-identity", "dataset-root", "teacher-small", "imagenette-root", "lease-directory"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--run-id", required=True)


def _write(root: Path, run_id: str, name: str, data: bytes) -> Path:
    source = f"exact DEC-SPLIT-003 artifact bytes={len(data)}"
    proof = preflight(run_id=run_id, allocations=[Allocation("artifact", len(data), "persistent", source, name), Allocation("artifact", len(data), "transient", source, name + "-incoming")], reserve_bytes=len(data), reserve_evidence={"max_pending_atomic_write_bytes":len(data),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
    target = root / name
    result = atomic_write(target, data, proof=proof, run_id=run_id, overwrite=False)
    if result.get("status") != READY or target.read_bytes() != data:
        raise ValueError("immutable artifact write failed")
    return target


def _model(args: Any, torch: Any):
    identity, run = load_training_identity(args.training_identity, args.artifact_root)
    admitted = admit_completed_checkpoint(args.checkpoint, args.artifact_root, identity, args.dataset_root, args.final_attempt, args.metrics)
    if run != admitted.run_id:
        raise ValueError("checkpoint/identity lineage mismatch")
    checkpoint = safe_load_checkpoint(admitted.path, admitted.checkpoint_sha256, torch)
    model, *_ = _lazy_runtime(G002Args(args.dataset_root,args.teacher_small,args.imagenette_root,args.run_id,args.lease_directory), PilotEvidence(args.run_id, "DEC-SPLIT-003", 70_000), 0.0, pilot_steps=None)
    model.load_state_dict(checkpoint["state_dict"]); model.eval(); model.to(torch.device("cuda:0"))
    return admitted, model


def _failure_log_path(root: Path, run_id: str, payload: bytes) -> Path:
    digest = sha256(payload).hexdigest()
    return root / f"g002-e2-split-testpub-FAILED-{run_id}-{digest}.json"


def _failure_logs(root: Path, run_id: str) -> list[Path]:
    return sorted(root.glob(f"g002-e2-split-testpub-FAILED-{run_id}-*"))


def _attempt_recovery(root: Path, run_id: str, latch_path: Path) -> dict[str, str] | None:
    logs = _failure_logs(root, run_id)
    if not logs:
        return None
    attempt_payload = {"attempt_latch_sha256": sha256(latch_path.read_bytes()).hexdigest(),
                      "original_failure_log_sha256": sha256(logs[-1].read_bytes()).hexdigest(),
                      "code_fix_commit": ""}
    root_dir = Path(__file__).resolve().parents[2]
    try:
        attempt_payload["code_fix_commit"] = subprocess.check_output(["git", "-C", str(root_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        attempt_payload["code_fix_commit"] = os.environ.get("GIT_COMMIT", "unknown")
    if not attempt_payload["code_fix_commit"]:
        attempt_payload["code_fix_commit"] = "unknown"
    return attempt_payload


def run_validation(args: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    root=Path(args.artifact_root).resolve()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
    with GpuLease(args.lease_directory,args.run_id,VALIDATION_COMMAND):
        admitted, model = _model(args, torch); rows=[]; locals_=[]; globals_=[]; geometry=None
        for index,(identity, source_hash) in enumerate(admitted.validation_identities):
            source=(admitted.dataset_root/"sheet_metal"/identity).resolve()
            if _hash(source) != source_hash: raise ValueError("validation source hash changed")
            local, global_, geometry = split_branch_raw_maps(decode_rgb01(source), model, torch, device=torch.device("cuda:0"))
            local_raw, global_raw = local.tobytes(), global_.tobytes()
            lh, gh = sha256(local_raw).hexdigest(), sha256(global_raw).hexdigest()
            _write(root,args.run_id,f"g002-e2-split-validation-{args.run_id}-st-{index:02d}-{lh}.bin",local_raw)
            _write(root,args.run_id,f"g002-e2-split-validation-{args.run_id}-stae-{index:02d}-{gh}.bin",global_raw)
            rows.append({"image_identity":identity,"source_sha256":source_hash,"local_st_sha256":lh,"global_stae_sha256":gh,"local_st_shape":list(local.shape),"global_stae_shape":list(global_.shape),"_local_st":local,"_global_stae":global_})
            locals_.append(local); globals_.append(global_)
        quantiles=split_quantiles(locals_,globals_,torch_module=torch)
        freeze=freeze_split_validation(admitted=admitted,quantiles=quantiles,map_rows=rows,geometry=geometry)
        path=_write(root,args.run_id,f"g002-e2-split-pretest-freeze-{args.run_id}-{freeze['freeze_sha256']}.json",_canon(freeze))
    return {"status":"READY","decision_id":SPLIT_DECISION_ID,"freeze":str(path),"freeze_sha256":freeze["freeze_sha256"],"quantiles":quantiles,"validation_maps":19}


def run_test_public_once(args: Any) -> dict[str, Any]:
    import torch
    root=Path(args.artifact_root).resolve(); freeze_path=Path(args.split_freeze).resolve()
    freeze=json.loads(freeze_path.read_text()); verify_split_freeze(freeze)
    latch_path = root / f"g002-e2-split-testpub-ATTEMPTED-{args.run_id}.json"
    if any(root.glob("g002-e2-split-testpub-evidence-*.json")):
        raise ValueError("TESTpub one-shot already consumed")
    if latch_path.exists() and not _failure_logs(root, args.run_id):
        raise ValueError("TESTpub one-shot already consumed")
    latch = latch_path if latch_path.exists() else _write(root,args.run_id,f"g002-e2-split-testpub-ATTEMPTED-{args.run_id}.json",_canon({"decision_id":SPLIT_DECISION_ID,"freeze_sha256":freeze["freeze_sha256"],"status":"ATTEMPTED"}))
    # The latch is persisted before the first test image is decoded.
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
    with GpuLease(args.lease_directory,args.run_id,TEST_COMMAND):
        _admitted, model=_model(args,torch); verify_split_lineage(freeze, _admitted); rows=[]
        recovery = _attempt_recovery(root, args.run_id, latch_path)
        try:
            for index, entry in enumerate(test_public_entries(args.dataset_root)):
                local,global_,_geometry=split_branch_raw_maps(decode_rgb01(entry["source"]),model,torch,device=torch.device("cuda:0"))
                value=combine_split_maps(local,global_,freeze["quantiles"],torch); raw=value.tobytes(); digest=sha256(raw).hexdigest()
                _write(root,args.run_id,f"g002-e2-split-test-public-raw-{index:03d}-{digest}.bin",raw)
                rows.append({"image_identity":entry["image_identity"],"label":entry["label"],"source_sha256":entry["source_sha256"],"mask_sha256":entry["mask_sha256"],"map_sha256":digest,"dtype":"<f4","byte_order":"<","shape":list(SPLIT_TARGET_SHAPE)})
            manifest={"status":"SPLIT_E2_TEST_PUBLIC_RAW_MAPS","decision_id":SPLIT_DECISION_ID,"freeze_sha256":freeze["freeze_sha256"],"official_evaluator_sha256":sha256(Path(args.evaluator).read_bytes()).hexdigest(),"maps":rows}
            manifest_path=_write(root,args.run_id,f"g002-e2-split-test-public-raw-maps-{args.run_id}.json",_canon(manifest))
            result=evaluate_persisted_split_test_public(artifact_root=root,dataset_root=args.dataset_root,raw_manifest=manifest_path,split_freeze=freeze_path,evaluator=args.evaluator,run_id=args.run_id,recovery=recovery)
        except Exception as exc:
            failure = _canon({"schema_version": "1.0", "operation": TEST_COMMAND, "run_id": args.run_id, "decision_id": SPLIT_DECISION_ID, **safe_failure_fields(exc)})
            _write(root, args.run_id, _failure_log_path(root, args.run_id, failure).name, failure)
            raise
    return {**result,"attempt_latch":str(latch)}


def parse_args(argv: Sequence[str] | None = None, *, description: str | None = None):
    p = argparse.ArgumentParser(description=description or __doc__); sub=p.add_subparsers(dest="operation",required=True)
    for name in ("validation","testpub"):
        q=sub.add_parser(name)
        add_common_arguments(q)
        if name=="testpub": q.add_argument("--split-freeze",type=Path,required=True);q.add_argument("--evaluator",type=Path,required=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None=None)->int:
    args=parse_args(argv); result=run_validation(args) if args.operation=="validation" else run_test_public_once(args); print(json.dumps(result,sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
