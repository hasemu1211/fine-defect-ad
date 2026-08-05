"""Production wiring for G002's identity-bound validation raw-map collection only."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path, PosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .evidence import immutable_json, new_evidence
from .g002_evaluate import CATEGORY, admit_completed_checkpoint, collect_validation_maps, persist_validation_maps
from .g002_pilot import G002Args, LEASE_WRITE_BYTES, _lazy_runtime
from .gpu_lock import GpuLease
from .pilot import PilotEvidence, host_rss_bytes, lease_events
from .storage import Allocation, READY, STOPPED_INCOMPLETE, atomic_write, preflight

COMMAND = "g002-eval-validation-raw-maps"
TRANSFORM_IDENTITY = {"normalize": False, "resize": 256, "interpolation": "bilinear"}


@dataclass(frozen=True)
class EvaluationArgs:
    artifact_root: Path; checkpoint: Path; sidecar: Path; metrics: Path; final_attempt: Path
    training_identity: Path; dataset_root: Path; teacher_small: Path; imagenette_root: Path
    run_id: str; lease_directory: Path


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity_run_id(path: Path, digest: str) -> str:
    prefix, suffix = "g002-training-identity-", f"-{digest}.json"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError("training identity filename/content hash mismatch")
    run_id = path.name[len(prefix):-len(suffix)]
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("training identity run ID invalid")
    return run_id


def load_training_identity(path: Path, artifact_root: Path) -> tuple[dict[str, Any], str]:
    """Accept canonical identity bytes whose filename commits both lineage and content hash."""
    path, root = Path(path).resolve(), Path(artifact_root).resolve()
    try: path.relative_to(root)
    except ValueError as exc: raise ValueError("training identity must be under artifact root") from exc
    try: raw, value = path.read_bytes(), json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ValueError("invalid training identity artifact") from exc
    if not isinstance(value, dict) or raw != _canonical(value): raise ValueError("training identity must use canonical JSON bytes")
    return value, _identity_run_id(path, sha256(raw).hexdigest())


def safe_load_checkpoint(path: Path, expected_sha256: str, torch_module: Any | None = None) -> Mapping[str, Any]:
    path = Path(path).resolve()
    if sha256(path.read_bytes()).hexdigest() != expected_sha256: raise ValueError("checkpoint hash changed after admission")
    torch = torch_module
    if torch is None: import torch
    with torch.serialization.safe_globals([PosixPath]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping): raise ValueError("full Lightning checkpoint state_dict required")
    if payload.get("global_step") != 70_000: raise ValueError("checkpoint global_step must be 70000")
    return payload


def _one(value: Any) -> Any:
    if isinstance(value, (str, Path)): return value
    if isinstance(value, (list, tuple)) and len(value) == 1: return value[0]
    try:
        if getattr(value, "ndim", None) == 1 and len(value) == 1: return value[0]
    except TypeError: pass
    return value


def _batch_values(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping): return batch.get("image"), batch.get("image_path", batch.get("source_path"))
    # Anomalib ImageBatch is a dataclass. Keep this structural to avoid importing the overlay at CLI-help time.
    if is_dataclass(batch) and {field.name for field in fields(batch)} >= {"image", "image_path"}:
        return batch.image, batch.image_path
    raise ValueError("validation batch must be an ImageBatch or mapping")


def _pinned_transform(transform: Any) -> Any:
    """EfficientAD's only permitted validation transform is Resize(256, bilinear)."""
    transforms = getattr(transform, "transforms", None)
    if not isinstance(transforms, (list, tuple)) or len(transforms) != 1:
        raise ValueError("validation transform must be exactly one resize")
    resize = transforms[0]
    if type(resize).__name__ != "Resize" or tuple(getattr(resize, "size", ())) != (256, 256) or "bilinear" not in str(getattr(resize, "interpolation", "")).lower():
        raise ValueError("validation transform must be Resize(256, bilinear)")
    return transform


def _validation_batches(loader: Iterable[Any], dataset_root: Path, device: Any) -> Iterable[dict[str, Any]]:
    leaf = (Path(dataset_root).resolve() / CATEGORY / "validation" / "good").resolve()
    for batch in loader:
        image, source = _batch_values(batch); source = _one(source)
        if image is None or not isinstance(source, (str, Path)): raise ValueError("validation batch image/source path required")
        shape = tuple(getattr(image, "shape", ()))
        if shape != (1, 3, 256, 256): raise ValueError("validation image must be 1x3x256x256")
        if not hasattr(image, "to"): raise ValueError("validation image must be a tensor")
        source = Path(source).resolve()
        try: identity = source.relative_to(leaf).as_posix()
        except ValueError as exc: raise ValueError("validation batch escapes validation/good") from exc
        yield {"image": image.to(device), "image_identity": f"validation/good/{identity}", "source_path": source}


class _MapModel:
    def __init__(self, model: Any, torch_module: Any) -> None: self._model, self._torch = model, torch_module
    def get_maps(self, image: Any, *, normalize: bool) -> Any:
        if normalize is not False: raise ValueError("normalized maps forbidden")
        with self._torch.inference_mode(): return self._model.model.get_maps(image, normalize=False)


def _vram(torch: Any) -> dict[str, int]:
    return {"allocated_bytes": int(torch.cuda.max_memory_allocated()), "reserved_bytes": int(torch.cuda.max_memory_reserved())}


def _final_evidence(root: Path, run_id: str, record: dict[str, Any], admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
    payload, digest = immutable_json(record); source = f"exact immutable evaluation evidence bytes={len(payload)} sha256={digest}"
    proof = admit(run_id=run_id, allocations=[Allocation("artifact", len(payload), "persistent", source, "g002-eval-final-evidence"), Allocation("artifact", len(payload), "transient", source, "g002-eval-final-evidence-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ValueError("fresh proof artifact root changed")
    final = root / f"g002-eval-validation-raw-evidence-{run_id}-{digest}.json"; result = writer(final, payload, proof=proof, run_id=run_id, overwrite=False)
    if result.get("status") != READY or final.read_bytes() != payload: raise ValueError("immutable final evidence write failed")
    return {**record, "artifact": str(final), "artifact_sha256": digest}


def _lease_proof(root: Path, args: EvaluationArgs, admit: Callable[..., Any]) -> None:
    """Validate storage before GpuLease creates lock/holder/event files."""
    source = f"G002 evaluation GpuLease bounded writes <= {LEASE_WRITE_BYTES} bytes"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", LEASE_WRITE_BYTES, "persistent", source, "g002-eval-gpu-lease"), Allocation("artifact", LEASE_WRITE_BYTES, "transient", source, "g002-eval-gpu-lease-incoming")], reserve_bytes=LEASE_WRITE_BYTES, reserve_evidence={"max_pending_atomic_write_bytes": LEASE_WRITE_BYTES, "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ValueError("fresh proof artifact root changed")


def _lease_record(events: Sequence[Mapping[str, Any]], run_id: str, outcome: str, *, expected_command: str = COMMAND) -> list[dict[str, str]]:
    events = [event for event in events if event.get("run_id") == run_id and event.get("command") == expected_command]
    if len(events) != 2 or [event.get("state") for event in events] != ["acquired", "released"]:
        raise ValueError("evaluation lease lifecycle missing")
    if events[1].get("outcome") != outcome:
        raise ValueError("evaluation lease outcome mismatch")
    try:
        acquired, released = (datetime.fromisoformat(str(event["timestamp"])) for event in events)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evaluation lease timestamps invalid") from exc
    if acquired > released: raise ValueError("evaluation lease timestamps out of order")
    return [{"state": "acquired", "timestamp": str(events[0]["timestamp"])}, {"state": "released", "timestamp": str(events[1]["timestamp"]), "outcome": outcome}]


def run_evaluation(args: EvaluationArgs, *, runtime_factory: Callable[..., Any] = _lazy_runtime, lease_factory: Callable[..., Any] = GpuLease,
                   torch_module: Any | None = None, admit: Callable[..., Any] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write,
                   lease_event_loader: Callable[..., Sequence[Mapping[str, Any]]] = lease_events) -> dict[str, Any]:
    """GPU-only, validation-only E1 map extraction; all admitted failures emit STOPPED evidence."""
    root = Path(args.artifact_root).resolve(); started = time.monotonic(); admitted = None; persisted = None; failure = None
    lease_outcome, recorded_lease, lease_entered = "not_acquired", None, False
    torch = torch_module
    try:
        # This proof is deliberately first: GpuLease itself persists material files.
        _lease_proof(root, args, admit)
        lease_directory = Path(args.lease_directory).resolve()
        try:
            relative = lease_directory.relative_to(root)
        except ValueError as exc:
            raise ValueError("lease_directory must be an artifact-root descendant") from exc
        if not relative.parts: raise ValueError("lease_directory must be beneath artifact root")
        identity, identity_run_id = load_training_identity(args.training_identity, root)
        if Path(args.sidecar).resolve() != Path(args.checkpoint).resolve().with_suffix(Path(args.checkpoint).suffix + ".json"):
            raise ValueError("sidecar must be the selected checkpoint sidecar")
        admitted = admit_completed_checkpoint(args.checkpoint, root, identity, args.dataset_root, args.final_attempt, args.metrics)
        if identity_run_id != admitted.run_id: raise ValueError("identity artifact lineage does not match admitted checkpoint")
        if sha256(Path(args.sidecar).read_bytes()).hexdigest() != admitted.sidecar_sha256: raise ValueError("admitted sidecar changed")
        if torch is None: import torch
        checkpoint = safe_load_checkpoint(admitted.path, admitted.checkpoint_sha256, torch)
        try:
            with lease_factory(lease_directory, args.run_id, COMMAND):
                lease_entered = True
                if not bool(torch.cuda.is_available()): raise RuntimeError("CUDA_UNAVAILABLE")
                device = torch.device("cuda:0")
                g002 = G002Args(args.dataset_root, args.teacher_small, args.imagenette_root, args.run_id, lease_directory)
                model, datamodule, _trainer, _validator = runtime_factory(g002, PilotEvidence(args.run_id, COMMAND, 70_000), started, pilot_steps=None)
                model.load_state_dict(checkpoint["state_dict"]); model.eval(); model.to(device); datamodule.setup("validate")
                datamodule.val_data.augmentations = _pinned_transform(model.pre_processor.transform)
                collected = collect_validation_maps(_MapModel(model, torch), _validation_batches(datamodule.val_dataloader(), admitted.dataset_root, device), admitted)
                persisted = persist_validation_maps(collected, root, args.run_id, TRANSFORM_IDENTITY, admit=admit, writer=writer)
            lease_outcome = "normal"
        except Exception as exc:
            failure, lease_outcome = f"RUNNER:{type(exc).__name__}:{exc}", "exception"
        if lease_entered:
            try: recorded_lease = _lease_record(lease_event_loader(lease_directory, args.run_id), args.run_id, lease_outcome)
            except Exception as exc: failure, lease_outcome = f"LEASE_EVIDENCE:{type(exc).__name__}:{exc}", "invalid"
    except Exception as exc:
        failure = f"ADMISSION:{type(exc).__name__}:{exc}"
    status = READY if failure is None else STOPPED_INCOMPLETE
    record = new_evidence(args.run_id, COMMAND, status, [] if failure is None else [failure])
    record.update({"transform_identity": TRANSFORM_IDENTITY, "timing_seconds": time.monotonic() - started,
                   "rss_bytes": host_rss_bytes(), "lease_outcome": lease_outcome})
    if torch is not None:
        try: record["vram"] = _vram(torch)
        except Exception: record["vram"] = {"allocated_bytes": None, "reserved_bytes": None}
    if admitted is not None: record.update({"checkpoint_sha256": admitted.checkpoint_sha256, "identity_sha256": admitted.identity_sha256})
    if persisted is not None: record["raw_maps"] = persisted
    if recorded_lease is not None: record["lease_events"] = recorded_lease
    try: return _final_evidence(root, args.run_id, record, admit, writer)
    except Exception as exc:
        return {**record, "status": STOPPED_INCOMPLETE, "limitations": [*record["limitations"], f"EVIDENCE:{type(exc).__name__}"]}

def parse_args(argv: Sequence[str] | None = None) -> EvaluationArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "checkpoint", "sidecar", "metrics", "final-attempt", "training-identity", "dataset-root", "teacher-small", "imagenette-root", "lease-directory"): parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True); raw = parser.parse_args(argv)
    return EvaluationArgs(**{key.replace("-", "_"): value for key, value in vars(raw).items()})


def main(argv: Sequence[str] | None = None) -> int:
    result = run_evaluation(parse_args(argv)); print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0 if result["status"] == READY else 2

if __name__ == "__main__": raise SystemExit(main())
