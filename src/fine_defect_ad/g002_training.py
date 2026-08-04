"""G002 fixed-schedule training admission and lightweight runtime instrumentation."""
from __future__ import annotations

import argparse, csv, json, math, os, signal, time, subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .g002_pilot import G002Args, _lazy_runtime, verify_local_assets, train_val_file_identity, _sha256
from .gpu_lock import GpuLease
from .storage import Allocation, atomic_write, preflight
from .pilot import MAX_STEPS, READY, STOPPED_INCOMPLETE, expected_pilot_protocol_metadata, host_rss_bytes, lease_events

PILOT_SHA256 = "0a5fc82e0e306cdd34ac8e5ee925e895010945816af6645dea4eb5be8aa9013c"
DECISION_ID = "DEC-TRN-002"
RPO_SECONDS = 300

class TrainingBlocked(ValueError): pass

@dataclass(frozen=True)
class TrainingArgs:
    pilot_evidence: Path
    run_id: str
    checkpoint_directory: Path
    metrics_path: Path
    g002: G002Args
    resume_checkpoint: Path | None = None


def file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def training_identity(args: G002Args) -> dict[str, Any]:
    assets = verify_local_assets(args)
    lock = Path(__file__).resolve().parents[2] / "requirements/r1-overlay.txt"
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    return {"teacher_sha256": assets["teacher_small"]["sha256"], "imagenette": "verified-local-imagefolder",
            "data": assets["file_identity"], "protocol": expected_pilot_protocol_metadata(),
            "overlay_lock_sha256": file_sha256(lock), "git": git, "git_dirty": dirty,
            "schedule": {"max_steps": MAX_STEPS, "max_epochs": 1000}}

def require_artifact_child(path: Path, artifact: Path) -> Path:
    resolved, root = Path(path).resolve(), Path(artifact).resolve()
    try: resolved.relative_to(root)
    except ValueError as exc: raise TrainingBlocked("output must be under admitted artifact root") from exc
    return resolved

def derived_write_bound(identity: Mapping[str, Any]) -> tuple[int, str]:
    # Exact serialized identity is the durable sidecar/evidence floor; checkpoint probe is required before READY.
    size = len(json.dumps(dict(identity), sort_keys=True).encode())
    return max(4096, size * 4), "serialized identity sidecar/evidence bytes x4; checkpoint size measured after save"


def admit_pilot(path: Path) -> dict[str, Any]:
    """Exact content-hash and protocol gate; this never reads a test/OOD path."""
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != PILOT_SHA256:
        raise TrainingBlocked("pilot artifact hash is not the approved G002 READY artifact")
    record = json.loads(raw)
    expected = expected_pilot_protocol_metadata()
    if (record.get("status") != READY or record.get("completed_steps") != 1000
            or record.get("gradient_finite") is not True or record.get("termination_cause") is not None
            or {key: record.get(key) for key in expected} != expected):
        raise TrainingBlocked("pilot status/protocol/finite-step gate failed")
    return record


def checkpoint_interval_steps(pilot: Mapping[str, Any]) -> int:
    median = pilot.get("median_seconds_per_step")
    if not isinstance(median, (int, float)) or not math.isfinite(median) or median <= 0:
        raise TrainingBlocked("pilot median step time is required for checkpoint RPO")
    return max(1, int(RPO_SECONDS / median))


def resume_sidecar(checkpoint: Path, pilot_hash: str, identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"checkpoint_sha256": file_sha256(checkpoint), "pilot_sha256": pilot_hash,
            "identity": dict(identity or {}), "decision_id": DECISION_ID, "resume_exactness": "NOT_ESTABLISHED"}

def validate_resume(checkpoint: Path, sidecar: Path, identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = json.loads(Path(sidecar).read_text())
    if value != resume_sidecar(checkpoint, PILOT_SHA256, identity):
        raise TrainingBlocked("checkpoint sidecar identity gate failed")
    return value

def public_attempt(run_id: str, pilot: Mapping[str, Any], *, cause: str | None = None) -> dict[str, Any]:
    return {"run_id": run_id, "status": READY if cause is None else STOPPED_INCOMPLETE,
            "limitations": [] if cause is None else [cause], "decision_id": DECISION_ID,
            "pilot_sha256": PILOT_SHA256, "pilot_median_seconds": pilot["median_seconds_per_step"],
            "schedule": {"max_steps": MAX_STEPS, "early_stopping": "forbidden", "checkpoint_selection": "rolling-last-only"},
            "resume_exactness": "NOT_ESTABLISHED", "termination_cause": cause}


def run_training(args: TrainingArgs) -> dict[str, Any]:
    """Full fixed-schedule run; only integrity failures stop it before 70k."""
    try:
        pilot = admit_pilot(args.pilot_evidence); interval = checkpoint_interval_steps(pilot)
        identity = training_identity(args.g002)
        if args.resume_checkpoint: validate_slot_resume(args.resume_checkpoint, identity)
        bound, source = derived_write_bound(identity)
        proof = preflight(run_id=args.run_id, allocations=[Allocation("artifact", bound, "persistent", source, "g002-sidecar-evidence"), Allocation("artifact", bound, "transient", source, "g002-incoming")], reserve_bytes=0, reserve_evidence={"max_pending_atomic_write_bytes":0,"measured_high_water_bytes":0,"runtime_or_source_citation":source})
        artifact = Path(proof.roots["artifact"]); require_artifact_child(args.g002.lease_directory, artifact); require_artifact_child(args.checkpoint_directory, artifact); require_artifact_child(args.metrics_path, artifact)
    except Exception as exc:
        return public_attempt(args.run_id, {"median_seconds_per_step": 0.0}, cause=f"PROVENANCE:{type(exc).__name__}")
    cause = None
    artifact_hashes: dict[str, str] = {}
    try:
        with GpuLease(args.g002.lease_directory, args.run_id, "g002-training", defer_signals=True) as lease:
            import torch
            from lightning.pytorch import Callback
            artifacts = TrainingArtifacts(Path(proof.roots["artifact"]), args.run_id, identity)
            pending = {"signal": None}
            class SafetyMetrics(Callback):
                def __init__(self): self.started = time.monotonic(); self.rows = []
                def on_before_optimizer_step(self, trainer, module, optimizer):
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()): raise RuntimeError("NONFINITE_GRADIENT")
                def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                    if trainer.global_step and trainer.global_step % interval == 0:
                        artifacts.checkpoint(trainer.global_step, lambda: _checkpoint_bytes(trainer))
                    if lease.pending_signal:
                        artifacts.checkpoint(trainer.global_step, lambda: _checkpoint_bytes(trainer)); trainer.should_stop = True
                def on_train_epoch_end(self, trainer, module):
                    step = trainer.global_step; elapsed = max(time.monotonic()-self.started, 1e-9)
                    row = {"epoch":trainer.current_epoch,"step":step,"lr":float(trainer.optimizers[0].param_groups[0]["lr"]),"rss_bytes":host_rss_bytes(),"vram_allocated":torch.cuda.max_memory_allocated(),"vram_reserved":torch.cuda.max_memory_reserved(),"throughput_steps_per_second":step/elapsed,"rolling_eta_seconds":max(0,MAX_STEPS-step)/(step/elapsed)}
                    if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int,float))): raise RuntimeError("NONFINITE_METRIC")
                    self.rows.append(row); artifacts.metrics(self.rows)
            from .pilot import PilotEvidence
            model, module, trainer, _ = _lazy_runtime(args.g002, PilotEvidence(args.run_id, "g002-training", MAX_STEPS), time.monotonic(), pilot_steps=None)
            safety = SafetyMetrics()
            trainer.callbacks.append(safety)
            trainer.fit(model, datamodule=module, ckpt_path=str(args.resume_checkpoint) if args.resume_checkpoint else None)
            checkpoint_path, sidecar_path = artifacts.checkpoint(trainer.global_step, lambda: _checkpoint_bytes(trainer))
            metrics_path = artifacts.metrics(safety.rows)
            if lease.pending_signal: cause = "INTERRUPTED_RESUMABLE"
            elif trainer.global_step != MAX_STEPS: cause = f"INCOMPLETE_STEPS:{trainer.global_step}_OF_{MAX_STEPS}"
            else: artifacts.final({"run_id":args.run_id,"status":READY,"checkpoint_sha256":file_sha256(checkpoint_path),"metrics_sha256":file_sha256(metrics_path),"resume_exactness":"NOT_ESTABLISHED"})

    except Exception as exc:
        cause = "OOM" if "out of memory" in str(exc).lower() else f"RUNNER:{type(exc).__name__}"
    try:
        expected = "normal" if cause is None else (f"signal:{lease.pending_signal}" if 'lease' in locals() and lease.pending_signal else "exception")
        events = lease_events(args.g002.lease_directory, args.run_id)
        validate_training_lease(events, args.run_id, expected)
        record = public_attempt(args.run_id, pilot, cause=cause); record["checkpoint_interval_steps"] = interval
        record["artifacts"] = artifact_hashes
        final = artifacts.final({"run_id":args.run_id,"status":record["status"],"artifacts":artifact_hashes,"lease_outcome":expected})
        record["artifacts"]["final"] = file_sha256(final)
        return record
    except Exception as exc:
        record = public_attempt(args.run_id, pilot, cause=f"EVIDENCE:{type(exc).__name__}"); record["checkpoint_interval_steps"] = interval
        return record

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot-evidence", type=Path, required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True); parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--teacher-small", type=Path, required=True); parser.add_argument("--imagenette-root", type=Path, required=True); parser.add_argument("--lease-directory", type=Path, required=True)
    raw = parser.parse_args(argv); g002 = G002Args(raw.dataset_root, raw.teacher_small, raw.imagenette_root, raw.run_id, raw.lease_directory)
    result = run_training(TrainingArgs(raw.pilot_evidence, raw.run_id, raw.checkpoint_directory, raw.metrics_path, g002)); print(json.dumps(result, sort_keys=True)); return 0 if result["status"] == READY else 2

def _checkpoint_bytes(trainer: Any) -> bytes:
    """Serialize Lightning's full connector checkpoint (loops/optimizers/RNG), not weights-only."""
    import io, torch
    output = io.BytesIO()
    checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
    torch.save(checkpoint, output)
    return output.getvalue()


@dataclass
class TrainingArtifacts:
    """Single proof-bound writer for G002 checkpoint, sidecar, metrics and attempt evidence."""
    artifact_root: Path
    run_id: str
    identity: Mapping[str, Any]
    admit: Any = preflight
    write: Any = atomic_write

    def __post_init__(self) -> None:
        self._slot = 0
        self.artifact_root = Path(self.artifact_root).resolve()
        self.identity_hash = sha256(json.dumps(dict(self.identity), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _commit(self, name: str, payload: bytes, *, overwrite: bool = True) -> Path:
        path = require_artifact_child(self.artifact_root / name, self.artifact_root)
        source = f"exact in-memory {name} bytes={len(payload)} sha256={sha256(payload).hexdigest()}"
        proof = self.admit(run_id=self.run_id, allocations=[Allocation("artifact", len(payload), "persistent", source, "g002-training-artifact-final"), Allocation("artifact", len(payload), "transient", source, "g002-training-artifact-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes":len(payload),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
        if Path(proof.roots["artifact"]).resolve() != self.artifact_root:
            raise TrainingBlocked("fresh proof artifact root changed")
        result = self.write(path, payload, proof=proof, run_id=self.run_id, overwrite=overwrite)
        if result.get("status") != READY or not path.is_file() or file_sha256(path) != sha256(payload).hexdigest():
            raise TrainingBlocked("artifact commit verification failed")
        return path

    def checkpoint(self, step: int, serialize_checkpoint: Any) -> tuple[Path, Path]:
        if not 0 <= step <= MAX_STEPS: raise TrainingBlocked("checkpoint step invalid")
        payload = serialize_checkpoint()
        if not isinstance(payload, bytes) or not payload: raise TrainingBlocked("checkpoint serializer returned no bytes")
        slot = self._slot % 2
        checkpoint = self._commit(f"g002-last-{self.run_id}-{slot}.ckpt", payload)
        sidecar = {"checkpoint_sha256": sha256(payload).hexdigest(), "identity_sha256": self.identity_hash,
                   "pilot_sha256": PILOT_SHA256, "global_step": step, "lineage": self.run_id,
                   "resume_exactness": "NOT_ESTABLISHED", "checkpoint_name": checkpoint.name}
        result = checkpoint, self._commit(f"g002-last-{self.run_id}-{slot}.ckpt.json", json.dumps(sidecar,sort_keys=True).encode())
        self._slot += 1
        return result

    def metrics(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        if len(rows) > 511: raise TrainingBlocked("metrics snapshot exceeds 511 epoch rows")
        return self._commit(f"g002-metrics-{self.run_id}.json", json.dumps(list(rows),sort_keys=True,allow_nan=False).encode())

    def final(self, record: Mapping[str, Any]) -> Path:
        return self._commit(f"g002-attempt-{self.run_id}-{sha256(json.dumps(dict(record),sort_keys=True).encode()).hexdigest()}.json", json.dumps(dict(record),sort_keys=True,allow_nan=False).encode(), overwrite=False)

def validate_slot_resume(checkpoint: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    value = json.loads(sidecar.read_text())
    required = {"checkpoint_name", "checkpoint_sha256", "identity_sha256", "pilot_sha256", "global_step", "lineage", "resume_exactness"}
    if (not required <= value.keys() or value["checkpoint_name"] != checkpoint.name
            or value["checkpoint_sha256"] != file_sha256(checkpoint)
            or value["identity_sha256"] != sha256(json.dumps(dict(identity),sort_keys=True,separators=(",",":")).encode()).hexdigest()
            or value["pilot_sha256"] != PILOT_SHA256 or not 0 <= value["global_step"] < MAX_STEPS):
        raise TrainingBlocked("slot checkpoint resume identity gate failed")
    return value


def validate_training_lease(events: Sequence[Mapping[str, Any]], run_id: str, outcome: str) -> None:
    if len(events) != 2 or [event.get("state") for event in events] != ["acquired", "released"]:
        raise TrainingBlocked("training lease lifecycle missing")
    if any(event.get("run_id") != run_id or event.get("command") != "g002-training" for event in events) or events[1].get("outcome") != outcome:
        raise TrainingBlocked("training lease outcome mismatch")

if __name__ == "__main__":
    raise SystemExit(main())
