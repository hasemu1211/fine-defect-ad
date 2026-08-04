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
from .pilot import MAX_STEPS, READY, STOPPED_INCOMPLETE, expected_pilot_protocol_metadata, host_rss_bytes

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
            "schedule": {"max_steps": MAX_STEPS, "max_epochs": 1000}, "lineage": args.run_id}

def require_artifact_child(path: Path, artifact: Path) -> Path:
    resolved, root = Path(path).resolve(), Path(artifact).resolve()
    try: resolved.relative_to(root)
    except ValueError as exc: raise TrainingBlocked("output must be under admitted artifact root") from exc
    return resolved

def derived_write_bound(identity: Mapping[str, Any]) -> tuple[int, str]:
    # Exact serialized identity is the durable sidecar/evidence floor; checkpoint probe is required before READY.
    size = len(json.dumps(dict(identity), sort_keys=True).encode())
    return max(4096, size * 4), "serialized identity sidecar/evidence bytes x4; checkpoint size measured after save"


def _atomic_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Atomic cumulative JSONL snapshot: prior rows survive each epoch update."""
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_text() if path.exists() else ""
    payload = previous + json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


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
        if args.resume_checkpoint: validate_resume(args.resume_checkpoint, args.resume_checkpoint.with_suffix(args.resume_checkpoint.suffix + ".json"), identity)
        bound, source = derived_write_bound(identity)
        proof = preflight(run_id=args.run_id, allocations=[Allocation("artifact", bound, "persistent", source, "g002-sidecar-evidence"), Allocation("artifact", bound, "transient", source, "g002-incoming")], reserve_bytes=0, reserve_evidence={"max_pending_atomic_write_bytes":0,"measured_high_water_bytes":0,"runtime_or_source_citation":source})
        artifact = Path(proof.roots["artifact"]); require_artifact_child(args.checkpoint_directory, artifact); require_artifact_child(args.metrics_path, artifact)
    except Exception as exc:
        return public_attempt(args.run_id, {"median_seconds_per_step": 0.0}, cause=f"PROVENANCE:{type(exc).__name__}")
    cause = None
    try:
        with GpuLease(args.g002.lease_directory, args.run_id, "g002-training"):
            import torch
            from lightning.pytorch import Callback
            from lightning.pytorch.callbacks import ModelCheckpoint
            checkpoint = ModelCheckpoint(dirpath=args.checkpoint_directory, filename="last", save_last=True, save_top_k=0, every_n_train_steps=interval)
            pending = {"signal": None}
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            for sig in previous: signal.signal(sig, lambda signum, frame: pending.update(signal=signum))
            class SafetyMetrics(Callback):
                def __init__(self): self.started = time.monotonic()
                def on_before_optimizer_step(self, trainer, module, optimizer):
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()): raise RuntimeError("NONFINITE_GRADIENT")
                def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                    if pending["signal"]:
                        trainer.save_checkpoint(str(args.checkpoint_directory / "last.ckpt")); trainer.should_stop = True
                def on_train_epoch_end(self, trainer, module):
                    step = trainer.global_step; elapsed = max(time.monotonic()-self.started, 1e-9)
                    _atomic_jsonl(args.metrics_path, {"epoch":trainer.current_epoch,"step":step,"lr":float(trainer.optimizers[0].param_groups[0]["lr"]),"rss_bytes":host_rss_bytes(),"vram_allocated":torch.cuda.max_memory_allocated(),"vram_reserved":torch.cuda.max_memory_reserved(),"throughput_steps_per_second":step/elapsed,"rolling_eta_seconds":max(0,MAX_STEPS-step)/(step/elapsed)})
            from .pilot import PilotEvidence
            model, module, trainer, _ = _lazy_runtime(args.g002, PilotEvidence(args.run_id, "g002-training", MAX_STEPS), time.monotonic(), pilot_steps=None)
            trainer.callbacks.extend([checkpoint, SafetyMetrics()])
            trainer.fit(model, datamodule=module, ckpt_path=str(args.resume_checkpoint) if args.resume_checkpoint else None)
            for sig, old in previous.items(): signal.signal(sig, old)
            last = Path(checkpoint.last_model_path)
            if not last.is_file() or not args.metrics_path.is_file(): raise RuntimeError("MISSING_VERIFIED_CHECKPOINT_OR_METRICS")
            last.with_suffix(last.suffix+".json").write_text(json.dumps(resume_sidecar(last,PILOT_SHA256,identity),sort_keys=True))
            if pending["signal"]: cause = "INTERRUPTED_RESUMABLE"
            elif trainer.global_step != MAX_STEPS: cause = f"INCOMPLETE_STEPS:{trainer.global_step}_OF_{MAX_STEPS}"
    except Exception as exc:
        cause = "OOM" if "out of memory" in str(exc).lower() else f"RUNNER:{type(exc).__name__}"
    record = public_attempt(args.run_id, pilot, cause=cause); record["checkpoint_interval_steps"] = interval
    return record

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot-evidence", type=Path, required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True); parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--teacher-small", type=Path, required=True); parser.add_argument("--imagenette-root", type=Path, required=True); parser.add_argument("--lease-directory", type=Path, required=True)
    raw = parser.parse_args(argv); g002 = G002Args(raw.dataset_root, raw.teacher_small, raw.imagenette_root, raw.run_id, raw.lease_directory)
    result = run_training(TrainingArgs(raw.pilot_evidence, raw.run_id, raw.checkpoint_directory, raw.metrics_path, g002)); print(json.dumps(result, sort_keys=True)); return 0 if result["status"] == READY else 2
if __name__ == "__main__": raise SystemExit(main())
