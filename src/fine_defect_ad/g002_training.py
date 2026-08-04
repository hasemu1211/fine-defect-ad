"""G002 fixed-schedule training admission and lightweight runtime instrumentation."""
from __future__ import annotations

import argparse, csv, json, math, os, signal, time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .g002_pilot import G002Args, _lazy_runtime
from .gpu_lock import GpuLease
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


def resume_sidecar(checkpoint: Path, pilot_hash: str) -> dict[str, str]:
    return {"checkpoint_sha256": file_sha256(checkpoint), "pilot_sha256": pilot_hash,
            "decision_id": DECISION_ID, "resume_exactness": "NOT_ESTABLISHED"}


def validate_resume(checkpoint: Path, sidecar: Path) -> dict[str, str]:
    value = json.loads(Path(sidecar).read_text())
    if value != resume_sidecar(checkpoint, PILOT_SHA256):
        raise TrainingBlocked("checkpoint sidecar identity gate failed")
    return value


def public_attempt(run_id: str, pilot: Mapping[str, Any], *, cause: str | None = None) -> dict[str, Any]:
    return {"run_id": run_id, "status": READY if cause is None else STOPPED_INCOMPLETE,
            "limitations": [] if cause is None else [cause], "decision_id": DECISION_ID,
            "pilot_sha256": PILOT_SHA256, "pilot_median_seconds": pilot["median_seconds_per_step"],
            "schedule": {"max_steps": MAX_STEPS, "early_stopping": "forbidden", "checkpoint_selection": "rolling-last-only"},
            "resume_exactness": "NOT_ESTABLISHED", "termination_cause": cause}


def run_training(args: TrainingArgs) -> dict[str, Any]:
    """Lazy full runner. No validation/test/predict or model selection is invoked."""
    try:
        pilot = admit_pilot(args.pilot_evidence)
        interval = checkpoint_interval_steps(pilot)
        if args.resume_checkpoint:
            validate_resume(args.resume_checkpoint, args.resume_checkpoint.with_suffix(args.resume_checkpoint.suffix + ".json"))
    except Exception as exc:
        return public_attempt(args.run_id, {"median_seconds_per_step": 0.0}, cause=f"PROVENANCE:{type(exc).__name__}")
    # Runtime-only imports are inside the GPU lease, after admission.
    cause = None
    try:
        with GpuLease(args.g002.lease_directory, args.run_id, "g002-training"):
            from lightning.pytorch import Callback
            class Metrics(Callback):
                def __init__(self): self.rows: list[dict[str, Any]] = []
                def on_train_epoch_end(self, trainer, module):
                    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if "loss" in k or k == "lr"}
                    self.rows.append({"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                                      "rss_bytes": host_rss_bytes(), **metrics})
            # Reuse the scoped train/validation-only G002 model/datamodule; no pilot stop callback.
            from .pilot import PilotEvidence
            model, module, trainer, _validator = _lazy_runtime(args.g002, PilotEvidence(args.run_id, "g002-training", MAX_STEPS), time.monotonic(), pilot_steps=None)
            metrics = Metrics(); trainer.callbacks.append(metrics)
            trainer.fit(model, datamodule=module, ckpt_path=str(args.resume_checkpoint) if args.resume_checkpoint else None)
            # Logs are small epoch rows; caller must preflight their bounded bytes before production launch.
            args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with args.metrics_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted({k for row in metrics.rows for k in row})); writer.writeheader(); writer.writerows(metrics.rows)
    except Exception as exc:
        cause = f"RUNNER:{type(exc).__name__}"
    record = public_attempt(args.run_id, pilot, cause=cause)
    record["checkpoint_interval_steps"] = interval
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot-evidence", type=Path, required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True); parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--teacher-small", type=Path, required=True); parser.add_argument("--imagenette-root", type=Path, required=True); parser.add_argument("--lease-directory", type=Path, required=True)
    raw = parser.parse_args(argv); g002 = G002Args(raw.dataset_root, raw.teacher_small, raw.imagenette_root, raw.run_id, raw.lease_directory)
    result = run_training(TrainingArgs(raw.pilot_evidence, raw.run_id, raw.checkpoint_directory, raw.metrics_path, g002)); print(json.dumps(result, sort_keys=True)); return 0 if result["status"] == READY else 2
if __name__ == "__main__": raise SystemExit(main())
