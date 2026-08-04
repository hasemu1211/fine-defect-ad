"""Dependency-free evidence collection for the R1 1,000-step pilot.

A training runner supplies callbacks; importing this module never imports torch,
Lightning, or anomalib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import platform
import resource
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping

from .gpu_lock import BusyError, GpuLease

PILOT_STEPS = 1_000
MAX_STEPS = 70_000
READY = "READY"
STOPPED_INCOMPLETE = "STOPPED_INCOMPLETE"
REQUIRED_PROTOCOL_FIELDS = ("anomalib_version", "anomalib_commit", "config_hash", "seed_provenance_hash", "precision")
REQUIRED_PEAK_FIELDS = ("peak_host_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")


class MeasurementError(ValueError):
    """A callback provided no usable measured value."""


def pilot_step_budget(train_loader: object) -> int:
    """Return the R1 budget, based on the measured loader length."""
    try:
        batches = len(train_loader)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("train_loader must have a measured length") from exc
    if not isinstance(batches, int) or batches <= 0:
        raise ValueError("train_loader length must be a positive integer")
    return min(MAX_STEPS, PILOT_STEPS * batches)


def median_step_seconds(timestamps: list[float]) -> float:
    """Calculate step time from actual completion timestamps, never a guess."""
    if len(timestamps) < 2:
        raise ValueError("at least two measured step timestamps are required")
    if any(not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) for timestamp in timestamps):
        raise ValueError("step timestamps must be finite numeric")
    intervals = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    if any(interval < 0 for interval in intervals):
        raise MeasurementError("step timestamps must be nondecreasing")
    return statistics.median(intervals)


def estimate_eta_seconds(*, total_steps: int, step_timestamps: list[float],
                         setup_overhead_seconds: float | None,
                         validation_overhead_seconds: float | None) -> float:
    """Estimate total run time only from observed step and overhead measurements."""
    if not isinstance(total_steps, int) or total_steps < 0:
        raise ValueError("total_steps must be a nonnegative integer")
    if setup_overhead_seconds is None or validation_overhead_seconds is None:
        raise ValueError("setup and validation overhead must be explicitly measured")
    if (not math.isfinite(setup_overhead_seconds) or not math.isfinite(validation_overhead_seconds)
            or setup_overhead_seconds < 0 or validation_overhead_seconds < 0):
        raise ValueError("measured overhead must be finite and nonnegative")
    return total_steps * median_step_seconds(step_timestamps) + setup_overhead_seconds + validation_overhead_seconds


def host_rss_bytes() -> int:
    """Return the process high-water RSS in bytes using the native runtime metric."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if platform.system() == "Darwin" else rss * 1024)


def _json_value(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0):
        raise MeasurementError(f"{name} must be finite nonnegative numeric or None")
    return float(value)


@dataclass
class PilotEvidence:
    """A small recorder suitable for callbacks from a future training runner."""
    run_id: str
    command: str
    planned_steps: int
    protocol_metadata: Mapping[str, str] | None = None
    setup_overhead_seconds: float | None = None
    validation_overhead_seconds: float | None = None
    step_timestamps: list[float] = field(default_factory=list)
    peak_host_rss_bytes: float | None = None
    peak_gpu_allocated_bytes: float | None = None
    peak_gpu_reserved_bytes: float | None = None
    gradient_finite: bool | None = None
    termination_cause: str | None = None
    lease_events: list[dict[str, Any]] = field(default_factory=list)

    def record_setup(self, seconds: float) -> None:
        self.setup_overhead_seconds = _json_value(seconds, "setup_overhead_seconds")

    def record_validation(self, seconds: float) -> None:
        self.validation_overhead_seconds = _json_value(seconds, "validation_overhead_seconds")

    def record_step(self, *, timestamp: float, gradients_finite: bool,
                    host_rss_bytes: float | None = None,
                    gpu_allocated_bytes: float | None = None,
                    gpu_reserved_bytes: float | None = None) -> None:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise MeasurementError("timestamp must be finite numeric")
        if self.step_timestamps and timestamp < self.step_timestamps[-1]:
            raise MeasurementError("step timestamps must be nondecreasing")
        if not isinstance(gradients_finite, bool):
            raise MeasurementError("gradients_finite must be measured as a bool")
        self.step_timestamps.append(float(timestamp))
        self.gradient_finite = gradients_finite if self.gradient_finite is None else self.gradient_finite and gradients_finite
        for field_name, value in (("peak_host_rss_bytes", host_rss_bytes),
                                  ("peak_gpu_allocated_bytes", gpu_allocated_bytes),
                                  ("peak_gpu_reserved_bytes", gpu_reserved_bytes)):
            measured = _json_value(value, field_name)
            old = getattr(self, field_name)
            if measured is not None:
                setattr(self, field_name, measured if old is None else max(old, measured))

    def record_lease_events(self, events: list[dict[str, Any]]) -> None:
        self.lease_events = events

    def to_record(self, termination_cause: str | None = None) -> dict[str, Any]:
        cause = termination_cause or self.termination_cause
        completed = len(self.step_timestamps)
        missing_protocol = [key for key in REQUIRED_PROTOCOL_FIELDS
                            if not isinstance((self.protocol_metadata or {}).get(key), str)
                            or not (self.protocol_metadata or {})[key]]
        missing_peaks = [key for key in REQUIRED_PEAK_FIELDS if getattr(self, key) is None]
        if self.gradient_finite is False:
            cause = cause or "GRADIENT_NONFINITE"
        elif completed != PILOT_STEPS:
            cause = cause or f"PILOT_STEPS_{completed}_OF_{PILOT_STEPS}"
        elif missing_protocol:
            cause = cause or "PROTOCOL_METADATA_MISSING:" + ",".join(missing_protocol)
        elif missing_peaks:
            cause = cause or "REQUIRED_MEASUREMENTS_MISSING:" + ",".join(missing_peaks)
        elif self.setup_overhead_seconds is None or self.validation_overhead_seconds is None:
            cause = cause or "PILOT_OVERHEAD_UNMEASURED"
        status = READY if cause is None else STOPPED_INCOMPLETE
        eta = None
        if self.setup_overhead_seconds is not None and self.validation_overhead_seconds is not None and completed >= 2:
            eta = estimate_eta_seconds(total_steps=self.planned_steps, step_timestamps=self.step_timestamps,
                                       setup_overhead_seconds=self.setup_overhead_seconds,
                                       validation_overhead_seconds=self.validation_overhead_seconds)
        record = {
            "run_id": self.run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "command": self.command,
            "status": status, "limitations": [] if status == READY else [cause],
            "planned_steps": self.planned_steps, "pilot_target_steps": PILOT_STEPS, "completed_steps": completed,
            "termination_cause": cause, "step_timestamps": self.step_timestamps,
            "median_seconds_per_step": median_step_seconds(self.step_timestamps) if completed >= 2 else None,
            "eta_seconds": eta, "setup_overhead_seconds": self.setup_overhead_seconds,
            "validation_overhead_seconds": self.validation_overhead_seconds,
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
            "peak_gpu_allocated_bytes": self.peak_gpu_allocated_bytes,
            "peak_gpu_reserved_bytes": self.peak_gpu_reserved_bytes,
            "gradient_finite": self.gradient_finite, "protocol_metadata": dict(self.protocol_metadata or {}),
            **{key: (self.protocol_metadata or {}).get(key) for key in REQUIRED_PROTOCOL_FIELDS},
            "lease_events": self.lease_events,
        }
        json.dumps(record, allow_nan=False)
        return record


def lease_events(directory: Path, run_id: str) -> list[dict[str, Any]]:
    """Read this run's immutable events after the lease exits."""
    events: list[dict[str, Any]] = []
    for path in sorted((Path(directory) / "gpu-heavy-events").glob("*.json")):
        try:
            event = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if event.get("run_id") == run_id:
            events.append(event)
    return events


def run_pilot(*, lease_directory: Path, run_id: str, command: str, train_loader: object,
              setup: Callable[[], None], step: Callable[[], Mapping[str, Any]],
              validate: Callable[[], None], protocol_metadata: Mapping[str, str] | None = None,
              clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Execute exactly the future runner boundary under one GPU lease.

    `step` returns measured ``gradients_finite`` and optional RSS/GPU byte values.
    """
    evidence = PilotEvidence(run_id, command, pilot_step_budget(train_loader), protocol_metadata)
    cause: str | None = None
    try:
        with GpuLease(lease_directory, run_id, command):
            started = clock(); setup(); evidence.record_setup(clock() - started)
            for _ in range(PILOT_STEPS):
                measured = step()
                evidence.record_step(timestamp=clock(), gradients_finite=measured["gradients_finite"],
                                     host_rss_bytes=measured.get("host_rss_bytes", host_rss_bytes()),
                                     gpu_allocated_bytes=measured.get("gpu_allocated_bytes"),
                                     gpu_reserved_bytes=measured.get("gpu_reserved_bytes"))
                if evidence.gradient_finite is False:
                    cause = "GRADIENT_NONFINITE"
                    break
            if cause is None:
                started = clock(); validate(); evidence.record_validation(clock() - started)
    except BusyError:
        raise
    except MeasurementError as exc:
        cause = cause or f"MEASUREMENT_INVALID:{exc}"
    except Exception as exc:
        cause = cause or f"RUNNER_EXCEPTION:{type(exc).__name__}"
    evidence.record_lease_events(lease_events(lease_directory, run_id))
    return evidence.to_record(cause)
