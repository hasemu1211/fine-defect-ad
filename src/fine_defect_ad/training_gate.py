"""Pure admission and time projection for the R1 full-training run.

This is deliberately only a gate: a future runner must supply the actual
training, validation, and checkpoint wiring after this admission succeeds.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

from .pilot import (MAX_STEPS, PILOT_STEPS, READY, REQUIRED_PEAK_FIELDS,
                    expected_pilot_protocol_metadata, pilot_step_budget)
from .r1 import R1_SEED, R1_SEED_IDENTITY, R1_SEED_IDENTITY_SHA256

R1_SPLIT_PURPOSE = MappingProxyType({
    "train": "fitting_only",
    "validation": "calibration_and_checkpoint_only",
    "test": "no_access",
})


class TrainingAdmissionError(ValueError):
    """The measured pilot is not sufficient to start full training."""


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrainingAdmissionError(f"pilot {name} must be finite numeric")
    if value < 0 or (positive and value <= 0):
        raise TrainingAdmissionError(f"pilot {name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _require_ready_pilot(pilot: Mapping[str, Any]) -> tuple[float, float, float]:
    if pilot.get("status") != READY:
        raise TrainingAdmissionError("full training requires a READY pilot")
    cause = pilot.get("termination_cause")
    if cause is not None:
        raise TrainingAdmissionError(f"pilot terminated: {cause}")
    if pilot.get("gradient_finite") is not True:
        raise TrainingAdmissionError("full training requires finite pilot gradients")
    if pilot.get("pilot_target_steps") != PILOT_STEPS or pilot.get("completed_steps") != PILOT_STEPS:
        raise TrainingAdmissionError("full training requires exactly the complete pilot")
    if dict(pilot.get("protocol_metadata") or {}) != expected_pilot_protocol_metadata():
        raise TrainingAdmissionError("pilot protocol provenance does not match fixed R1")
    for field in REQUIRED_PEAK_FIELDS:
        _finite(field, pilot.get(field))
    return (
        _finite("median_seconds_per_step", pilot.get("median_seconds_per_step"), positive=True),
        _finite("setup_overhead_seconds", pilot.get("setup_overhead_seconds")),
        _finite("validation_overhead_seconds", pilot.get("validation_overhead_seconds")),
    )


def admit_full_training(*, train_loader: object, pilot: Mapping[str, Any],
                        split_purpose: Mapping[str, str]) -> dict[str, Any]:
    """Admit a fixed R1 run only from complete measured pilot evidence.

    ``split_purpose`` is a declarative boundary, not runner callbacks: the
    test split cannot be supplied to fitting, calibration, or checkpointing.
    """
    if dict(split_purpose) != dict(R1_SPLIT_PURPOSE):
        raise TrainingAdmissionError("R1 requires train fitting, validation-only calibration/checkpoint, and no test access")
    median, setup, validation = _require_ready_pilot(pilot)
    steps = pilot_step_budget(train_loader)
    return {
        "status": READY,
        "max_steps": steps,
        "eta_seconds": steps * median + setup + validation,
        "median_seconds_per_step": median,
        "setup_overhead_seconds": setup,
        "validation_overhead_seconds": validation,
        "seed": R1_SEED,
        "seed_identity": R1_SEED_IDENTITY,
        "seed_identity_sha256": R1_SEED_IDENTITY_SHA256,
        "protocol_metadata": expected_pilot_protocol_metadata(),
        "split_purpose": dict(R1_SPLIT_PURPOSE),
        "limitations": ["admission/projection only; no training, validation, checkpoint, or test callbacks are executed"],
    }
