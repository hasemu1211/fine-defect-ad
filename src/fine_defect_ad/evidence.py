"""Small schema validators for immutable R0 evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


DER_REQUIRED = frozenset({
    "decision_id", "scope", "decision", "status", "drivers", "alternatives",
    "evidence_type", "primary_source", "reference_locator", "measurement_artifact",
    "derivation", "limitations", "reviewed_at", "artifact_or_model_version",
})
EVIDENCE_REQUIRED = frozenset({"run_id", "timestamp", "command", "status", "limitations"})
FAILURE_STATUSES = frozenset({"STORAGE_BLOCKED", "STOPPED_INCOMPLETE", "INVALIDATED"})


def validate_der(item: dict[str, Any]) -> None:
    missing = DER_REQUIRED - item.keys()
    if missing or not str(item.get("decision_id", "")).startswith("DEC-"):
        raise ValueError(f"invalid DER: missing={sorted(missing)}")
    if item["status"] not in {"proposed", "accepted", "superseded", "blocked"}:
        raise ValueError("invalid DER status")

def validate_decision_register(path: Path) -> list[dict[str, Any]]:
    """The .yaml register deliberately uses JSON, a YAML subset, avoiding a parser dependency."""
    try: register = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc: raise ValueError('DER must be JSON-compatible YAML') from exc
    if not isinstance(register, list) or not register: raise ValueError('DER register must be a nonempty array')
    for item in register:
        if not isinstance(item, dict): raise ValueError('DER item must be an object')
        validate_der(item)
    return register


def validate_evidence(record: dict[str, Any]) -> None:
    missing = EVIDENCE_REQUIRED - record.keys()
    if missing:
        raise ValueError(f"missing evidence fields: {sorted(missing)}")
    if record["status"] in FAILURE_STATUSES and not record["limitations"]:
        raise ValueError("failure evidence requires a cause/limitation")
    if record.get("lock_mode") and record["lock_mode"] != "fcntl.flock":
        raise ValueError("unsupported lock provenance")
    if record.get("storage_proof") and not {"run_id", "fingerprint", "filesystems"} <= record["storage_proof"].keys():
        raise ValueError("incomplete storage provenance")


def immutable_json(record: dict[str, Any]) -> tuple[bytes, str]:
    validate_evidence(record)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return encoded, sha256(encoded).hexdigest()


def new_evidence(run_id: str, command: str, status: str, limitations: list[str]) -> dict[str, Any]:
    return {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "command": command,
            "status": status, "limitations": limitations}
