"""Fail-closed, validation-only raw-score threshold calibration for G002."""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .evidence import immutable_json, new_evidence
from .g002_training import PILOT_SHA256
from .storage import Allocation, PreflightProof, READY, atomic_write, preflight

COMMAND = "g002-calibrate-validation-raw-threshold"
FORMULA = "mean + 3 * population_standard_deviation"
STATUS = "CALIBRATED_RAW_THRESHOLD_NO_DECISIONS"
VALIDATION_GOOD_COUNT = 19
CATEGORY = "sheet_metal"
_SHA256_LENGTH = 64


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _inside(path: Path, root: Path) -> Path:
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact-root containment required") from exc
    return path


def _load_json(path: Path, root: Path, what: str, *, canonical: bool = False) -> tuple[dict[str, Any], bytes]:
    path = _inside(path, root)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {what}") from exc
    if not isinstance(value, dict) or (canonical and raw != _canonical(value)):
        raise ValueError(f"invalid {what}")
    return value, raw


def _sha256(value: bytes | Path) -> str:
    return sha256(value if isinstance(value, bytes) else value.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_LENGTH and all(char in "0123456789abcdef" for char in value)


def _identity_run_id(path: Path, digest: str) -> str:
    prefix, suffix = "g002-training-identity-", f"-{digest}.json"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError("training identity filename/content hash mismatch")
    value = path.name[len(prefix):-len(suffix)]
    if not value or "/" in value or "\\" in value:
        raise ValueError("training identity run ID invalid")
    return value


def _validation_good_identity(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("validation/good/"):
        return False
    parts = value.split("/")
    forbidden = ("test", "ood", "private", "mask", "label", "target")
    return len(parts) > 2 and all(part and part not in {".", ".."} for part in parts) and not any(
        token in part.casefold() for part in parts[2:] for token in forbidden
    )


@dataclass(frozen=True)
class CalibrationInput:
    artifact_root: Path
    run_id: str
    raw_map_manifest: Path
    training_identity: Path
    checkpoint: Path
    sidecar: Path
    metrics: Path
    final_attempt: Path
    dataset_root: Path
    geometry_evidence: Path
    geometry_evidence_sha256: str
    geometry_decision_id: str


@dataclass(frozen=True)
class _MapRecord:
    image_identity: str
    path: Path
    expected_bytes: int
    digest: str


def _admit_lineage(args: CalibrationInput, checkpoint: Mapping[str, Any]) -> None:
    root = Path(args.artifact_root).resolve()
    actual_checkpoint = _inside(args.checkpoint, root)
    sidecar_path = _inside(args.sidecar, root)
    metrics_path = _inside(args.metrics, root)
    final_path = _inside(args.final_attempt, root)
    required = {"checkpoint_sha256", "sidecar_sha256", "metrics_sha256", "final_attempt_sha256", "identity_sha256", "pilot_sha256"}
    if set(checkpoint) != required or checkpoint["pilot_sha256"] != PILOT_SHA256 or not all(_is_sha256(checkpoint[key]) for key in required):
        raise ValueError("raw-map checkpoint lineage invalid")
    if sidecar_path != actual_checkpoint.with_suffix(actual_checkpoint.suffix + ".json"):
        raise ValueError("sidecar/checkpoint binding mismatch")
    if actual_checkpoint.name not in {f"g002-last-{args.run_id}-0.ckpt", f"g002-last-{args.run_id}-1.ckpt"}:
        raise ValueError("checkpoint filename/run ID mismatch")
    if metrics_path.name != f"g002-metrics-{args.run_id}.json":
        raise ValueError("metrics filename/run ID mismatch")
    if any(_sha256(path) != checkpoint[key] for path, key in ((actual_checkpoint, "checkpoint_sha256"), (sidecar_path, "sidecar_sha256"), (metrics_path, "metrics_sha256"), (final_path, "final_attempt_sha256"))):
        raise ValueError("checkpoint lineage artifact hash mismatch")
    sidecar, _ = _load_json(sidecar_path, root, "checkpoint sidecar")
    if (sidecar.get("checkpoint_name") != actual_checkpoint.name or sidecar.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
            or sidecar.get("identity_sha256") != checkpoint["identity_sha256"] or sidecar.get("pilot_sha256") != PILOT_SHA256
            or sidecar.get("global_step") != 70_000 or sidecar.get("lineage") != args.run_id):
        raise ValueError("checkpoint sidecar linkage invalid")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid metrics") from exc
    if not isinstance(metrics, list):
        raise ValueError("invalid metrics")
    final, raw_final = _load_json(final_path, root, "final attempt")
    if final_path.name != f"g002-attempt-{args.run_id}-{_sha256(raw_final)}.json":
        raise ValueError("final-attempt filename/content binding mismatch")
    expected = {"checkpoint": checkpoint["checkpoint_sha256"], "sidecar": checkpoint["sidecar_sha256"], "metrics": checkpoint["metrics_sha256"]}
    if final.get("run_id") != args.run_id or final.get("status") != READY or final.get("lease_outcome") != "normal" or final.get("artifacts") != expected:
        raise ValueError("final-attempt lineage invalid")


def _admit_input(args: CalibrationInput) -> tuple[dict[str, Any], list[_MapRecord], dict[str, Any]]:
    root = Path(args.artifact_root).resolve()
    manifest_path = _inside(args.raw_map_manifest, root)
    manifest, _ = _load_json(manifest_path, root, "raw-map manifest", canonical=True)
    if manifest_path.name != f"g002-validation-raw-maps-{args.run_id}.json":
        raise ValueError("raw-map manifest run ID binding mismatch")
    required = {"status", "run_id", "transform_identity", "checkpoint", "maps"}
    if set(manifest) != required or manifest["status"] != "RAW_MAPS_ONLY" or manifest["run_id"] != args.run_id:
        raise ValueError("raw-map manifest schema/run ID invalid")
    if manifest["transform_identity"] != {"normalize": False, "resize": 256, "interpolation": "bilinear"}:
        raise ValueError("raw-map transform identity is not pinned")
    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, dict):
        raise ValueError("raw-map checkpoint lineage invalid")
    _admit_lineage(args, checkpoint)
    identity_path = _inside(args.training_identity, root)
    identity, raw_identity = _load_json(identity_path, root, "training identity", canonical=True)
    identity_hash = _sha256(raw_identity)
    if identity_hash != checkpoint["identity_sha256"] or _identity_run_id(identity_path, identity_hash) != args.run_id:
        raise ValueError("training identity lineage mismatch")
    validation = identity.get("data", {}).get("validation") if isinstance(identity.get("data"), Mapping) else None
    if not isinstance(validation, list) or len(validation) != VALIDATION_GOOD_COUNT:
        raise ValueError("training identity validation set invalid")
    allowed = {row.get("path"): row.get("sha256") for row in validation if isinstance(row, Mapping) and set(row) == {"path", "sha256"}}
    if len(allowed) != VALIDATION_GOOD_COUNT or any(not _validation_good_identity(key) or not _is_sha256(value) for key, value in allowed.items()):
        raise ValueError("training identity permits only 19 validation/good identities")
    dataset = Path(args.dataset_root).resolve()
    source_leaf = (dataset / CATEGORY).resolve()
    records: list[_MapRecord] = []
    expected_row = {"image_identity", "source_sha256", "map_sha256", "dtype", "shape", "byte_order", "checkpoint_sha256"}
    maps = manifest["maps"]
    if not isinstance(maps, list) or len(maps) != VALIDATION_GOOD_COUNT:
        raise ValueError("exactly 19 validation raw maps are required")
    seen: set[str] = set()
    for index, row in enumerate(maps):
        if not isinstance(row, dict) or set(row) != expected_row or row["image_identity"] in seen:
            raise ValueError("raw-map record schema/identity invalid")
        identity_name, map_hash, source_hash = row["image_identity"], row["map_sha256"], row["source_sha256"]
        shape = row["shape"]
        if (identity_name not in allowed or source_hash != allowed[identity_name] or not _is_sha256(map_hash)
                or row["checkpoint_sha256"] != checkpoint["checkpoint_sha256"] or row["dtype"] != "<f4" or row["byte_order"] != "<"
                or not isinstance(shape, list) or not shape or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape)):
            raise ValueError("raw-map provenance/dtype/shape invalid")
        source = (source_leaf / identity_name).resolve()
        try:
            source.relative_to(source_leaf)
        except ValueError as exc:
            raise ValueError("validation source escapes dataset root") from exc
        if not source.is_file() or _sha256(source) != source_hash:
            raise ValueError("validation source identity hash mismatch")
        path = _inside(root / f"g002-validation-raw-{index:02d}-{map_hash}.bin", root)
        expected_bytes = math.prod(shape) * 4
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError("raw-map filename/size mismatch")
        seen.add(identity_name)
        records.append(_MapRecord(identity_name, path, expected_bytes, map_hash))
    if seen != set(allowed):
        raise ValueError("raw-map identities do not match training identity")
    geometry_path = _inside(args.geometry_evidence, root)
    geometry, raw_geometry = _load_json(geometry_path, root, "frozen geometry evidence", canonical=True)
    if (not _is_sha256(args.geometry_evidence_sha256) or _sha256(raw_geometry) != args.geometry_evidence_sha256
            or geometry.get("decision_id") != args.geometry_decision_id or geometry.get("status") != "FROZEN"):
        raise ValueError("pre-frozen geometry evidence binding invalid")
    return manifest, records, geometry


def _stats(records: Iterable[_MapRecord], *, chunk_values: int = 65_536) -> tuple[int, float, float, list[dict[str, Any]]]:
    """Stream hash, byte count, and Welford statistics together; no map is retained."""
    if chunk_values < 1:
        raise ValueError("chunk_values must be positive")
    count, mean, m2 = 0, 0.0, 0.0
    maxima: list[dict[str, Any]] = []
    for record in records:
        maximum, byte_count, digest = -math.inf, 0, sha256()
        with record.path.open("rb") as stream:
            while raw := stream.read(chunk_values * 4):
                byte_count += len(raw)
                digest.update(raw)
                if len(raw) % 4:
                    raise ValueError("raw-map byte alignment invalid")
                for (value,) in struct.iter_unpack("<f", raw):
                    if not math.isfinite(value):
                        raise ValueError("raw-map contains non-finite score")
                    count += 1
                    delta = value - mean
                    mean += delta / count
                    m2 += delta * (value - mean)
                    maximum = max(maximum, value)
        if maximum == -math.inf or byte_count != record.expected_bytes or digest.hexdigest() != record.digest:
            raise ValueError("raw-map changed during calibration")
        maxima.append({"image_identity": record.image_identity, "map_sha256": record.digest, "max_raw_score": maximum})
    if not count:
        raise ValueError("raw-map collection cannot be empty")
    return count, mean, math.sqrt(m2 / count), maxima


def _relative(path: Path, root: Path) -> str:
    return _inside(path, root).relative_to(root).as_posix()


def calibrate(args: CalibrationInput, *, admit: Callable[..., PreflightProof] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write) -> dict[str, Any]:
    """Persist one immutable calibration artifact without enabling any decision path."""
    root = Path(args.artifact_root).resolve()
    manifest, records, geometry = _admit_input(args)
    count, mean, population_std, maxima = _stats(records)
    record = new_evidence(args.run_id, COMMAND, STATUS, [
        "official comparator provenance is unavailable; comparator and all verdict/F1 paths remain blocked",
        "TESTpub/TESTpriv/OOD inputs are forbidden from calibration and audit remains blocked",
    ])
    record.update({
        "raw_threshold": mean + 3.0 * population_std,
        "formula": FORMULA,
        "comparator": None,
        "mean": mean,
        "population_standard_deviation": population_std,
        "pixel_count": count,
        "per_image_max_raw_scores": maxima,
        "raw_map_manifest": {"path": _relative(args.raw_map_manifest, root), "sha256": _sha256(_canonical(manifest))},
        "checkpoint": manifest["checkpoint"],
        "geometry": {"path": _relative(args.geometry_evidence, root), "sha256": args.geometry_evidence_sha256, "decision_id": args.geometry_decision_id, "status": geometry["status"]},
        "blocked": {"comparator": "BLOCKED_MISSING_VERIFIED_PROTOCOL_PROVENANCE", "pixel_verdict": "BLOCKED", "image_verdict": "BLOCKED", "f1": "BLOCKED", "testpub_audit": "BLOCKED"},
    })
    payload, digest = immutable_json(record)
    source = f"exact immutable calibration bytes={len(payload)} sha256={digest}; maps streamed in 65536-float chunks"
    allocations = [
        Allocation("artifact", len(payload), "persistent", source, "g002-calibration"),
        Allocation("artifact", len(payload), "transient", source, "g002-calibration-incoming"),
    ]
    proof = admit(run_id=args.run_id, allocations=allocations, reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root:
        raise ValueError("fresh proof artifact root changed")
    destination = root / f"g002-calibration-{args.run_id}-{digest}.json"
    result = writer(destination, payload, proof=proof, run_id=args.run_id, overwrite=False)
    if result.get("status") != READY or not destination.is_file() or destination.read_bytes() != payload:
        raise ValueError("immutable calibration artifact write failed")
    return {**record, "artifact": str(destination), "artifact_sha256": digest}
