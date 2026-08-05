"""G002 validation raw-map collection and lossless artifact persistence.

No threshold, comparator, metric, or verdict path exists here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .g002_training import PILOT_SHA256
from .storage import Allocation, PreflightProof, READY, atomic_write, preflight

VALIDATION_GOOD_COUNT = 19
CATEGORY = "sheet_metal"


@dataclass(frozen=True)
class AdmittedCheckpoint:
    path: Path; checkpoint_sha256: str; sidecar_sha256: str; metrics_sha256: str
    final_attempt_sha256: str; identity_sha256: str; pilot_sha256: str
    run_id: str; dataset_root: Path; validation_identities: tuple[tuple[str, str], ...]


def _hash(path: Path) -> str: return sha256(path.read_bytes()).hexdigest()
def _canonical(value: Mapping[str, Any]) -> bytes: return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _inside(path: Path, root: Path) -> Path:
    path, root = Path(path).resolve(), Path(root).resolve()
    try: path.relative_to(root)
    except ValueError as exc: raise ValueError("artifact-root containment required") from exc
    return path


def _load_json(path: Path, what: str) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ValueError(f"invalid {what}") from exc
    if not isinstance(value, dict): raise ValueError(f"invalid {what}")
    return value


def _validation_identities(identity: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    try: rows = identity["data"]["validation"]
    except (KeyError, TypeError) as exc: raise ValueError("training identity has no validation file identity") from exc
    if not isinstance(rows, list): raise ValueError("invalid validation file identity")
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ValueError("invalid validation file identity")
        path, digest = row["path"], row["sha256"]
        parts = Path(path).parts if isinstance(path, str) else ()
        if (not isinstance(path, str) or Path(path).is_absolute() or parts[:2] != ("validation", "good")
                or any(part in {"", ".", ".."} for part in parts) or any(segment.casefold() in {"test", "ood", "private"} for segment in parts)
                or not isinstance(digest, str) or len(parts) != 3):
            raise ValueError("validation identity must be canonical validation/good only")
        result.append((path, digest))
    if len(result) != VALIDATION_GOOD_COUNT or len({path for path, _ in result}) != VALIDATION_GOOD_COUNT:
        raise ValueError("training identity must contain exactly 19 unique validation/good images")
    return tuple(sorted(result))


def admit_completed_checkpoint(checkpoint: Path, artifact_root: Path, training_identity: Mapping[str, Any],
                               dataset_root: Path, final_attempt: Path, metrics: Path | None = None) -> AdmittedCheckpoint:
    """Admit only one complete, identity-bound 70k training result."""
    root = Path(artifact_root).resolve(); checkpoint = _inside(checkpoint, root); final_attempt = _inside(final_attempt, root)
    sidecar_path = _inside(checkpoint.with_suffix(checkpoint.suffix + ".json"), root)
    if metrics is None:
        candidates = sorted(root.glob("g002-metrics-*.json"))
        if len(candidates) != 1: raise ValueError("exactly one artifact-root metrics file required")
        metrics = candidates[0]
    metrics = _inside(metrics, root)
    sidecar, attempt = _load_json(sidecar_path, "checkpoint sidecar"), _load_json(final_attempt, "final attempt")
    checkpoint_hash, sidecar_hash, metrics_hash, final_hash = map(_hash, (checkpoint, sidecar_path, metrics, final_attempt))
    identity_hash = sha256(_canonical(training_identity)).hexdigest(); identities = _validation_identities(training_identity)
    run_id = sidecar.get("lineage")
    if not isinstance(run_id, str) or not run_id or checkpoint.name not in {f"g002-last-{run_id}-0.ckpt", f"g002-last-{run_id}-1.ckpt"}:
        raise ValueError("checkpoint lineage/run ID mismatch")
    if metrics.name != f"g002-metrics-{run_id}.json" or final_attempt.name != f"g002-attempt-{run_id}-{final_hash}.json":
        raise ValueError("final attempt/metrics filename binding mismatch")
    expected = {"checkpoint": checkpoint_hash, "sidecar": sidecar_hash, "metrics": metrics_hash}
    if (sidecar.get("checkpoint_name") != checkpoint.name or sidecar.get("global_step") != 70_000
            or sidecar.get("checkpoint_sha256") != checkpoint_hash or sidecar.get("identity_sha256") != identity_hash
            or sidecar.get("pilot_sha256") != PILOT_SHA256 or attempt.get("run_id") != run_id
            or attempt.get("status") != READY or attempt.get("lease_outcome") != "normal" or attempt.get("artifacts") != expected):
        raise ValueError("completed checkpoint admission failed")
    dataset_root = Path(dataset_root).resolve()
    if not (dataset_root / CATEGORY / "validation" / "good").is_dir(): raise ValueError("trusted dataset validation root required")
    return AdmittedCheckpoint(checkpoint, checkpoint_hash, sidecar_hash, metrics_hash, final_hash, identity_hash, PILOT_SHA256,
                              run_id, dataset_root, identities)


def _array(value: Any):
    try:
        import numpy as np
        if hasattr(value, "detach"): value = value.detach().cpu().contiguous().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not array.size: raise ValueError
        array = np.ascontiguousarray(array.astype("<f4", copy=False))
        if not bool(np.isfinite(array).all()): raise ValueError
        return array
    except Exception as exc: raise ValueError("raw map must be nonempty finite numeric data") from exc


def raw_map(st: Any, stae: Any):
    """Canonical little-endian float32 0.5 * (ST + STAE), without normalization."""
    st, stae = _array(st), _array(stae)
    if st.shape != stae.shape: raise ValueError("map component shape mismatch")
    return _array(0.5 * (st + stae))


def _source_path(batch: Mapping[str, Any], admitted: AdmittedCheckpoint, identity: str) -> Path:
    if set(batch) - {"image", "image_identity", "source_path", "image_path"}: raise ValueError("label/mask/test/OOD metadata forbidden")
    supplied = next((batch[key] for key in ("source_path", "image_path") if key in batch), None)
    if supplied is None: raise ValueError("source path required for source-content hash verification")
    expected = (admitted.dataset_root / CATEGORY / identity).resolve()
    if Path(supplied).resolve() != expected: raise ValueError("source path is outside the admitted validation/good leaf")
    return expected


def collect_validation_maps(model: Any, loader: Iterable[Mapping[str, Any]], admitted: AdmittedCheckpoint) -> dict[str, Any]:
    """Collect precisely the validation identities embedded in the admitted training identity."""
    expected = dict(admitted.validation_identities); rows: list[dict[str, Any]] = []; seen: set[str] = set()
    for batch in loader:
        key = batch.get("image_identity")
        if not isinstance(key, str) or key not in expected or key in seen: raise ValueError("validation identity is unapproved or duplicate")
        source = _source_path(batch, admitted, key)
        if _hash(source) != expected[key]: raise ValueError("source content hash mismatch")
        st, stae = model.get_maps(batch["image"], normalize=False); value = raw_map(st, stae); raw = value.tobytes(order="C")
        rows.append({"image_identity": key, "source_sha256": expected[key], "map_sha256": sha256(raw).hexdigest(),
                     "dtype": "<f4", "shape": list(value.shape), "byte_order": "<", "checkpoint_sha256": admitted.checkpoint_sha256, "_bytes": raw})
        seen.add(key)
    if seen != set(expected): raise ValueError("validation identity set mismatch")
    return {"status": "RAW_MAPS_ONLY", "maps": rows, "checkpoint": {key: getattr(admitted, key) for key in
            ("checkpoint_sha256", "sidecar_sha256", "metrics_sha256", "final_attempt_sha256", "identity_sha256", "pilot_sha256")}}


def _transform_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"normalize", "resize", "crop", "interpolation"}
    if not isinstance(value, Mapping) or not value or set(value) - allowed or value.get("normalize") is not False:
        raise ValueError("invalid transform identity")
    for key in ("resize", "crop"):
        if key in value and (not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0):
            raise ValueError("invalid transform identity")
    if "interpolation" in value and value["interpolation"] not in {"nearest", "bilinear", "bicubic"}:
        raise ValueError("invalid transform identity")
    encoded = _canonical(value).decode().lower()
    if "/" in encoded or "\\" in encoded or "test" in encoded or "ood" in encoded: raise ValueError("transform identity contains forbidden path/test data")
    return dict(value)


def persist_validation_maps(collected: Mapping[str, Any], artifact_root: Path, run_id: str, transform_identity: Mapping[str, Any], *,
                            admit: Callable[..., PreflightProof] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write) -> dict[str, Any]:
    """Atomically persist canonical maps; one proof covers their exact aggregate bytes."""
    maps = list(collected.get("maps", ())); transform = _transform_identity(transform_identity)
    if collected.get("status") != "RAW_MAPS_ONLY" or len(maps) != VALIDATION_GOOD_COUNT: raise ValueError("only a complete raw-map collection can be persisted")
    root = Path(artifact_root).resolve(); manifest_maps = []
    for row in maps:
        raw = row.get("_bytes")
        if not isinstance(raw, bytes) or sha256(raw).hexdigest() != row.get("map_sha256"): raise ValueError("raw-map bytes/hash mismatch")
        manifest_maps.append({key: value for key, value in row.items() if key != "_bytes"})
    payload = json.dumps({"status": "RAW_MAPS_ONLY", "run_id": run_id, "transform_identity": transform,
                          "checkpoint": dict(collected.get("checkpoint", {})), "maps": manifest_maps}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    raw_total, pending = sum(len(row["_bytes"]) for row in maps), max([len(payload), *(len(row["_bytes"]) for row in maps)])
    source = f"exact canonical map bytes={raw_total}; manifest bytes={len(payload)}; pending atomic bytes={pending}"
    allocations = [Allocation("artifact", raw_total + len(payload), "persistent", source, "g002-validation-raw-maps"),
                   Allocation("artifact", pending, "transient", source, "g002-validation-raw-maps-incoming")]
    proof = admit(run_id=run_id, allocations=allocations, reserve_bytes=pending, reserve_evidence={"max_pending_atomic_write_bytes": pending, "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ValueError("fresh proof artifact root changed")
    written = []
    for index, row in enumerate(maps):
        destination = root / f"g002-validation-raw-{index:02d}-{row['map_sha256']}.bin"; result = writer(destination, row["_bytes"], proof=proof, run_id=run_id, overwrite=False)
        if result.get("status") != READY or not destination.is_file() or _hash(destination) != row["map_sha256"]: raise ValueError("raw-map artifact write failed")
        written.append(destination)
    manifest = root / f"g002-validation-raw-maps-{run_id}.json"; result = writer(manifest, payload, proof=proof, run_id=run_id, overwrite=False)
    if result.get("status") != READY or not manifest.is_file() or manifest.read_bytes() != payload: raise ValueError("raw-map manifest write failed")
    return {"status": "RAW_MAPS_ONLY", "manifest": str(manifest), "map_paths": [str(path) for path in written]}
