"""Validation-only E2 split-map calibration; never reads TEST data or persists maps."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .g002_e2_runtime import SPLIT_TARGET_SHAPE, combine_split_maps, verify_split_freeze
from .storage import Allocation, READY, atomic_write, preflight

COMMAND = "g002-e2-split-calibration"
FORMULA = "mean + 3 * population_standard_deviation"
COMPARATOR = ">"
VALIDATION_COUNT = 19


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, root: Path) -> Path:
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact-root containment required") from exc
    return path


def _valid_identity(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("validation/good/") and "test" not in value.casefold()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _archives() -> dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "evidence" / "mvtec-metric-provenance.json"
    try:
        data = json.loads(path.read_text())['sources']
        return {"mvtec_ad2_public_code_utils_sha256": data['mvtec_ad2_public_code_utils']['archive_sha256'],
                "mvtec_ad_evaluator_v1_sha256": data['mvtec_ad_evaluator_v1']['archive_sha256']}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("MVTec archive provenance unavailable") from exc


@dataclass(frozen=True)
class SplitCalibrationInput:
    artifact_root: Path
    split_freeze: Path
    run_id: str


def _load_freeze(args: SplitCalibrationInput) -> tuple[Path, dict[str, Any], str]:
    root = Path(args.artifact_root).resolve()
    path = _inside(args.split_freeze, root)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid split freeze") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError("split freeze must be canonical JSON")
    verify_split_freeze(value)
    if value.get("checkpoint_sha256") is None:
        raise ValueError("split freeze checkpoint binding invalid")
    return path, value, _hash(path)


def _admit_rows(freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    identities = freeze.get("validation_identities")
    rows = freeze.get("maps")
    if not isinstance(identities, list) or not isinstance(rows, list) or len(identities) != VALIDATION_COUNT or len(rows) != VALIDATION_COUNT:
        raise ValueError("exactly 19 validation maps required")
    allowed = {item.get("path"): item.get("sha256") for item in identities if isinstance(item, dict)}
    if len(allowed) != VALIDATION_COUNT or any(not _valid_identity(name) or not _sha(digest) for name, digest in allowed.items()):
        raise ValueError("TEST access or invalid validation identities")
    seen: set[str] = set()
    admitted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid split map record")
        identity = row.get("image_identity")
        if identity in seen or identity not in allowed or row.get("source_sha256") != allowed[identity]:
            raise ValueError("split map identity/hash binding invalid")
        for name in ("local_st_sha256", "global_stae_sha256"):
            if not _sha(row.get(name)):
                raise ValueError("split map hash invalid")
        for name in ("local_st_shape", "global_stae_shape"):
            shape = row.get(name)
            if not isinstance(shape, list) or not shape or any(type(x) is not int or x <= 0 for x in shape):
                raise ValueError("split map shape invalid")
        seen.add(identity); admitted.append(dict(row))
    if seen != set(allowed):
        raise ValueError("split maps do not match validation freeze")
    return admitted


def _artifact(root: Path, digest: str, run_id: str) -> tuple[Path, dict[str, Any]]:
    # Producers can retain a short alias and a run-qualified immutable filename.
    candidates = sorted(path for path in root.glob(f"*-{digest}.bin") if path.is_file())
    if not candidates or any(_hash(path) != digest for path in candidates):
        raise ValueError("missing or mismatched hash-bound split map artifact")
    preferred = f"g002-e2-split-validation-{run_id}-"
    path = next((item for item in candidates if item.name.startswith(preferred)), candidates[0])
    proof = {"artifact_duplicate_count": len(candidates)}
    # Never serialize arbitrary operator filenames; producer-owned basenames have no source identity.
    if all(item.name.startswith("g002-e2-split-validation-") for item in candidates):
        proof["artifact_basenames"] = [item.name for item in candidates]
    return path, proof


def _read_map(root: Path, digest: str, shape: list[int], run_id: str):
    import numpy as np
    path, proof = _artifact(root, digest, run_id)
    expected = math.prod(shape) * 4
    if path.stat().st_size != expected:
        raise ValueError("split map byte-size/shape mismatch")
    value = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape)
    if not bool(np.isfinite(value).all()):
        raise ValueError("split map must be finite")
    return value, proof


def _write(root: Path, run_id: str, payload: bytes, *, admit=preflight, writer=atomic_write) -> Path:
    digest = sha256(payload).hexdigest()
    target = root / f"g002-e2-split-calibration-{run_id}-{digest}.json"
    proof = admit(run_id=run_id, allocations=[Allocation("artifact", len(payload), "persistent", "canonical split calibration proof", target.name), Allocation("artifact", len(payload), "transient", "canonical split calibration proof", target.name + ".incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0, "runtime_or_source_citation": "exact canonical calibration proof bytes"})
    result = writer(target, payload, proof=proof, run_id=run_id, overwrite=False)
    if result.get("status") != READY or target.read_bytes() != payload:
        raise ValueError("immutable calibration write failed")
    return target


def calibrate(args: SplitCalibrationInput, *, torch: Any = None, admit=preflight, writer=atomic_write) -> dict[str, Any]:
    """Create a proof-bound threshold from only the frozen 19 validation maps."""
    import numpy as np
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id:
        raise ValueError("run_id invalid")
    root = Path(args.artifact_root).resolve()
    freeze_path, freeze, freeze_hash = _load_freeze(args)
    rows = _admit_rows(freeze)
    if torch is None:
        import torch as torch_module
        torch = torch_module
    count = 0
    mean = 0.0
    m2 = 0.0
    map_proofs = []
    for row in rows:
        local, local_artifact = _read_map(root, row["local_st_sha256"], row["local_st_shape"], args.run_id)
        global_map, global_artifact = _read_map(root, row["global_stae_sha256"], row["global_stae_shape"], args.run_id)
        combined = np.asarray(combine_split_maps(local, global_map, freeze["quantiles"], torch), dtype="<f4")
        if combined.shape != SPLIT_TARGET_SHAPE or not bool(np.isfinite(combined).all()):
            raise ValueError("combined split map must be finite 528x2112")
        values = combined.reshape(-1).astype(np.float64, copy=False)
        # Parallel Welford update: no combined map is retained or written.
        part_count = values.size; part_mean = float(values.mean()); part_m2 = float(((values - part_mean) ** 2).sum())
        delta = part_mean - mean
        mean += delta * part_count / (count + part_count)
        m2 += part_m2 + delta * delta * count * part_count / (count + part_count)
        count += part_count
        map_proofs.append({"image_identity": row["image_identity"], "source_sha256": row["source_sha256"], "local_st_sha256": row["local_st_sha256"], "global_stae_sha256": row["global_stae_sha256"], "local_st_shape": row["local_st_shape"], "global_stae_shape": row["global_stae_shape"], "local_st_artifact": local_artifact, "global_stae_artifact": global_artifact})
    std = math.sqrt(m2 / count)
    payload = {"schema_version": "1.0", "operation": COMMAND, "status": "CALIBRATED_VALIDATION_ONLY", "run_id": args.run_id,
               "decision_id": freeze["decision_id"], "split_freeze": {"path": freeze_path.name, "sha256": freeze_hash, "freeze_sha256": freeze["freeze_sha256"]},
               "checkpoint_sha256": freeze["checkpoint_sha256"], "validation_maps": map_proofs, "map_count": VALIDATION_COUNT,
               "output_shape": list(SPLIT_TARGET_SHAPE), "pixel_count": count, "formula": FORMULA, "population_mean": mean,
               "population_standard_deviation": std, "threshold": mean + 3.0 * std,
               "decision": {"comparator": COMPARATOR, "image_rule": "any pixel > threshold", "provenance": "project decision; not claimed official MVTec comparator"},
               "official_utility_scope": "Inspected MVTec AD 2 public utility validates binary {0,255} submission masks but does not mandate > versus >= comparator semantics.",
               "official_archive_hashes": _archives(), "test_access": "NONE", "combined_map_persistence": "NONE"}
    proof_hash = sha256(_canonical(payload)).hexdigest()
    data = _canonical({**payload, "calibration_sha256": proof_hash})
    path = _write(root, args.run_id, data, admit=admit, writer=writer)
    return {"status": READY, "artifact": str(path), "calibration_sha256": proof_hash, "threshold": payload["threshold"], "pixel_count": count, "comparator": COMPARATOR}


def parse_args(argv: Sequence[str] | None = None) -> SplitCalibrationInput:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--split-freeze", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    value = parser.parse_args(argv)
    return SplitCalibrationInput(value.artifact_root, value.split_freeze, value.run_id)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(calibrate(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def load_calibration_artifact(path: Path, *, split_freeze: Path, checkpoint_sha256: str) -> tuple[float, str]:
    """Read only a canonical, validation-only split calibration bound to this model."""
    raw=Path(path).read_bytes()
    try: value=json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError('invalid calibration artifact') from exc
    if not isinstance(value,dict) or raw != _canonical(value): raise ValueError('calibration artifact must be canonical JSON')
    digest=value.get('calibration_sha256'); core={k:v for k,v in value.items() if k!='calibration_sha256'}
    if not _sha(digest) or sha256(_canonical(core)).hexdigest()!=digest: raise ValueError('calibration artifact hash mismatch')
    if value.get('status')!='CALIBRATED_VALIDATION_ONLY' or value.get('operation')!=COMMAND or value.get('checkpoint_sha256')!=checkpoint_sha256: raise ValueError('calibration lineage/status mismatch')
    if value.get('split_freeze',{}).get('sha256') != _hash(Path(split_freeze)): raise ValueError('calibration freeze mismatch')
    if value.get('decision',{}).get('comparator') != COMPARATOR: raise ValueError('calibration comparator mismatch')
    threshold=value.get('threshold')
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold): raise ValueError('calibration threshold invalid')
    return float(threshold),digest
