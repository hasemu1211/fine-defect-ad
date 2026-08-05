"""Identity-bound E1 raw-map extraction for the fixed MVTec AD 2 public test set.

This command deliberately emits maps/provenance and optional local AU-PRO only.  It
never recalibrates the frozen threshold or emits threshold decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence import immutable_json, new_evidence
from .g002_calibration import CalibrationInput, _admit_input
from .g002_eval_runtime import (TRANSFORM_IDENTITY, _MapModel, _lease_proof, _lease_record,
                                _pinned_transform, _vram, safe_load_checkpoint)
from .g002_pilot import G002Args, _lazy_runtime
from .g002_evaluate import _array, raw_map
from .gpu_lock import GpuLease
from .mvtec_aupro import local_au_pro_0_05
from .pilot import PilotEvidence, host_rss_bytes, lease_events
from .storage import Allocation, READY, STOPPED_INCOMPLETE, atomic_write, preflight

COMMAND = "g002-eval-test-public-e1"
CATEGORY = "sheet_metal"
GOOD_COUNT, BAD_COUNT = 24, 90


@dataclass(frozen=True)
class TestPublicArgs:
    artifact_root: Path; checkpoint: Path; sidecar: Path; metrics: Path; final_attempt: Path; training_identity: Path
    dataset_root: Path; teacher_small: Path; imagenette_root: Path; run_id: str; lease_directory: Path
    validation_manifest: Path; geometry_evidence: Path; geometry_evidence_sha256: str; geometry_decision_id: str
    pretest_freeze: Path; post_selection_binding: Path; evaluator: Path | None = None


def _hash(path: Path) -> str: return sha256(path.read_bytes()).hexdigest()
def _canon(value: Mapping[str, Any]) -> bytes: return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_public_entries(dataset_root: Path) -> list[dict[str, Any]]:
    """Return the exact, mask-bound public-test identity set; reject anything else."""
    root = Path(dataset_root).resolve() / CATEGORY / "test_public"
    rows: list[dict[str, Any]] = []
    for label, expected in (("good", GOOD_COUNT), ("bad", BAD_COUNT)):
        leaf = root / label
        files = sorted(path for path in leaf.glob("*.png") if path.is_file() and not path.is_symlink())
        if len(files) != expected:
            raise ValueError(f"test_public/{label} must contain exactly {expected} PNG files")
        for image in files:
            mask = None if label == "good" else root / "ground_truth" / "bad" / f"{image.stem}_mask.png"
            if mask is not None and (not mask.is_file() or mask.is_symlink()):
                raise ValueError(f"missing public-test mask: {image.name}")
            rows.append({"image_identity": f"test_public/{label}/{image.name}", "source": image,
                         "source_sha256": _hash(image), "mask": mask,
                         "mask_sha256": None if mask is None else _hash(mask), "label": label})
    if len(rows) != GOOD_COUNT + BAD_COUNT or len({row["image_identity"] for row in rows}) != len(rows):
        raise ValueError("public-test identity set mismatch")
    return rows


def _write(root: Path, run_id: str, maps: list[dict[str, Any]], binding: Mapping[str, Any], *, admit: Any, writer: Any) -> dict[str, Any]:
    manifest_rows = [{key: value for key, value in row.items() if key not in {"_bytes", "_mask"}} for row in maps]
    payload = _canon({"status": "TEST_PUBLIC_RAW_MAPS_ONLY", "run_id": run_id, "selected_measurement": "E1",
                      "transform_identity": TRANSFORM_IDENTITY, "lineage": dict(binding), "maps": manifest_rows,
                      "threshold_metrics": "BLOCKED_NO_VERIFIED_COMPARATOR"})
    raw_total = sum(len(row["_bytes"]) for row in maps); pending = max([len(payload), *(len(row["_bytes"]) for row in maps)])
    source = f"exact canonical public-test map bytes={raw_total}; manifest bytes={len(payload)}; pending atomic bytes={pending}"
    proof = admit(run_id=run_id, allocations=[Allocation("artifact", raw_total + len(payload), "persistent", source, "g002-test-public-raw-maps"), Allocation("artifact", pending, "transient", source, "g002-test-public-raw-maps-incoming")], reserve_bytes=pending, reserve_evidence={"max_pending_atomic_write_bytes": pending, "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ValueError("fresh proof artifact root changed")
    paths = []
    for index, row in enumerate(maps):
        destination = root / f"g002-test-public-raw-{index:03d}-{row['map_sha256']}.bin"
        result = writer(destination, row["_bytes"], proof=proof, run_id=run_id, overwrite=False)
        if result.get("status") != READY or _hash(destination) != row["map_sha256"]: raise ValueError("raw map write failed")
        paths.append(destination)
    manifest = root / f"g002-test-public-raw-maps-{run_id}.json"
    result = writer(manifest, payload, proof=proof, run_id=run_id, overwrite=False)
    if result.get("status") != READY or manifest.read_bytes() != payload: raise ValueError("manifest write failed")
    return {"manifest": str(manifest), "map_paths": [str(path) for path in paths]}


def _final(root: Path, run_id: str, record: dict[str, Any], *, admit: Any, writer: Any) -> dict[str, Any]:
    payload, digest = immutable_json(record); source = f"exact immutable public-test evidence bytes={len(payload)} sha256={digest}"
    proof = admit(run_id=run_id, allocations=[Allocation("artifact", len(payload), "persistent", source, "g002-test-public-final-evidence"), Allocation("artifact", len(payload), "transient", source, "g002-test-public-final-evidence-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ValueError("fresh proof artifact root changed")
    path = root / f"g002-test-public-evidence-{run_id}-{digest}.json"; result = writer(path, payload, proof=proof, run_id=run_id, overwrite=False)
    if result.get("status") != READY or path.read_bytes() != payload: raise ValueError("evidence write failed")
    return {**record, "artifact": str(path), "artifact_sha256": digest}


def evaluate_persisted_test_public(*, artifact_root: Path, dataset_root: Path, raw_manifest: Path,
                                   evaluator: Path, run_id: str, admit: Any = preflight,
                                   writer: Any = atomic_write) -> dict[str, Any]:
    """Compute local AU-PRO from the already immutable public-test raw maps only."""
    import numpy as np
    from anomalib.data.utils.image import read_mask
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import Resize

    root = Path(artifact_root).resolve(); manifest_path = Path(raw_manifest).resolve()
    if manifest_path.parent != root: raise ValueError("raw manifest must be directly under artifact root")
    try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ValueError("invalid public-test raw-map manifest") from exc
    if manifest.get("status") != "TEST_PUBLIC_RAW_MAPS_ONLY" or manifest.get("selected_measurement") != "E1":
        raise ValueError("only the frozen E1 public-test raw map manifest is accepted")
    rows, current = manifest.get("maps"), test_public_entries(dataset_root)
    if not isinstance(rows, list) or len(rows) != len(current): raise ValueError("public-test raw map count mismatch")
    lineage = manifest.get("lineage")
    expected_lineage = {"checkpoint_sha256", "sidecar_sha256", "metrics_sha256", "final_attempt_sha256", "identity_sha256", "pilot_sha256", "freeze_sha256", "post_selection_binding_sha256"}
    if not isinstance(lineage, Mapping) or set(lineage) != expected_lineage or any(not isinstance(value, str) or len(value) != 64 for value in lineage.values()):
        raise ValueError("public-test lineage mismatch")
    resize = Resize((256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True)
    maps, masks = [], []
    for index, (row, entry) in enumerate(zip(rows, current)):
        if not isinstance(row, Mapping) or any(row.get(key) != entry[key] for key in ("image_identity", "label", "source_sha256", "mask_sha256")):
            raise ValueError("public-test source or mask identity/hash mismatch")
        if row.get("checkpoint_sha256") != lineage["checkpoint_sha256"] or row.get("dtype") != "<f4" or row.get("byte_order") != "<":
            raise ValueError("public-test map provenance mismatch")
        shape = row.get("shape")
        if not isinstance(shape, list) or shape != [1, 1, 256, 256]: raise ValueError("public-test map shape mismatch")
        digest = row.get("map_sha256")
        if not isinstance(digest, str) or len(digest) != 64: raise ValueError("public-test map hash missing")
        path = root / f"g002-test-public-raw-{index:03d}-{digest}.bin"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != digest or len(raw) != 256 * 256 * 4: raise ValueError("public-test raw-map bytes/hash mismatch")
        maps.append(np.frombuffer(raw, dtype="<f4").reshape(256, 256))
        masks.append(None if entry["mask"] is None else _array(resize(read_mask(entry["mask"], as_tensor=True))).squeeze())
    started = time.monotonic()
    metric = local_au_pro_0_05(maps, masks, evaluator, include_curve=False)
    record = new_evidence(run_id, "g002-eval-test-public-au-pro-0.05", READY, [])
    record.update({"protocol": "TEST_PUBLIC_RAW_MAPS_ONLY_NO_SELECTION_OR_CALIBRATION_MUTATION",
                   "counts": {"good": GOOD_COUNT, "bad": BAD_COUNT, "total": GOOD_COUNT + BAD_COUNT},
                   "selected_measurement": "E1", "raw_manifest": str(manifest_path),
                   "raw_manifest_sha256": _hash(manifest_path), "lineage": dict(lineage),
                   "mask_transform_identity": TRANSFORM_IDENTITY, "local_au_pro": metric,
                   "comparator": None, "threshold_metrics": "BLOCKED_NO_VERIFIED_COMPARATOR",
                   "timing_seconds": time.monotonic() - started})
    return _final(root, run_id, record, admit=admit, writer=writer)


def run_test_public(args: TestPublicArgs, *, runtime_factory: Any = _lazy_runtime, lease_factory: Any = GpuLease,
                    torch_module: Any | None = None, admit: Any = preflight, writer: Any = atomic_write,
                    lease_event_loader: Any = lease_events) -> dict[str, Any]:
    root = Path(args.artifact_root).resolve(); started = time.monotonic(); failure = None; persisted = None; metrics = None
    lease_outcome, recorded_lease, lease_entered = "not_acquired", None, False; torch = torch_module
    try:
        _lease_proof(root, args, admit)
        if Path(args.lease_directory).resolve().parent != root and root not in Path(args.lease_directory).resolve().parents: raise ValueError("lease directory must be beneath artifact root")
        calibration = CalibrationInput(root, args.run_id, args.validation_manifest, args.training_identity, args.checkpoint, args.sidecar, args.metrics, args.final_attempt, args.dataset_root, args.geometry_evidence, args.geometry_evidence_sha256, args.geometry_decision_id, args.pretest_freeze, args.post_selection_binding)
        _manifest, _records, _geometry, freeze = _admit_input(calibration)
        if freeze["selection"]["selected"] != "E1": raise ValueError("TESTpub command permits selected E1 only")
        entries = test_public_entries(args.dataset_root)
        binding = {key: freeze[key] for key in ("checkpoint_sha256", "sidecar_sha256", "metrics_sha256", "final_attempt_sha256", "identity_sha256", "pilot_sha256", "freeze_sha256")}
        binding["post_selection_binding_sha256"] = _hash(args.post_selection_binding)
        if torch is None: import torch
        checkpoint = safe_load_checkpoint(args.checkpoint, binding["checkpoint_sha256"], torch)
        with lease_factory(args.lease_directory, args.run_id, COMMAND):
            lease_entered = True
            if not bool(torch.cuda.is_available()): raise RuntimeError("CUDA_UNAVAILABLE")
            device = torch.device("cuda:0")
            g002 = G002Args(args.dataset_root, args.teacher_small, args.imagenette_root, args.run_id, args.lease_directory)
            model, _data, _trainer, _validator = runtime_factory(g002, PilotEvidence(args.run_id, COMMAND, 70_000), started, pilot_steps=None)
            model.load_state_dict(checkpoint["state_dict"]); model.eval(); model.to(device); transform = _pinned_transform(model.pre_processor.transform)
            from anomalib.data.utils.image import read_image, read_mask
            rows = []
            for entry in entries:
                image = read_image(entry["source"], as_tensor=True); mask = None if entry["mask"] is None else read_mask(entry["mask"], as_tensor=True)
                if mask is None: image = transform(image)
                else: image, mask = transform(image, mask)
                image = image.unsqueeze(0).to(device)
                value = raw_map(*_MapModel(model, torch).get_maps(image, normalize=False)); raw = value.tobytes(order="C")
                row = {"image_identity": entry["image_identity"], "label": entry["label"], "source_sha256": entry["source_sha256"], "mask_sha256": entry["mask_sha256"], "map_sha256": sha256(raw).hexdigest(), "dtype": "<f4", "shape": list(value.shape), "byte_order": "<", "checkpoint_sha256": binding["checkpoint_sha256"], "_bytes": raw, "_mask": None if mask is None else _array(mask).squeeze().astype("<f4", copy=False)}
                rows.append(row)
            persisted = _write(root, args.run_id, rows, binding, admit=admit, writer=writer)
            if args.evaluator is None:
                metrics = {"status": "BLOCKED_NO_HASH_VERIFIED_OFFICIAL_MVTEC_AD_V1_EVALUATOR", "comparator": None, "threshold_metrics": "BLOCKED_NO_VERIFIED_COMPARATOR"}
            else:
                import numpy as np
                maps = [np.frombuffer(row["_bytes"], dtype="<f4").reshape(row["shape"]).squeeze() for row in rows]
                metrics = local_au_pro_0_05(maps, [row["_mask"] for row in rows], args.evaluator)
            lease_outcome = "normal"
    except Exception as exc:
        failure, lease_outcome = f"RUNNER:{type(exc).__name__}:{exc}", "exception"
    if lease_entered:
        try: recorded_lease = _lease_record(lease_event_loader(args.lease_directory, args.run_id), args.run_id, lease_outcome, expected_command=COMMAND, expected_pid=os.getpid())
        except Exception as exc: failure, lease_outcome = f"LEASE_EVIDENCE:{type(exc).__name__}:{exc}", "invalid"
    record = new_evidence(args.run_id, COMMAND, READY if failure is None else STOPPED_INCOMPLETE, [] if failure is None else [failure])
    record.update({"selected_measurement": "E1", "protocol": "TEST_PUBLIC_ONLY_NO_SELECTION_OR_CALIBRATION_MUTATION", "counts": {"good": GOOD_COUNT, "bad": BAD_COUNT, "total": GOOD_COUNT + BAD_COUNT}, "threshold_metrics": "BLOCKED_NO_VERIFIED_COMPARATOR", "comparator": None, "timing_seconds": time.monotonic() - started, "rss_bytes": host_rss_bytes(), "lease_outcome": lease_outcome})
    if persisted is not None: record["raw_maps"] = persisted
    if metrics is not None: record["local_au_pro"] = metrics
    if recorded_lease is not None: record["lease_events"] = recorded_lease
    if torch is not None:
        try: record["vram"] = _vram(torch)
        except Exception: record["vram"] = {"allocated_bytes": None, "reserved_bytes": None}
    try: return _final(root, args.run_id, record, admit=admit, writer=writer)
    except Exception: return record


def parse_args(argv: Sequence[str] | None = None) -> TestPublicArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "checkpoint", "sidecar", "metrics", "final-attempt", "training-identity", "dataset-root", "teacher-small", "imagenette-root", "lease-directory", "validation-manifest", "geometry-evidence", "pretest-freeze", "post-selection-binding"): parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--geometry-evidence-sha256", required=True); parser.add_argument("--geometry-decision-id", required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--evaluator", type=Path)
    return TestPublicArgs(**{key.replace("-", "_"): value for key, value in vars(parser.parse_args(argv)).items()})


def main(argv: Sequence[str] | None = None) -> int:
    result = run_test_public(parse_args(argv)); print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0 if result["status"] == READY else 2

if __name__ == "__main__": raise SystemExit(main())
