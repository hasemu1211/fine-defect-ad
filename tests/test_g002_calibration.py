import json
import math
import struct
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import fine_defect_ad.g002_calibration as calibration
from fine_defect_ad.g002_calibration import CalibrationInput, calibrate
from fine_defect_ad.g002_training import PILOT_SHA256
from fine_defect_ad.storage import PreflightProof, READY


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(value if isinstance(value, bytes) else Path(value).read_bytes()).hexdigest()


def fixture(root):
    dataset = root / "data"
    leaf = dataset / "sheet_metal" / "validation" / "good"
    leaf.mkdir(parents=True, exist_ok=True)
    validation = []
    for index in range(19):
        source = leaf / f"{index:02d}.png"
        source.write_bytes(f"source-{index}".encode())
        validation.append({"path": f"validation/good/{source.name}", "sha256": digest(source)})
    identity = {"data": {"validation": validation}}
    raw_identity = canonical(identity)
    identity_hash = digest(raw_identity)
    identity_path = root / f"g002-training-identity-run-{identity_hash}.json"
    identity_path.write_bytes(raw_identity)
    checkpoint = root / "g002-last-run-0.ckpt"
    checkpoint.write_bytes(b"complete checkpoint")
    checkpoint_hash = digest(checkpoint)
    sidecar = {
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "identity_sha256": identity_hash,
        "pilot_sha256": PILOT_SHA256,
        "global_step": 70_000,
        "lineage": "run",
    }
    sidecar_path = checkpoint.with_suffix(".ckpt.json")
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True))
    metrics_path = root / "g002-metrics-run.json"
    metrics_path.write_text("[]")
    checkpoint_lineage = {
        "checkpoint_sha256": checkpoint_hash,
        "sidecar_sha256": digest(sidecar_path),
        "metrics_sha256": digest(metrics_path),
        "identity_sha256": identity_hash,
        "pilot_sha256": PILOT_SHA256,
    }
    final = {
        "run_id": "run",
        "status": READY,
        "lease_outcome": "normal",
        "artifacts": {
            "checkpoint": checkpoint_lineage["checkpoint_sha256"],
            "sidecar": checkpoint_lineage["sidecar_sha256"],
            "metrics": checkpoint_lineage["metrics_sha256"],
        },
    }
    raw_final = json.dumps(final, sort_keys=True).encode()
    final_path = root / f"g002-attempt-run-{digest(raw_final)}.json"
    final_path.write_bytes(raw_final)
    checkpoint_lineage["final_attempt_sha256"] = digest(final_path)
    maps = []
    for index, item in enumerate(validation):
        payload = struct.pack("<ff", float(index), float(index + 1))
        map_hash = digest(payload)
        path = root / f"g002-validation-raw-{index:02d}-{map_hash}.bin"
        path.write_bytes(payload)
        maps.append({
            "image_identity": item["path"],
            "source_sha256": item["sha256"],
            "map_sha256": map_hash,
            "dtype": "<f4",
            "shape": [2],
            "byte_order": "<",
            "checkpoint_sha256": checkpoint_hash,
        })
    manifest = {
        "status": "RAW_MAPS_ONLY",
        "run_id": "run",
        "transform_identity": {"normalize": False, "resize": 256, "interpolation": "bilinear"},
        "checkpoint": checkpoint_lineage,
        "maps": maps,
    }
    manifest_path = root / "g002-validation-raw-maps-run.json"
    manifest_path.write_bytes(canonical(manifest))
    geometry = {"decision_id": "DEC-GEO-FINAL", "status": "FROZEN"}
    raw_geometry = canonical(geometry)
    geometry_path = root / "g002-geometry-frozen.json"
    geometry_path.write_bytes(raw_geometry)
    args = CalibrationInput(
        root,
        "run",
        manifest_path,
        identity_path,
        checkpoint,
        sidecar_path,
        metrics_path,
        final_path,
        dataset,
        geometry_path,
        digest(raw_geometry),
        "DEC-GEO-FINAL",
    )
    return args, manifest_path, geometry_path


def gate(root, seen):
    def admit(**kwargs):
        seen.update(kwargs)
        return PreflightProof("run", {"artifact": str(root)}, "x", "2000-01-01T00:00:00+00:00", {}, [], {})

    return admit


def writer(path, payload, **_kwargs):
    Path(path).write_bytes(payload)
    return {"status": READY}


def test_calibration_known_array_oracle_and_blocks_all_decisions():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        args, _manifest, _geometry = fixture(root)
        seen = {}
        result = calibrate(args, admit=gate(root, seen), writer=writer)
        values = [item for index in range(19) for item in (float(index), float(index + 1))]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((item - mean) ** 2 for item in values) / len(values))
        assert result["pixel_count"] == len(values)
        assert result["raw_threshold"] == pytest.approx(mean + 3 * std)
        assert result["comparator"] is None and set(result["blocked"].values()) >= {"BLOCKED"}
        assert len(result["per_image_max_raw_scores"]) == 19
        assert seen["allocations"][0].bytes == Path(result["artifact"]).stat().st_size


@pytest.mark.parametrize("tamper", ["map", "manifest", "identity", "checkpoint", "sidecar", "metrics", "final", "source"])
def test_calibration_rejects_tampered_admitted_lineage(tamper):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        args, manifest, _geometry = fixture(root)
        paths = {
            "map": next(root.glob("g002-validation-raw-00-*.bin")),
            "manifest": manifest,
            "identity": args.training_identity,
            "checkpoint": args.checkpoint,
            "sidecar": args.sidecar,
            "metrics": args.metrics,
            "final": args.final_attempt,
            "source": args.dataset_root / "sheet_metal" / "validation" / "good" / "00.png",
        }
        paths[tamper].write_bytes(b"tampered")
        with pytest.raises(ValueError):
            calibrate(args, admit=gate(root, {}), writer=writer)


def test_calibration_rejects_test_leakage_and_geometry_binding_change():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        args, manifest_path, _geometry = fixture(root)
        manifest = json.loads(manifest_path.read_text())
        manifest["maps"][0]["image_identity"] = "test/good/00.png"
        manifest_path.write_bytes(canonical(manifest))
        with pytest.raises(ValueError):
            calibrate(args, admit=gate(root, {}), writer=writer)
        args, _manifest, geometry_path = fixture(root)
        geometry_path.write_bytes(canonical({"decision_id": "DEC-GEO-FINAL", "status": "FROZEN", "other": "changed"}))
        with pytest.raises(ValueError):
            calibrate(args, admit=gate(root, {}), writer=writer)


def test_calibration_detects_raw_map_mutation_after_admission(monkeypatch):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        args, _manifest, _geometry = fixture(root)
        original = calibration._stats

        def mutate_then_stream(records):
            records = list(records)
            records[0].path.write_bytes(struct.pack("<ff", 99.0, 100.0))
            return original(records)

        monkeypatch.setattr(calibration, "_stats", mutate_then_stream)
        with pytest.raises(ValueError, match="changed during calibration"):
            calibrate(args, admit=gate(root, {}), writer=writer)
