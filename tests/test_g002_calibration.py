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
    measurement = {"maps": maps, "geometry": {}, "probe_summary": {}}
    freeze = {"stage": "PRE_TEST_FREEZE", "status": "FROZEN", "decision_id": "DEC-GEO-002", "selection": {"selected": "E1"}, **checkpoint_lineage, "validation_identities": validation, "hardware": {}, "e1_measurement": measurement, "e1_measurement_sha256": digest(canonical(measurement)), "e2_measurement": measurement, "e2_measurement_sha256": digest(canonical(measurement)), "geometry": {}, "revision": {}}
    freeze["freeze_sha256"] = digest(canonical(freeze))
    freeze_path = root / f"g002-e2-pretest-freeze-run-{freeze['freeze_sha256']}.json"
    freeze_path.write_bytes(canonical(freeze))
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
        freeze_path,
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

def test_calibration_e2_freeze_manifest_binding_and_cli_help():
    import os
    import subprocess
    import sys
    from PIL import Image
    with TemporaryDirectory() as directory:
        root = Path(directory)
        args, manifest_path, _geometry = fixture(root)
        # E2 validates original dimensions; make the admitted 19 sources real 256x256 images.
        for source in (args.dataset_root / "sheet_metal" / "validation" / "good").glob("*.png"):
            Image.new("RGB", (256, 256)).save(source)
        identity = json.loads(args.training_identity.read_text())
        for row in identity["data"]["validation"]:
            row["sha256"] = digest(args.dataset_root / "sheet_metal" / row["path"])
        raw_identity = canonical(identity); identity_hash = digest(raw_identity)
        identity_path = root / f"g002-training-identity-run-{identity_hash}.json"; identity_path.write_bytes(raw_identity)
        sidecar = json.loads(args.sidecar.read_text()); sidecar["identity_sha256"] = identity_hash; args.sidecar.write_text(json.dumps(sidecar, sort_keys=True))
        checkpoint = {"checkpoint_sha256": digest(args.checkpoint), "sidecar_sha256": digest(args.sidecar), "metrics_sha256": digest(args.metrics), "final_attempt_sha256": digest(args.final_attempt), "identity_sha256": identity_hash, "pilot_sha256": PILOT_SHA256}
        # Final attempt's hash is part of checkpoint lineage; regenerate its immutable binding.
        final = json.loads(args.final_attempt.read_text()); final["artifacts"] = {"checkpoint": checkpoint["checkpoint_sha256"], "sidecar": checkpoint["sidecar_sha256"], "metrics": checkpoint["metrics_sha256"]}
        raw_final = json.dumps(final, sort_keys=True).encode(); final_path = root / f"g002-attempt-run-{digest(raw_final)}.json"; final_path.write_bytes(raw_final); checkpoint["final_attempt_sha256"] = digest(final_path)
        border = 16; maps = []
        for index, row in enumerate(identity["data"]["validation"]):
            raw = struct.pack("<f", float(index)); h = digest(raw); path = root / f"g002-e2-validation-raw-b016-{index:02d}-{h}.bin"; path.write_bytes(raw)
            maps.append({"image_identity": row["path"], "source_sha256": row["sha256"], "map_sha256": h, "dtype": "<f4", "shape": [256, 256], "byte_order": "<", "checkpoint_sha256": checkpoint["checkpoint_sha256"], "coverage_min": 1, "coverage_max": 1, "seam_max_abs": 0.0, "border": border, "artifact": str(path)})
            # shape requires exactly 256*256 floats, not a convenience one-value map.
            path.write_bytes(raw * (256 * 256)); maps[-1]["map_sha256"] = digest(path); renamed = root / f"g002-e2-validation-raw-b016-{index:02d}-{maps[-1]['map_sha256']}.bin"; path.rename(renamed); maps[-1]["artifact"] = str(renamed)
        geometry = {"empirical_border": border}; probe = {"cases": []}
        manifest = {"status": "E2_RAW_MAPS_ONLY", "run_id": "run", "checkpoint": checkpoint, "maps": maps, "geometry": geometry, "probe_summary": probe, "claim": "NO_EXTERNAL_MINIMUM_AVAILABLE"}
        e2_manifest = root / "g002-e2-validation-raw-maps-run.json"; e2_manifest.write_bytes(canonical(manifest))
        e1 = {"maps": [], "geometry": {}, "probe_summary": {}}; e2 = {"maps": maps, "geometry": geometry, "probe_summary": probe}
        freeze = {"stage":"PRE_TEST_FREEZE","status":"FROZEN","decision_id":"DEC-GEO-002","selection":{"selected":"E2"},**checkpoint,"validation_identities":identity["data"]["validation"],"hardware":{},"e1_measurement":e1,"e1_measurement_sha256":digest(canonical(e1)),"e2_measurement":e2,"e2_measurement_sha256":digest(canonical(e2)),"geometry":geometry,"revision":{"e2_eligible":True}}
        freeze["freeze_sha256"] = digest(canonical(freeze)); freeze_path = root / f"g002-e2-pretest-freeze-run-{freeze['freeze_sha256']}.json"; freeze_path.write_bytes(canonical(freeze))
        e2_args = CalibrationInput(root, "run", e2_manifest, identity_path, args.checkpoint, args.sidecar, args.metrics, final_path, args.dataset_root, args.geometry_evidence, args.geometry_evidence_sha256, args.geometry_decision_id, freeze_path)
        result = calibrate(e2_args, admit=gate(root, {}), writer=writer)
        assert result["selected_measurement"] == "E2" and result["pretest_freeze"]["freeze_sha256"] == freeze["freeze_sha256"]
        freeze["selection"] = {"selected": "E1"}; freeze["freeze_sha256"] = digest(canonical({k:v for k,v in freeze.items() if k != "freeze_sha256"})); freeze_path.write_bytes(canonical(freeze))
        with pytest.raises(ValueError): calibrate(e2_args, admit=gate(root, {}), writer=writer)
    got = subprocess.run([sys.executable, "-m", "fine_defect_ad.g002_calibration", "--help"], env={**os.environ, "PYTHONPATH": f"{Path.cwd() / 'src'}:{Path.cwd() / '.internal/venv/r1-overlay'}"}, text=True, capture_output=True)
    assert got.returncode == 0 and "--pretest-freeze" in got.stdout
