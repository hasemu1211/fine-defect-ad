from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fine_defect_ad import highres_split as public


def _args(root: Path, operation: str = "validation"):
    files = {}
    for name in ("checkpoint", "metrics", "final_attempt", "training_identity", "split_freeze", "evaluator"):
        path = root / f"{name}.bin"; path.write_bytes(name.encode()); files[name] = path
    return SimpleNamespace(operation=operation, artifact_root=root, run_id="run-1", mode="e2-split", repeat=2, **files)


def test_public_help_and_dispatch(capsys, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        public.parse_args(["--help"])
    assert exc.value.code == 0
    assert "Public E2-Split high-resolution inference and evaluation entry point" in capsys.readouterr().out
    args = _args(tmp_path)
    monkeypatch.setattr(public._runner, "run_validation", lambda value: {"status": "READY", "run_id": value.run_id})
    assert public.run(args) == {"status": "READY", "run_id": "run-1"}


def test_validation_failure_writes_private_digest_named_record(monkeypatch, tmp_path):
    args = _args(tmp_path)
    def write(root, run_id, name, payload):
        path = root / name; path.write_bytes(payload); return path
    monkeypatch.setattr(public._runner, "_write", write)
    monkeypatch.setattr(public._runner, "run_validation", lambda _args: (_ for _ in ()).throw(RuntimeError("/private/source.png")))
    with pytest.raises(RuntimeError, match="private/source"):
        public.run(args)
    records = list(tmp_path.glob("highres-split-FAILED-run-1-*.json"))
    assert len(records) == 1
    payload = records[0].read_bytes(); record = json.loads(payload)
    assert records[0].stem.endswith(sha256(payload).hexdigest())
    assert record["public_pipeline"] == public.PUBLIC_PIPELINE and record["operation"] == "validation"
    assert record["exception_type"] == "RuntimeError" and "/" not in record["exception_message"] and record["stage"] == "validation"
    assert record["exception_fingerprint_sha256"] == sha256(b"RuntimeError:/private/source.png").hexdigest()
    assert record["input_sha256"]["checkpoint"] == sha256(b"checkpoint").hexdigest()
    assert str(tmp_path) not in payload.decode()


def test_failure_record_error_preserves_original_exception(monkeypatch, tmp_path):
    args = _args(tmp_path)
    monkeypatch.setattr(public._runner, "run_validation", lambda _args: (_ for _ in ()).throw(ValueError("original failure")))
    monkeypatch.setattr(public, "_write_failure_record", lambda *_args: (_ for _ in ()).throw(OSError("storage failed")))
    with pytest.raises(ValueError, match="original failure"):
        public.run(args)

def test_invalid_operation_is_blocked_before_runner(monkeypatch, tmp_path):
    args = _args(tmp_path, operation="unknown")
    monkeypatch.setattr(public._runner, "run_test_public_once", lambda _: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="unsupported public operation"):
        public.run(args)


def test_infer_writes_path_free_manifest_without_testpub(monkeypatch, tmp_path):
    import sys
    import numpy as np
    args = _args(tmp_path, operation="infer")
    args.input_image = tmp_path / "input.png"; args.input_image.write_bytes(b"image")
    args.dataset_root = tmp_path / "dataset"; args.lease_directory = tmp_path / "lease"
    args.split_freeze.write_text(json.dumps({"quantiles": {"qa_st": 0, "qb_st": 1, "qa_stae": 0, "qb_stae": 1}}))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), device=lambda _: "cuda:0")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(public, "verify_split_freeze", lambda _: None)
    monkeypatch.setattr(public, "GpuLease", lambda *_: __import__("contextlib").nullcontext())
    monkeypatch.setattr(public._runner, "_model", lambda *_: (SimpleNamespace(checkpoint_sha256="c" * 64), object()))
    monkeypatch.setattr(public, "decode_rgb01", lambda _: "rgb")
    monkeypatch.setattr(public, "split_branch_raw_maps", lambda *_args, **_kwargs: ("local", "global", {"tile": 256, "stride": 128, "boxes": [[0, 0, 1, 1]], "weight_min": 1, "weight_max": 1}))
    monkeypatch.setattr(public, "combine_split_maps", lambda *_: np.zeros((2, 3), dtype="<f4"))
    monkeypatch.setattr(public._runner, "test_public_entries", lambda *_: pytest.fail("infer must not access TESTpub"), raising=False)
    def write(root, _run, name, payload):
        path = root / name; path.write_bytes(payload); return path
    monkeypatch.setattr(public._runner, "_write", write)
    result = public.run(args)
    manifest = Path(result["manifest"]).read_text()
    assert result["status"] == "READY" and str(args.input_image) not in manifest
    assert json.loads(manifest)["input_sha256"] == sha256(b"image").hexdigest()

def test_infer_rejects_any_dataset_root_input(tmp_path):
    args = _args(tmp_path, operation="infer")
    args.dataset_root = tmp_path / "dataset"; args.dataset_root.mkdir()
    args.input_image = args.dataset_root / "validation.png"; args.input_image.write_bytes(b"image")
    with pytest.raises(ValueError, match="dataset-root input"):
        public._reject_dataset_input(args)

def test_infer_lineage_mismatch_fails_before_decode(monkeypatch, tmp_path):
    import sys
    args = _args(tmp_path, operation="infer"); args.input_image = tmp_path / "input.png"; args.input_image.write_bytes(b"image")
    args.dataset_root = tmp_path / "dataset"; args.lease_directory = tmp_path / "lease"
    args.split_freeze.write_text(json.dumps({"checkpoint_sha256": "x" * 64, "quantiles": {}}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), device=lambda _: "cuda:0"))
    monkeypatch.setattr(public, "verify_split_freeze", lambda _: None); monkeypatch.setattr(public, "GpuLease", lambda *_: __import__("contextlib").nullcontext())
    monkeypatch.setattr(public._runner, "_model", lambda *_: (SimpleNamespace(checkpoint_sha256="c" * 64), object()))
    monkeypatch.setattr(public, "decode_rgb01", lambda _: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="lineage mismatch"):
        public.run(args)


def test_infer_mode_validation(tmp_path):
    args = _args(tmp_path, operation="infer"); args.mode = "e1"; args.repeat = 1
    with pytest.raises(ValueError, match="repeat"):
        public.run_inference(args)

def test_e1_wrapper_uses_core_raw_map_and_resets_peak(monkeypatch, tmp_path):
    import sys
    import numpy as np
    args = _args(tmp_path, operation="infer"); args.mode = "e1"; args.split_freeze = None
    args.input_image = tmp_path / "input.png"; args.input_image.write_bytes(b"image")
    args.dataset_root = tmp_path / "dataset"; args.lease_directory = tmp_path / "lease"
    reset = []; cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: reset.append(True), max_memory_allocated=lambda: 3, max_memory_reserved=lambda: 4)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, device=lambda _: "cuda:0"))
    monkeypatch.setattr(public, "GpuLease", lambda *_: __import__("contextlib").nullcontext())
    monkeypatch.setattr(public, "decode_rgb01", lambda _: "rgb"); monkeypatch.setattr(public, "canonical_256", lambda *_args, **_kwargs: "canonical")
    class Core:
        def get_maps(self, image, normalize=False):
            assert image == "canonical" and normalize is False
            return np.ones((1, 1, 2, 3), dtype=np.float32), np.full((1, 1, 2, 3), 3, dtype=np.float32)
    monkeypatch.setattr(public._runner, "_model", lambda *_: (SimpleNamespace(checkpoint_sha256="c" * 64), SimpleNamespace(model=Core())))
    ticks = iter((0, .002, .003, .007)); monkeypatch.setattr(public.time, "perf_counter", lambda: next(ticks))
    def write(root, _run, name, payload):
        path = root / name; path.write_bytes(payload); return path
    monkeypatch.setattr(public._runner, "_write", write)
    result = public.run(args); manifest = json.loads(Path(result["manifest"]).read_text())
    assert reset == [True] and manifest["decision_id"] == "DEC-GEO-002" and manifest["smoke"]["median_map_latency_ms"] == pytest.approx(3.0)
    assert np.frombuffer(Path(result["raw_map"]).read_bytes(), dtype="<f4").tolist() == [2.0] * 6 and Path(result["heatmap"]).is_file()


def test_delegated_failure_records_operation_stage(monkeypatch, tmp_path):
    args = _args(tmp_path, operation="validation")
    def write(root, _run, name, payload):
        path = root / name; path.write_bytes(payload); return path
    monkeypatch.setattr(public._runner, "_write", write)
    monkeypatch.setattr(public._runner, "run_validation", lambda _: (_ for _ in ()).throw(RuntimeError("fail")))
    with pytest.raises(RuntimeError): public.run(args)
    record = json.loads(next(tmp_path.glob("highres-split-FAILED-*.json")).read_text())
    assert record["stage"] == "validation" and record["decision_id"] == "DEC-SPLIT-003"

def test_e1_failure_payload_uses_baseline_decision(tmp_path):
    args = _args(tmp_path, operation="infer"); args.mode = "e1"
    assert json.loads(public._failure_payload(args, RuntimeError("x")))["decision_id"] == "DEC-GEO-002"
