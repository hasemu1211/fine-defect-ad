import json
import os
import subprocess
import sys
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from fine_defect_ad.g002_eval_runtime import (COMMAND, EvaluationArgs, TRANSFORM_IDENTITY, _batch_values, _lease_record, load_training_identity,
                                               run_evaluation, safe_load_checkpoint)
from fine_defect_ad.g002_training import PILOT_SHA256
from fine_defect_ad.storage import PreflightProof, READY


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def inputs(root):
    data = root / "data"; leaf = data / "sheet_metal" / "validation" / "good"; leaf.mkdir(parents=True)
    rows, batches = [], []
    for i in range(19):
        path = leaf / f"{i:02}.png"; path.write_bytes(f"image-{i}".encode()); key = f"validation/good/{path.name}"
        rows.append({"path": key, "sha256": digest(path)}); batches.append({"image": np.asarray([i, i + 1]), "image_path": [str(path)]})
    return data, {"data": {"validation": rows}, "protocol": "canonical"}, batches


def training_artifacts(root, identity):
    checkpoint = root / "g002-last-train-0.ckpt"; checkpoint.write_bytes(b"lightning-checkpoint")
    metrics = root / "g002-metrics-train.json"; metrics.write_bytes(b"[]")
    sidecar = {"checkpoint_name": checkpoint.name, "checkpoint_sha256": digest(checkpoint), "identity_sha256": sha256(canonical(identity)).hexdigest(), "pilot_sha256": PILOT_SHA256, "global_step": 70000, "lineage": "train"}
    sidecar_path = checkpoint.with_suffix(".ckpt.json"); sidecar_path.write_bytes(json.dumps(sidecar, sort_keys=True).encode())
    attempt = {"run_id": "train", "status": READY, "lease_outcome": "normal", "artifacts": {"checkpoint": digest(checkpoint), "sidecar": digest(sidecar_path), "metrics": digest(metrics)}}
    payload = json.dumps(attempt, sort_keys=True).encode(); final = root / f"g002-attempt-train-{sha256(payload).hexdigest()}.json"; final.write_bytes(payload)
    identity_path = root / f"g002-training-identity-train-{sha256(canonical(identity)).hexdigest()}.json"; identity_path.write_bytes(canonical(identity))
    return checkpoint, sidecar_path, metrics, final, identity_path


class Tensor:
    def __init__(self, value): self.value = np.asarray(value); self.shape = (1, 3, 256, 256); self.device = None
    def to(self, device): self.device = device; return self

class Torch:
    entered = 0
    class serialization:
        called = None
        @staticmethod
        def safe_globals(values): Torch.serialization.called = values; return nullcontext()
    class cuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def max_memory_allocated(): return 12
        @staticmethod
        def max_memory_reserved(): return 34
    @staticmethod
    def device(value): assert value == "cuda:0"; return value
    @staticmethod
    def inference_mode():
        class Context:
            def __enter__(self): Torch.entered += 1
            def __exit__(self, *args): return False
        return Context()
    @staticmethod
    def load(path, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is True
        return {"state_dict": {"weight": 1}, "global_step": 70000}

class Lease:
    def __init__(self, *args): self.args = args
    def __enter__(self): return self
    def __exit__(self, *args): return False

def lease_events_for(outcome="normal"):
    return [{"state": "acquired", "run_id": "eval", "command": COMMAND, "pid": os.getpid(), "timestamp": "2026-01-01T00:00:00+00:00"}, {"state": "released", "run_id": "eval", "command": COMMAND, "pid": os.getpid(), "timestamp": "2026-01-01T00:00:01+00:00", "outcome": outcome}]

def test_identity_artifact_is_canonical_and_filename_bound():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); _, identity, _ = inputs(root); *_, identity_path = training_artifacts(root, identity)
        assert load_training_identity(identity_path, root) == (identity, "train")
        identity_path.write_text("{}")
        with pytest.raises(ValueError): load_training_identity(identity_path, root)


def test_cpu_end_to_end_wires_real_artifact_format_and_never_touches_test_loader():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); dataset, identity, unused = inputs(root); checkpoint, sidecar, metrics, final, identity_path = training_artifacts(root, identity)
        leaf = dataset / "sheet_metal" / "validation" / "good"
        batches = [{"image": Tensor([i, i + 1]), "image_path": [str(leaf / f"{i:02}.png")]} for i in range(19)]
        proof = PreflightProof("eval", {"artifact": str(root)}, "x", "2000-01-01T00:00:00+00:00", {}, [], {})
        admissions = []
        def admit(**kwargs): admissions.append(kwargs); assert kwargs["run_id"] == "eval"; return proof
        def writer(path, payload, **kwargs): Path(path).write_bytes(payload); return {"status": READY}
        class Resize:
            size = (256, 256); interpolation = "bilinear"
        class Transform:
            transforms = [Resize()]
        class RuntimeModel:
            class Inner:
                def get_maps(self, image, *, normalize): assert normalize is False and image.device == "cuda:0"; return image.value, image.value + 2
            model = Inner()
            class Pre: transform = Transform()
            pre_processor = Pre()
            def load_state_dict(self, state): assert state == {"weight": 1}
            def eval(self): self.evaluated = True; return self
            def to(self, device): assert self.evaluated and device == "cuda:0"; self.device = device; return self
        class DataModule:
            class Val: augmentations = None
            val_data = Val()
            def setup(self, stage): assert stage == "validate"
            def val_dataloader(self): assert type(self.val_data.augmentations).__name__ == "Transform"; return batches
            def test_dataloader(self): raise AssertionError("test loader forbidden")
        def runtime(*args, **kwargs): return RuntimeModel(), DataModule(), object(), object()
        args = EvaluationArgs(root, checkpoint, sidecar, metrics, final, identity_path, dataset, root / "teacher", root / "imagenette", "eval", root / "leases")
        result = run_evaluation(args, runtime_factory=runtime, lease_factory=Lease, torch_module=Torch, admit=admit, writer=writer, lease_event_loader=lambda *_: lease_events_for())
        assert result["status"] == READY and result["transform_identity"] == TRANSFORM_IDENTITY
        assert len(result["raw_maps"]["map_paths"]) == 19 and digest(result["artifact"]) == result["artifact_sha256"]
        assert Torch.serialization.called == [__import__("pathlib").PosixPath] and Torch.entered == 19
        assert result["lease_events"][-1]["outcome"] == "normal"
        assert [item.kind for item in admissions[0]["allocations"]] == ["persistent", "transient"]
        assert admissions[0]["reserve_bytes"] == admissions[0]["allocations"][1].bytes

def test_safe_load_rejects_missing_state_or_wrong_step():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.ckpt"; path.write_bytes(b"x")
        class BadTorch:
            class serialization:
                @staticmethod
                def safe_globals(values): return nullcontext()
            @staticmethod
            def load(*args, **kwargs): return {"state_dict": {}, "global_step": 1}
        with pytest.raises(ValueError, match="70000"): safe_load_checkpoint(path, digest(path), BadTorch)


def test_cli_help_lists_required_runtime_inputs():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run([sys.executable, "-m", "fine_defect_ad.g002_eval_runtime", "--help"], text=True, capture_output=True, check=True, env=env)
    assert "--training-identity" in result.stdout and "--final-attempt" in result.stdout and "--artifact-root" in result.stdout


def test_actual_anomalib_imagebatch_is_accepted_when_overlay_is_installed():
    anomalib = pytest.importorskip("anomalib")
    import torch
    from anomalib.data.dataclasses import ImageBatch
    batch = ImageBatch(image=torch.zeros((1, 3, 256, 256)), image_path=["/tmp/x.png"])
    image, path = _batch_values(batch)
    assert tuple(image.shape) == (1, 3, 256, 256) and path == ["/tmp/x.png"]


def test_cuda_unavailable_returns_stopped_immutable_evidence():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); dataset, identity, _ = inputs(root); checkpoint, sidecar, metrics, final, identity_path = training_artifacts(root, identity)
        proof = PreflightProof("eval", {"artifact": str(root)}, "x", "2000-01-01T00:00:00+00:00", {}, [], {})
        class NoCuda(Torch):
            class cuda(Torch.cuda):
                @staticmethod
                def is_available(): return False
        def writer(path, payload, **kwargs): Path(path).write_bytes(payload); return {"status": READY}
        result = run_evaluation(EvaluationArgs(root, checkpoint, sidecar, metrics, final, identity_path, dataset, root / "teacher", root / "imagenette", "eval", root / "leases"), lease_factory=Lease, torch_module=NoCuda, admit=lambda **kwargs: proof, writer=writer, lease_event_loader=lambda *_: lease_events_for("exception"))
        assert result["status"] == "STOPPED_INCOMPLETE" and result["lease_outcome"] == "exception" and Path(result["artifact"]).is_file()


def test_admission_failure_and_bad_lease_path_write_stopped_evidence_before_gpu():
    with TemporaryDirectory() as tmp:
        root = Path(tmp); dataset, identity, _ = inputs(root); checkpoint, sidecar, metrics, final, identity_path = training_artifacts(root, identity)
        proof = PreflightProof("eval", {"artifact": str(root)}, "x", "2000-01-01T00:00:00+00:00", {}, [], {})
        def writer(path, payload, **kwargs): Path(path).write_bytes(payload); return {"status": READY}
        args = EvaluationArgs(root, checkpoint, sidecar, metrics, final, identity_path, dataset, root / "teacher", root / "imagenette", "eval", root.parent / "outside")
        result = run_evaluation(args, torch_module=Torch, admit=lambda **kwargs: proof, writer=writer)
        assert result["status"] == "STOPPED_INCOMPLETE" and result["lease_outcome"] == "not_acquired" and Path(result["artifact"]).is_file()


@pytest.mark.parametrize("events", [
    [{"state": "acquired", "run_id": "eval", "command": COMMAND, "timestamp": "2026-01-02T00:00:00+00:00"}, {"state": "released", "run_id": "eval", "command": COMMAND, "timestamp": "2026-01-01T00:00:00+00:00", "outcome": "normal"}],
    [{"state": "acquired", "run_id": "eval", "command": COMMAND}, {"state": "released", "run_id": "eval", "command": COMMAND, "timestamp": "2026-01-01T00:00:00+00:00", "outcome": "normal"}],
])
def test_lease_record_rejects_reversed_or_missing_timestamps(events):
    with pytest.raises(ValueError): _lease_record(events, "eval", "normal")


def test_lease_record_ignores_other_command_for_same_run():
    e1 = lease_events_for()
    e2 = [{**row, "command": "g002-e2-tiled-validation-raw-maps"} for row in lease_events_for()]
    assert _lease_record([*e1, *e2], "eval", "normal")[-1]["outcome"] == "normal"


def test_lease_record_binds_current_pid_when_retries_share_run_and_command():
    earlier = [{**row, "pid": 11} for row in lease_events_for()]
    current = [{**row, "pid": 22} for row in lease_events_for()]
    assert _lease_record([*earlier, *current], "eval", "normal", expected_pid=22)[-1]["outcome"] == "normal"
    with pytest.raises(ValueError): _lease_record([*earlier, *current], "eval", "normal", expected_pid=33)
