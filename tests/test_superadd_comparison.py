import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fine_defect_ad import superadd_comparison as subject


def _canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
class _Proof:
    def __init__(self, root): self.roots = {"artifact": str(root)}
def _writer(path, payload, **_):
    path.write_bytes(payload); return {"status": "READY"}

def _args(root):
    return subject.ComparisonArgs(root, root/"preflight", root/"weight", root/"identity", root/"dataset", root/"lease", root/"source", root/"evaluator", "r1")

def test_contract_counts_and_tie_auroc():
    assert (subject.TRAIN_COUNT, subject.VALIDATION_COUNT, subject.GOOD_COUNT, subject.BAD_COUNT) == (137, 19, 24, 90)
    assert subject.tie_aware_auroc(({"label": "good", "max": 1}, {"label": "bad", "max": 1})) == .5

def test_latch_blocks_second_attempt_and_supports_read_only_recovery(tmp_path):
    args = _args(tmp_path); args.evaluator.write_bytes(b"official")
    binding = {"preflight_sha256": "a"*64, "recipe_sha256": "b"*64, "weight_sha256": "c"*64, "train_bank_sha256": "d"*64}
    validation = {"maps": [], "count": 19}
    first = subject.persist_initial_attempt_latch(args, binding, validation, admit=lambda **_: _Proof(tmp_path), writer=_writer)
    again = subject.persist_initial_attempt_latch(args, binding, validation, admit=lambda **_: _Proof(tmp_path), writer=_writer)
    assert not first["recovery"] and again["recovery"]
    with pytest.raises(subject.ChallengerBlocked):
        subject.persist_initial_attempt_latch(args, binding, {"count": 20}, admit=lambda **_: _Proof(tmp_path), writer=_writer)

def test_raw_map_hash_shape_and_path_free_manifest(tmp_path):
    args = _args(tmp_path)
    raw = np.array([[1, 2], [3, 4]], dtype="<f4").tobytes()
    rows = [{"id_sha256": "a"*64, "label": "good", "source_sha256": "b"*64, "mask_sha256": None, "map_sha256": hashlib.sha256(raw).hexdigest(), "dtype": "<f4", "shape": [2, 2], "byte_order": "<", "_bytes": raw}]
    result = subject._write_maps(tmp_path, args, rows, {"train_bank_sha256": "c"*64}, admit=lambda **_: _Proof(tmp_path), writer=_writer)
    body = result["manifest"].read_text()
    assert rows[0]["map_sha256"] in body and str(tmp_path) not in body and "/" not in body
    assert (tmp_path / f"{subject.MAP_PREFIX}-000-{rows[0]['map_sha256']}.bin").read_bytes() == raw

def test_parity_fp16_diagnostic_shape_and_finite():
    result = subject.parity_summary(np.ones((2, 2)), np.ones((2, 2), dtype=np.float16))
    assert result["shape"] == [2, 2] and result["finite"] and result["max_abs"] == 0

def test_no_host_path_or_url_in_public_record():
    assert subject._path_free({"id_sha256": "a"*64})
    assert not subject._path_free({"path": "/private/image.png"})
    assert not subject._path_free({"url": "https://signed.example/x"})

def test_latch_precedes_test_enumerator_in_runner_source():
    source = Path(subject.__file__).read_text()
    assert source.index("persist_initial_attempt_latch") < source.index("entries = entries_fn")

def test_preflight_recipe_normalizes_json_layers_and_seed_is_recipe_bound(monkeypatch, tmp_path):
    # JSON turns Recipe.layers into a list; admission compares canonical JSON, not Python tuple identity.
    recipe = {**subject.asdict(subject.PINNED_VITS), "fp16_status": "NOT_ADMITTED_PENDING_SAME_VARIANT_FP32_FP16_PARITY"}
    recipe["layers"] = list(recipe["layers"])
    identity = {"data": {"train": [], "validation": []}}
    weight = tmp_path / "weight"; weight.write_bytes(b"w")
    provenance = {"anomalib_commit": subject.ANOMALIB_COMMIT, "dino_commit": "6876159a11b4df116f30f667f8c9888617df0751", "timm_version": "1.0.28", "dino_license_identifier": "DINOv3 License", "dino_license_url_sha256": subject._sha("https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md"), "dino_license_acceptance": "accepted", "dino_license_content_sha256": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e", "exact_hplus_weight_sha256": "e"*64, "exact_hplus_download_url_sha256": "f"*64, "pinned_vits_weight_sha256": hashlib.sha256(b"w").hexdigest(), "pinned_vits_download_url_sha256": "a"*64}
    value = {"status": "READY", "workflow_status": "PINNED_VITS_ADMITTED", "comparison_status": "INCOMPLETE", "recipe": recipe, "training_identity_sha256": subject._sha(identity), "provenance": provenance}
    raw = _canon(value); path = tmp_path / f"superadd-preflight-r1-{hashlib.sha256(raw).hexdigest()}.json"; path.write_bytes(raw)
    binding = subject.validate_preflight(path, tmp_path, weight=weight, training_identity=identity)
    assert binding["coreset_seed"] == str(int(binding["recipe_sha256"][:8], 16) & 0x7fffffff)

def test_bank_is_inference_only_then_model_enters_eval_for_maps():
    source = Path(subject.__file__).read_text()
    assert source.rindex("with torch.inference_mode():", 0, source.index("model.subsample_embedding()")) < source.index("model.subsample_embedding()") < source.index("model.eval()")

def test_sheet_metal_storage_bound_and_geometry_contract():
    assert subject.MAP_SHAPE == (1056, 4224)
    assert subject.MAX_TEST_MAP_BYTES == 2 * 114 * 1056 * 4224 * 4
    assert subject.MAX_TEST_MAP_BYTES < 5 * 1024**3

def test_progress_rows_without_bytes_finalize_manifest(tmp_path):
    args = _args(tmp_path)
    raw = np.zeros(subject.MAP_SHAPE, dtype="<f4").tobytes(); digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / f"{subject.MAP_PREFIX}-000-{digest}.bin").write_bytes(raw)
    rows = [{"index": 0, "id_sha256": "a"*64, "label": "good", "source_sha256": "b"*64, "mask_sha256": None, "map_sha256": digest, "dtype": "<f4", "shape": list(subject.MAP_SHAPE), "byte_order": "<", "latency_seconds": .1}]
    result = subject._write_maps(tmp_path, args, rows, {"x": "y"}, admit=lambda **_: _Proof(tmp_path), writer=_writer)
    assert result["raw_bytes"] == subject.MAP_BYTES and result["manifest"].is_file()

def test_checkpoint_envelope_adoption_is_latch_bound(tmp_path):
    args = _args(tmp_path); raw = np.zeros(subject.MAP_SHAPE, dtype="<f4").tobytes(); digest = hashlib.sha256(raw).hexdigest()
    entry = {"image_identity": "test_public/good/a.png", "label": "good", "source_sha256": "a"*64, "mask_sha256": None}
    binding = {"preflight_sha256":"p"*64}; latch = "l"*64
    row = {"id_sha256": subject._anon(entry["image_identity"]), "label":"good", "source_sha256":"a"*64, "mask_sha256":None, "map_sha256":digest, "latency_seconds":.1, "_bytes":raw}
    subject._write_envelope(tmp_path,args,0,row,latch,binding, admit=lambda **_: _Proof(tmp_path), writer=_writer)
    adopted=subject._adopt_orphan(tmp_path,args,0,entry,latch,binding, admit=lambda **_: _Proof(tmp_path), writer=_writer)
    assert adopted["recovery"] == "CHECKPOINT_ENVELOPE_ADOPTED_NO_SOURCE_DECODE"

def test_checkpoint_envelope_filename_digest_and_raw_conflict_fail_closed(tmp_path):
    args = _args(tmp_path); raw = np.zeros(subject.MAP_SHAPE, dtype="<f4").tobytes(); digest = hashlib.sha256(raw).hexdigest()
    entry = {"image_identity":"test_public/good/a.png","label":"good","source_sha256":"a"*64,"mask_sha256":None}; binding={"preflight_sha256":"p"*64}; latch="l"*64
    row={"id_sha256":subject._anon(entry["image_identity"]),"label":"good","source_sha256":"a"*64,"mask_sha256":None,"map_sha256":digest,"latency_seconds":.1,"_bytes":raw}
    subject._write_envelope(tmp_path,args,0,row,latch,binding,admit=lambda **_:_Proof(tmp_path),writer=_writer)
    path=next(tmp_path.glob(f"{subject.ENVELOPE_PREFIX}-*")); path.write_bytes(path.read_bytes()[:-1]+b"x")
    with pytest.raises(subject.ChallengerBlocked): subject._adopt_orphan(tmp_path,args,0,entry,latch,binding,admit=lambda **_:_Proof(tmp_path),writer=_writer)

def test_cublas_workspace_config_sets_unset_and_blocks_conflict(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert subject._cublas_workspace_config() == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(subject.ChallengerBlocked): subject._cublas_workspace_config()

@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3), (1, 1, 2, 3)])
def test_superadd_map_normalizes_admitted_shapes(shape):
    torch = pytest.importorskip("torch")
    class Model:
        def __call__(self, _): return {"anomaly_map": torch.ones(shape)}
    result = subject._map(Model(), torch.ones((1, 3, 2, 3)), torch, (4, 5))
    assert result.shape == (4, 5) and np.isfinite(result).all()

def test_superadd_map_rejects_non_singleton_batch_or_channel():
    torch = pytest.importorskip("torch")
    class Model:
        def __call__(self, _): return {"anomaly_map": torch.ones((2, 1, 2, 3))}
    with pytest.raises(subject.ChallengerBlocked): subject._map(Model(), torch.ones((1, 3, 2, 3)), torch, (4, 5))
