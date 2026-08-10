"""One-shot, evidence-only comparison for the admitted pinned SuperADD ViT-S recipe."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
import subprocess
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .evaluation_history import _image_auroc
from .g002_eval_runtime import load_training_identity
from .g002_testpub_runtime import BAD_COUNT, GOOD_COUNT, test_public_entries
from .gpu_lock import GpuLease
from .mvtec_aupro import local_au_pro_0_05
from .g002_pilot import LEASE_WRITE_BYTES
from .pilot import host_rss_bytes
from .storage import Allocation, READY, atomic_write, preflight
from .superadd_hplus_probe import _load_pinned_superadd, _verify_anomalib_source
from .superadd_preflight import (
    ANOMALIB_COMMIT,
    DINO_COMMIT,
    DINO_LICENSE_CONTENT_SHA256,
    DINO_LICENSE_IDENTIFIER,
    DINO_LICENSE_URL,
    TIMM_VERSION,
    PINNED_VITS,
    ChallengerBlocked,
    _canonical,
    _dataset_category,
    _inside,
    _sha,
    _sha256,
)

COMMAND = "superadd-pinned-vits-evidence-comparison"
TRAIN_COUNT, VALIDATION_COUNT = 137, 19
MAP_PREFIX = "superadd-vits-test-public-raw"
ENVELOPE_PREFIX = "superadd-vits-test-public-checkpoint"
LATCH_PREFIX = "superadd-vits-test-public-latch"


@dataclass(frozen=True)
class ComparisonArgs:
    artifact_root: Path
    preflight: Path
    weight: Path
    training_identity: Path
    dataset_root: Path
    lease_directory: Path
    anomalib_source: Path
    evaluator: Path
    run_id: str


def _hash(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _anon(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _source_sha256() -> str:
    return _sha256(Path(__file__))


def _git_head() -> str:
    return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()


def _path_free(value: Any) -> Any:
    """Reject host paths and URLs in durable/public evidence."""
    if isinstance(value, Mapping):
        return all(_path_free(k) and _path_free(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_path_free(v) for v in value)
    if isinstance(value, str):
        return not ("/" in value or "\\" in value or "://" in value)
    return True


def _read_canonical(path: Path, root: Path, prefix: str) -> tuple[dict[str, Any], str]:
    path, root = Path(path).resolve(), Path(root).resolve()
    if path.parent != root or not path.name.startswith(prefix) or not path.name.endswith(".json"):
        raise ChallengerBlocked("immutable artifact must be directly under artifact root")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChallengerBlocked("immutable artifact is invalid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ChallengerBlocked("immutable artifact must have canonical JSON bytes")
    digest = sha256(raw).hexdigest()
    if digest not in path.name:
        raise ChallengerBlocked("immutable artifact filename/content hash mismatch")
    return value, digest


def validate_preflight(path: Path, root: Path, *, weight: Path, training_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Admit only the exact incomplete, pinned-ViT-S preflight artifact."""
    value, digest = _read_canonical(path, root, "superadd-preflight-")
    if value.get("status") != READY or value.get("workflow_status") != "PINNED_VITS_ADMITTED" or value.get("comparison_status") != "INCOMPLETE":
        raise ChallengerBlocked("preflight is not the admitted incomplete pinned-ViT-S artifact")
    expected_recipe = {**asdict(PINNED_VITS), "fp16_status": "NOT_ADMITTED_PENDING_SAME_VARIANT_FP32_FP16_PARITY"}
    if _canonical(value.get("recipe")) != _canonical(expected_recipe):
        raise ChallengerBlocked("preflight recipe differs from pinned ViT-S")
    if value.get("training_identity_sha256") != _sha(training_identity):
        raise ChallengerBlocked("preflight training identity mismatch")
    provenance = value.get("provenance")
    expected_keys = {"anomalib_commit", "dino_commit", "timm_version", "dino_license_identifier", "dino_license_url_sha256", "dino_license_acceptance", "dino_license_content_sha256", "exact_hplus_weight_sha256", "exact_hplus_download_url_sha256", "pinned_vits_weight_sha256", "pinned_vits_download_url_sha256"}
    if not isinstance(provenance, Mapping) or set(provenance) != expected_keys:
        raise ChallengerBlocked("preflight provenance record differs from admitted schema")
    if (provenance.get("anomalib_commit") != ANOMALIB_COMMIT or provenance.get("dino_commit") != DINO_COMMIT or provenance.get("timm_version") != TIMM_VERSION or provenance.get("dino_license_identifier") != DINO_LICENSE_IDENTIFIER or provenance.get("dino_license_url_sha256") != _sha(DINO_LICENSE_URL) or provenance.get("dino_license_content_sha256") != DINO_LICENSE_CONTENT_SHA256 or provenance.get("dino_license_acceptance") != "accepted"):
        raise ChallengerBlocked("preflight pinned commit/license binding mismatch")
    local = Path(weight).resolve()
    if not local.is_file() or not _inside(local, root) or _hash(local) != provenance.get("pinned_vits_weight_sha256"):
        raise ChallengerBlocked("local pinned ViT-S weight mismatch")
    recipe_sha256 = _sha(asdict(PINNED_VITS))
    return {"preflight_sha256": digest, "weight_sha256": str(provenance["pinned_vits_weight_sha256"]), "recipe_sha256": recipe_sha256, "coreset_seed": str(int(recipe_sha256[:8], 16) & 0x7fffffff), "coreset_seed_derivation": "int(recipe_sha256[:8],16)&0x7fffffff"}


def freeze_training_identity(identity: Mapping[str, Any], dataset_root: Path) -> dict[str, tuple[dict[str, str], ...]]:
    """Verify and freeze every admitted normal identity before model construction."""
    data = identity.get("data") if isinstance(identity, Mapping) else None
    train, validation = (data.get("train"), data.get("validation")) if isinstance(data, Mapping) else (None, None)
    if not isinstance(train, list) or not isinstance(validation, list) or len(train) != TRAIN_COUNT or len(validation) != VALIDATION_COUNT:
        raise ChallengerBlocked("training identity must contain exactly 137 train and 19 validation normals")
    category = _dataset_category(dataset_root)
    frozen: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    for split, rows in (("train", train), ("validation", validation)):
        seen = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"path", "sha256"} or not isinstance(row["path"], str) or not isinstance(row["sha256"], str):
                raise ChallengerBlocked("training identity entry invalid")
            relative, digest = row["path"], row["sha256"]
            parts = Path(relative).parts
            if Path(relative).is_absolute() or ".." in parts or len(parts) < 3 or parts[:2] != (split, "good") or relative in seen:
                raise ChallengerBlocked("training identity is not a unique good-only identity")
            source = (category / relative).resolve()
            if not source.is_file() or not _inside(source, category) or _sha256(source) != digest:
                raise ChallengerBlocked("training identity bytes changed")
            seen.add(relative)
            frozen[split].append({"path": relative, "sha256": digest})
    return {key: tuple(value) for key, value in frozen.items()}


def tie_aware_auroc(rows: Iterable[Mapping[str, Any]]) -> float:
    stats = {str(index): {"label": row.get("label"), "max": row.get("max")} for index, row in enumerate(rows)}
    return _image_auroc(stats, normal_label="good", anomaly_label="bad")


def parity_summary(fp32: Any, fp16: Any) -> dict[str, Any]:
    import numpy as np
    a, b = np.asarray(fp32, dtype=np.float32), np.asarray(fp16, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("FP32/FP16 parity maps must be finite aligned 2D maps")
    delta = np.abs(a - b).astype(np.float64)
    scale = np.maximum(np.abs(a.astype(np.float64)), np.finfo(np.float32).tiny)
    return {"shape": list(a.shape), "finite": True, "max_abs": float(delta.max()), "p99": float(np.quantile(delta, .99)), "p99_9": float(np.quantile(delta, .999)), "max_rel": float((delta / scale).max())}


def _latch_payload(args: ComparisonArgs, binding: Mapping[str, str], validation: Mapping[str, Any]) -> bytes:
    record = {"status": "TEST_PUBLIC_INITIAL_ATTEMPT_LATCH", "run_id": args.run_id, "command": COMMAND,
              "preflight_sha256": binding["preflight_sha256"], "recipe_sha256": binding["recipe_sha256"],
              "weight_sha256": binding["weight_sha256"], "train_bank_sha256": binding["train_bank_sha256"],
              "validation_parity_sha256": _sha(validation), "evaluator_sha256": _hash(args.evaluator),
              "runner_source_sha256": _source_sha256(), "code_git_commit": _git_head()}
    return _canonical(record)


def persist_initial_attempt_latch(args: ComparisonArgs, binding: Mapping[str, str], validation: Mapping[str, Any], *, admit: Callable[..., Any] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write) -> dict[str, Any]:
    """Durably consume the single TESTpub attempt before any test enumeration/decode."""
    root = Path(args.artifact_root).resolve()
    existing = list(root.glob(f"{LATCH_PREFIX}-*.json"))
    payload = _latch_payload(args, binding, validation); digest = sha256(payload).hexdigest()
    target = root / f"{LATCH_PREFIX}-{args.run_id}-{digest}.json"
    if existing:
        if len(existing) == 1 and existing[0] == target and existing[0].read_bytes() == payload:
            return {"path": target, "sha256": digest, "recovery": True}
        raise ChallengerBlocked("TESTpub initial attempt already consumed")
    source = f"exact initial TESTpub latch bytes={len(payload)}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", len(payload), "persistent", source, "superadd-test-latch"), Allocation("artifact", len(payload), "transient", source, "superadd-test-latch-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root:
        raise ChallengerBlocked("artifact root changed")
    outcome = writer(target, payload, proof=proof, run_id=args.run_id, overwrite=False)
    if outcome.get("status") != READY or not target.is_file() or target.read_bytes() != payload:
        raise ChallengerBlocked("initial TESTpub latch write failed")
    return {"path": target, "sha256": digest, "recovery": False}


# Conservative published runner bound: 114 maps at 16384x16384 float32.  It is
# intentionally larger than any admitted sheet-metal image, so TEST files need not
# be opened to prove capacity before the one-shot latch.
MAP_SHAPE = (1056, 4224)  # exact full geometry: 2 * admitted SPLIT_TARGET_SHAPE
MAP_BYTES = MAP_SHAPE[0] * MAP_SHAPE[1] * 4
MAX_TEST_MAP_BYTES = (GOOD_COUNT + BAD_COUNT) * MAP_BYTES * 2  # envelope + final raw <f4 coexist
EVIDENCE_MARGIN_BYTES = 4 * 1024 * 1024


def _pre_admit_window(root: Path, args: ComparisonArgs, *, admit: Callable[..., Any]) -> None:
    source = f"source-backed sheet-metal geometry: 2*(528,2112)={MAP_SHAPE}; {GOOD_COUNT + BAD_COUNT}*{MAP_SHAPE[0]}*{MAP_SHAPE[1]}*4={MAX_TEST_MAP_BYTES}; evidence_margin={EVIDENCE_MARGIN_BYTES}; lease={LEASE_WRITE_BYTES}"
    pending = max(MAP_BYTES + 4096, LEASE_WRITE_BYTES, EVIDENCE_MARGIN_BYTES)
    proof = admit(run_id=args.run_id, allocations=[
        Allocation("artifact", MAX_TEST_MAP_BYTES + EVIDENCE_MARGIN_BYTES, "persistent", source, "superadd-test-map-upper-bound"),
        Allocation("artifact", LEASE_WRITE_BYTES, "persistent", source, "superadd-gpu-lease"),
        Allocation("artifact", pending, "transient", source, "superadd-window-incoming"),
    ], reserve_bytes=pending, reserve_evidence={"max_pending_atomic_write_bytes": pending, "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ChallengerBlocked("artifact root changed before latch")


def _cublas_workspace_config() -> str:
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value is None:
        value = ":4096:8"; os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
    if value != ":4096:8": raise ChallengerBlocked("CUBLAS_WORKSPACE_CONFIG conflicts with deterministic CUDA")
    return value


def deterministic_settings(torch: Any, seed: int) -> dict[str, Any]:
    """Fail closed when the installed torch cannot request deterministic kernels."""
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as exc:
        raise ChallengerBlocked("deterministic torch/CUDA settings unavailable") from exc
    return {"seed": seed, "torch_deterministic_algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True, "cublas_workspace_config": _cublas_workspace_config()}


def _validate_recovery_manifest(root: Path, args: ComparisonArgs, binding: Mapping[str, str], latch: Mapping[str, Any], *, entries_fn: Callable[[Path], list[dict[str, Any]]]) -> tuple[str, Mapping[str, Any], list[Mapping[str, Any]], list[Any], list[Any], list[dict[str, Any]]]:
    manifests = list(root.glob(f"{MAP_PREFIX}-{args.run_id}-*.json"))
    if len(manifests) != 1: raise ChallengerBlocked("TESTpub latch consumed without exactly one recoverable raw manifest")
    manifest, manifest_sha = _read_canonical(manifests[0], root, f"{MAP_PREFIX}-{args.run_id}-")
    rows, lineage = manifest.get("maps"), manifest.get("lineage")
    full_keys = set(binding) | {"train_bank_sha256", "frozen_train_sha256", "frozen_validation_sha256", "evaluator_sha256", "runner_source_sha256", "code_git_commit"}
    if manifest.get("status") != "SUPERADD_TEST_PUBLIC_RAW_MAPS" or not isinstance(lineage, Mapping) or set(lineage) != full_keys or any(lineage.get(key) != value for key, value in binding.items()) or not isinstance(rows, list) or len(rows) != GOOD_COUNT + BAD_COUNT:
        raise ChallengerBlocked("recovery raw-map manifest mismatch")
    if latch.get("train_bank_sha256") != lineage.get("train_bank_sha256") or latch.get("evaluator_sha256") != lineage.get("evaluator_sha256") or latch.get("runner_source_sha256") != lineage.get("runner_source_sha256") or latch.get("code_git_commit") != lineage.get("code_git_commit") or any(not isinstance(lineage.get(key), str) or len(lineage[key]) != 64 for key in ("train_bank_sha256", "frozen_train_sha256", "frozen_validation_sha256", "evaluator_sha256", "runner_source_sha256")):
        raise ChallengerBlocked("recovery lineage/latch mismatch")
    entries = entries_fn(args.dataset_root)
    if len(entries) != len(rows): raise ChallengerBlocked("recovery TESTpub count mismatch")
    import numpy as np
    from PIL import Image
    maps, masks, stats = [], [], []
    for index, (row, entry) in enumerate(zip(rows, entries)):
        if row.get("id_sha256") != _anon(entry["image_identity"]) or row.get("label") != entry["label"] or row.get("source_sha256") != entry["source_sha256"] or row.get("mask_sha256") != entry["mask_sha256"] or row.get("dtype") != "<f4" or row.get("byte_order") != "<": raise ChallengerBlocked("recovery TESTpub identity/order mismatch")
        shape, digest = row.get("shape"), row.get("map_sha256")
        if not isinstance(shape, list) or tuple(shape) != MAP_SHAPE or not isinstance(digest, str): raise ChallengerBlocked("recovery map schema mismatch")
        raw = (root / f"{MAP_PREFIX}-{index:03d}-{digest}.bin").read_bytes()
        if sha256(raw).hexdigest() != digest or len(raw) != MAP_BYTES: raise ChallengerBlocked("recovery map bytes/hash mismatch")
        mapped = np.frombuffer(raw, dtype="<f4").reshape(shape)
        if not np.isfinite(mapped).all(): raise ChallengerBlocked("recovery map non-finite")
        maps.append(mapped); masks.append(None if entry["mask"] is None else np.asarray(Image.open(entry["mask"]))); stats.append({"label": row["label"], "max": float(mapped.max())})
    return manifest_sha, lineage, rows, maps, masks, stats


def _recover_existing(root: Path, args: ComparisonArgs, binding: Mapping[str, str], *, entries_fn: Callable[[Path], list[dict[str, Any]]], evaluator_fn: Callable[..., Any], admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]]) -> dict[str, Any] | None:
    """A prior latch is terminal: only its matching complete final is reusable."""
    latches = list(root.glob(f"{LATCH_PREFIX}-*.json"))
    if not latches: return None
    if len(latches) != 1: raise ChallengerBlocked("multiple TESTpub latches block recovery")
    latch, latch_sha = _read_canonical(latches[0], root, LATCH_PREFIX + "-")
    required = {"status", "run_id", "command", "preflight_sha256", "recipe_sha256", "weight_sha256", "train_bank_sha256", "validation_parity_sha256", "evaluator_sha256", "runner_source_sha256", "code_git_commit"}
    if set(latch) != required or latch.get("status") != "TEST_PUBLIC_INITIAL_ATTEMPT_LATCH" or latch.get("run_id") != args.run_id or latch.get("command") != COMMAND:
        raise ChallengerBlocked("invalid existing TESTpub latch")
    for key in ("preflight_sha256", "recipe_sha256", "weight_sha256"):
        if latch.get(key) != binding.get(key): raise ChallengerBlocked("existing latch provenance mismatch")
    if latch.get("evaluator_sha256") != _hash(args.evaluator) or latch.get("runner_source_sha256") != _source_sha256() or latch.get("code_git_commit") != _git_head():
        raise ChallengerBlocked("existing latch evaluator/code mismatch")
    finals = list(root.glob(f"superadd-vits-comparison-{args.run_id}-*.json"))
    if len(finals) == 1:
        manifest_sha, lineage, _rows, _maps, _masks, _stats = _validate_recovery_manifest(root, args, binding, latch, entries_fn=entries_fn)
        final, _ = _read_canonical(finals[0], root, f"superadd-vits-comparison-{args.run_id}-")
        final_binding = final.get("bindings")
        required_final = {"status", "comparison_status", "selected_precision", "determinism", "fp16", "latch_sha256", "bindings", "counts", "local_au_pro_0_05", "image_auroc_tie_aware", "threshold_metrics", "latency_seconds", "resources", "representatives", "raw_manifest_sha256", "runner_source_sha256", "code_git_commit", "limitations"}
        if not required_final <= set(final) or set(final) - (required_final | {"recovery"}) or final.get("status") != READY or final.get("raw_manifest_sha256") != manifest_sha or final.get("latch_sha256") != latch_sha or final_binding != lineage or not isinstance(final.get("local_au_pro_0_05"), Mapping) or not isinstance(final.get("image_auroc_tie_aware"), (int, float)) or not _path_free(final):
            raise ChallengerBlocked("existing final evidence does not bind complete maps/metrics/privacy")
        return {**final, "recovery": "READ_ONLY_COMPLETE_FINAL"}
    if len(finals) > 1: raise ChallengerBlocked("multiple final evidence artifacts block recovery")
    # A latch with no complete manifest is resumable after deterministic bank rebuild.
    if not list(root.glob(f"{MAP_PREFIX}-{args.run_id}-*.json")): return None
    manifest_sha, lineage, _rows, maps, masks, stats = _validate_recovery_manifest(root, args, binding, latch, entries_fn=entries_fn)
    record = {"status": READY, "comparison_status": "COMPLETE_EVIDENCE_ONLY", "selected_precision": "fp32", "determinism": {"recovery": "PERSISTED_RAW_MAPS_NO_KERNEL_REEXECUTION"}, "fp16": {"status": "PERSISTED_ORIGINAL_NOT_RECOMPUTED"}, "recovery": "RAW_MAPS_AND_GT_MASKS_ONLY_NO_SOURCE_IMAGE_DECODE", "latch_sha256": latch_sha, "bindings": dict(lineage), "counts": {"train_good": TRAIN_COUNT, "validation_good": VALIDATION_COUNT, "test_good": GOOD_COUNT, "test_bad": BAD_COUNT, "test_total": GOOD_COUNT + BAD_COUNT}, "raw_manifest_sha256": manifest_sha, "local_au_pro_0_05": evaluator_fn(maps, masks, args.evaluator, include_curve=False), "image_auroc_tie_aware": tie_aware_auroc(stats), "threshold_metrics": "NONE", "latency_seconds": {"status": "PERSISTED_ORIGINAL_NOT_AVAILABLE_IN_RAW_MAP_MANIFEST"}, "resources": {"status": "PERSISTED_ORIGINAL_NOT_AVAILABLE_IN_RAW_MAP_MANIFEST"}, "representatives": {"status": "PERSISTED_ORIGINAL_NOT_AVAILABLE_IN_RAW_MAP_MANIFEST"}, "runner_source_sha256": _source_sha256(), "code_git_commit": _git_head(), "limitations": ["Recovered from immutable raw maps; no TEST source image was decoded or inferred.", "FP32 was selected in the original one-shot attempt."]}
    if not _path_free(record): raise ChallengerBlocked("recovery evidence privacy violation")
    payload = _canonical(record); digest = sha256(payload).hexdigest(); source = f"immutable raw-map recovery evidence bytes={len(payload)}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", len(payload), "persistent", source, "superadd-recovery-final"), Allocation("artifact", len(payload), "transient", source, "superadd-recovery-final-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    target = root / f"superadd-vits-comparison-{args.run_id}-{digest}.json"
    if writer(target, payload, proof=proof, run_id=args.run_id, overwrite=False).get("status") != READY or target.read_bytes() != payload: raise ChallengerBlocked("recovery final evidence write failed")
    return {**record, "artifact_sha256": digest}


def _tensor_image(path: Path, torch: Any, *, device: Any) -> tuple[Any, tuple[int, int]]:
    import numpy as np
    from PIL import Image, ImageOps
    image = Image.open(path).convert("RGB"); original = (image.height, image.width)
    resized = image.resize((max(1, round(image.width * .625)), max(1, round(image.height * .625))))
    width, height = max(PINNED_VITS.input_size, ((resized.width + 15) // 16) * 16), max(PINNED_VITS.input_size, ((resized.height + 15) // 16) * 16)
    padded = ImageOps.expand(resized, border=(0, 0, width - resized.width, height - resized.height), fill=0)
    value = torch.from_numpy(np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / 255.).unsqueeze(0).to(device)
    mean = torch.tensor((.485, .456, .406), device=device).view(1, 3, 1, 1); std = torch.tensor((.229, .224, .225), device=device).view(1, 3, 1, 1)
    return (value - mean) / std, original


def _build_model(source: Path, weight: Path, torch: Any) -> Any:
    module, timm = _load_pinned_superadd(source); original = timm.create_model
    def local(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = True; kwargs.pop("checkpoint_path", None); kwargs["pretrained_cfg_overlay"] = {"file": str(weight)}
        return original(*args, **kwargs)
    module.timm.create_model = local
    try:
        return module.SuperADDModel(backbone=PINNED_VITS.backbone, layers=list(PINNED_VITS.layers), patch_size=PINNED_VITS.input_size, patch_overlap=PINNED_VITS.patch_overlap, max_database_size=PINNED_VITS.bank_per_layer, subsampling_iterations=PINNED_VITS.subsampling_iterations).cuda().train()
    finally:
        module.timm.create_model = original


def _map(model: Any, value: Any, torch: Any, original: tuple[int, int], *, fp16: bool = False) -> Any:
    import torch.nn.functional as F
    with torch.inference_mode():
        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16): output = model(value)
        else: output = model(value)
    candidate = output["anomaly_map"] if isinstance(output, Mapping) else getattr(output, "anomaly_map", output)
    return F.interpolate(candidate.float(), size=original, mode="bilinear", align_corners=False).squeeze().detach().cpu().numpy()


def _progress_path(root: Path, args: ComparisonArgs, payload: bytes) -> Path:
    return root / f"superadd-vits-test-public-progress-{args.run_id}-{sha256(payload).hexdigest()}.json"


def _load_progress(root: Path, args: ComparisonArgs, binding: Mapping[str, str]) -> list[dict[str, Any]]:
    paths = list(root.glob(f"superadd-vits-test-public-progress-{args.run_id}-*.json"))
    if not paths: return []
    # Content-addressed snapshots are append-only; choose the sole greatest next index.
    records = []
    for path in paths:
        value, _ = _read_canonical(path, root, f"superadd-vits-test-public-progress-{args.run_id}-")
        if value.get("status") != "SUPERADD_TEST_PUBLIC_PROGRESS" or value.get("lineage") != dict(binding) or not isinstance(value.get("next_index"), int) or not isinstance(value.get("rows"), list): raise ChallengerBlocked("progress journal provenance mismatch")
        records.append(value)
    highest = max(item["next_index"] for item in records)
    candidates = [item for item in records if item["next_index"] == highest]
    if len(candidates) != 1: raise ChallengerBlocked("conflicting greatest progress snapshots")
    record = candidates[0]
    if record["next_index"] != len(record["rows"]) or record["next_index"] > GOOD_COUNT + BAD_COUNT: raise ChallengerBlocked("progress journal order mismatch")
    rows = record["rows"]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("index") != index or not isinstance(row.get("map_sha256"), str) or row.get("shape") != list(MAP_SHAPE): raise ChallengerBlocked("progress row invalid")
        raw = (root / f"{MAP_PREFIX}-{index:03d}-{row['map_sha256']}.bin").read_bytes()
        if sha256(raw).hexdigest() != row["map_sha256"] or len(raw) != MAP_BYTES: raise ChallengerBlocked("progress raw map mismatch")
    return [dict(row) for row in rows]


def _checkpoint_progress(root: Path, args: ComparisonArgs, binding: Mapping[str, str], rows: list[dict[str, Any]], *, admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]]) -> None:
    public_rows = [{k:v for k,v in row.items() if not k.startswith("_")} for row in rows]
    payload = _canonical({"status":"SUPERADD_TEST_PUBLIC_PROGRESS", "run_id":args.run_id, "lineage":dict(binding), "next_index":len(rows), "rows":public_rows})
    source = f"atomic progress snapshot bytes={len(payload)}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", len(payload), "persistent", source, "superadd-progress"), Allocation("artifact", len(payload), "transient", source, "superadd-progress-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes":len(payload),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
    path = _progress_path(root,args,payload)
    if writer(path,payload,proof=proof,run_id=args.run_id,overwrite=False).get("status") != READY or path.read_bytes()!=payload: raise ChallengerBlocked("progress journal write failed")


def _adopt_orphan(root: Path, args: ComparisonArgs, index: int, entry: Mapping[str, Any], latch_sha256: str, binding: Mapping[str, str], *, admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]]) -> dict[str, Any] | None:
    """Adopt only a self-describing run/latch/lineage-bound checkpoint envelope."""
    paths = list(root.glob(f"{ENVELOPE_PREFIX}-{args.run_id}-{index:03d}-*.bin"))
    if not paths: return None
    if len(paths) != 1: raise ChallengerBlocked("multiple checkpoint envelopes block resume")
    envelope_path = paths[0]; filename_digest = envelope_path.stem.rsplit("-", 1)[-1]
    blob = envelope_path.read_bytes()
    if len(filename_digest) != 64 or sha256(blob).hexdigest() != filename_digest: raise ChallengerBlocked("checkpoint envelope filename/content mismatch")
    if len(blob) < 8: raise ChallengerBlocked("checkpoint envelope truncated")
    size = int.from_bytes(blob[:8], "big")
    try: header = json.loads(blob[8:8+size]); raw = blob[8+size:]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ChallengerBlocked("checkpoint envelope invalid") from exc
    digest = sha256(raw).hexdigest()
    expected = {"run_id":args.run_id,"latch_sha256":latch_sha256,"lineage_sha256":_sha(binding),"index":index,"id_sha256":_anon(entry["image_identity"]),"label":entry["label"],"source_sha256":entry["source_sha256"],"mask_sha256":entry["mask_sha256"],"map_sha256":digest,"dtype":"<f4","shape":list(MAP_SHAPE),"byte_order":"<"}
    if not isinstance(header, Mapping) or any(header.get(k) != v for k,v in expected.items()) or not (isinstance(header.get("latency_seconds"), (int,float)) and __import__("math").isfinite(header["latency_seconds"])) or len(raw) != MAP_BYTES:
        raise ChallengerBlocked("checkpoint envelope provenance mismatch")
    import numpy as np
    if not np.isfinite(np.frombuffer(raw,dtype="<f4").reshape(MAP_SHAPE)).all(): raise ChallengerBlocked("checkpoint envelope non-finite")
    raw_path=root/f"{MAP_PREFIX}-{index:03d}-{digest}.bin"
    source = f"verified envelope raw extraction bytes={len(raw)}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact",len(raw),"persistent",source,"superadd-envelope-raw"),Allocation("artifact",len(raw),"transient",source,"superadd-envelope-raw-incoming")],reserve_bytes=len(raw),reserve_evidence={"max_pending_atomic_write_bytes":len(raw),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
    if raw_path.exists():
        if _hash(raw_path) != digest: raise ChallengerBlocked("checkpoint raw artifact conflict")
    elif writer(raw_path, raw, proof=proof, run_id=args.run_id, overwrite=False).get("status") != READY or _hash(raw_path) != digest: raise ChallengerBlocked("checkpoint raw artifact mismatch")
    return {**expected,"latency_seconds":float(header["latency_seconds"]),"envelope_sha256":sha256(blob).hexdigest(),"recovery":"CHECKPOINT_ENVELOPE_ADOPTED_NO_SOURCE_DECODE"}


def _write_envelope(root: Path, args: ComparisonArgs, index: int, row: Mapping[str, Any], latch_sha256: str, binding: Mapping[str, str], *, admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]]) -> None:
    header={"run_id":args.run_id,"latch_sha256":latch_sha256,"lineage_sha256":_sha(binding),"index":index,"id_sha256":row["id_sha256"],"label":row["label"],"source_sha256":row["source_sha256"],"mask_sha256":row["mask_sha256"],"map_sha256":row["map_sha256"],"dtype":"<f4","shape":list(MAP_SHAPE),"byte_order":"<","latency_seconds":row["latency_seconds"]}
    body=_canonical(header); blob=len(body).to_bytes(8,"big")+body+row["_bytes"]; digest=sha256(blob).hexdigest()
    path=root/f"{ENVELOPE_PREFIX}-{args.run_id}-{index:03d}-{digest}.bin"
    source = f"exact checkpoint envelope bytes={len(blob)}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", len(blob), "persistent", source, "superadd-envelope"), Allocation("artifact", len(blob), "transient", source, "superadd-envelope-incoming")], reserve_bytes=len(blob), reserve_evidence={"max_pending_atomic_write_bytes":len(blob),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
    if writer(path, blob, proof=proof, run_id=args.run_id, overwrite=False).get("status") != READY or sha256(path.read_bytes()).hexdigest() != digest: raise ChallengerBlocked("checkpoint envelope write failed")
    row["envelope_sha256"]=digest


def _write_one_map(root: Path, args: ComparisonArgs, index: int, row: Mapping[str, Any], *, admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]]) -> None:
    path = root / f"{MAP_PREFIX}-{index:03d}-{row['map_sha256']}.bin"
    if path.exists():
        if _hash(path) != row["map_sha256"]: raise ChallengerBlocked("orphan raw map hash mismatch")
        return
    source = f"single exact raw map bytes={len(row['_bytes'])}"
    proof=admit(run_id=args.run_id,allocations=[Allocation("artifact",len(row["_bytes"]),"persistent",source,"superadd-progress-raw"),Allocation("artifact",len(row["_bytes"]),"transient",source,"superadd-progress-raw-incoming")],reserve_bytes=len(row["_bytes"]),reserve_evidence={"max_pending_atomic_write_bytes":len(row["_bytes"]),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
    if writer(path,row["_bytes"],proof=proof,run_id=args.run_id,overwrite=False).get("status") != READY or _hash(path)!=row["map_sha256"]: raise ChallengerBlocked("incremental raw map write failed")


def _write_maps(root: Path, args: ComparisonArgs, rows: list[dict[str, Any]], binding: Mapping[str, str], *, admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
    manifest_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    payload = _canonical({"status": "SUPERADD_TEST_PUBLIC_RAW_MAPS", "run_id": args.run_id, "lineage": dict(binding), "maps": manifest_rows})
    raw_total = len(rows) * MAP_BYTES; pending = max(len(payload), MAP_BYTES)
    source = f"source-backed raw float32 map bytes={raw_total}; max atomic write={pending}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", raw_total + len(payload), "persistent", source, "superadd-test-raw-maps"), Allocation("artifact", pending, "transient", source, "superadd-test-raw-maps-incoming")], reserve_bytes=pending, reserve_evidence={"max_pending_atomic_write_bytes": pending, "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    if Path(proof.roots["artifact"]).resolve() != root: raise ChallengerBlocked("artifact root changed")
    for index, row in enumerate(rows):
        path = root / f"{MAP_PREFIX}-{index:03d}-{row['map_sha256']}.bin"
        if path.exists():
            if _hash(path) != row["map_sha256"]: raise ChallengerBlocked("raw map hash mismatch")
        elif "_bytes" not in row: raise ChallengerBlocked("missing persisted raw map")
        elif writer(path, row["_bytes"], proof=proof, run_id=args.run_id, overwrite=False).get("status") != READY or _hash(path) != row["map_sha256"]: raise ChallengerBlocked("raw map write failed")
    manifest = root / f"{MAP_PREFIX}-{args.run_id}-{sha256(payload).hexdigest()}.json"
    if writer(manifest, payload, proof=proof, run_id=args.run_id, overwrite=False).get("status") != READY or manifest.read_bytes() != payload: raise ChallengerBlocked("raw map manifest write failed")
    return {"manifest": manifest, "raw_bytes": raw_total, "storage_bytes": raw_total + len(payload)}


def run_comparison(args: ComparisonArgs, *, admit: Callable[..., Any] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write, lease_factory: Callable[..., Any] = GpuLease, evaluator_fn: Callable[..., Any] = local_au_pro_0_05, entries_fn: Callable[[Path], list[dict[str, Any]]] = test_public_entries) -> dict[str, Any]:
    """Execute the sole admitted candidate: FP32 selected, FP16 validation diagnostic."""
    import numpy as np
    import torch
    from PIL import Image
    root = Path(args.artifact_root).resolve()
    if Path(args.lease_directory).resolve() == root or not _inside(Path(args.lease_directory), root): raise ChallengerBlocked("lease directory must be under artifact root")
    identity, _ = load_training_identity(args.training_identity, root)
    binding = validate_preflight(args.preflight, root, weight=args.weight, training_identity=identity)
    _pre_admit_window(root, args, admit=admit)
    recovered = _recover_existing(root, args, binding, entries_fn=entries_fn, evaluator_fn=evaluator_fn, admit=admit, writer=writer)
    if recovered is not None: return recovered
    frozen = freeze_training_identity(identity, args.dataset_root)
    _verify_anomalib_source(args.anomalib_source)
    _cublas_workspace_config()  # before any CUDA/CuBLAS handle creation
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
    started, latencies, fp16_error = time.perf_counter(), [], None
    with lease_factory(args.lease_directory, args.run_id, COMMAND):
        torch.cuda.reset_peak_memory_stats(); device = torch.device("cuda:0")
        deterministic = deterministic_settings(torch, int(binding["coreset_seed"]))
        model = _build_model(args.anomalib_source, args.weight, torch)
        category = _dataset_category(args.dataset_root)
        with torch.inference_mode():
            for row in frozen["train"]: model(_tensor_image(category / row["path"], torch, device=device)[0])
        model.subsample_embedding()
        model.eval()
        bank = model.memory_bank.detach().float().cpu().numpy().tobytes()
        binding = {**binding, "train_bank_sha256": sha256(bank).hexdigest(), "frozen_train_sha256": _sha(frozen["train"]), "frozen_validation_sha256": _sha(frozen["validation"]), "evaluator_sha256": _hash(args.evaluator), "runner_source_sha256": _source_sha256(), "code_git_commit": _git_head()}
        validation = []
        for row in frozen["validation"]:
            value, original = _tensor_image(category / row["path"], torch, device=device); fp32 = _map(model, value, torch, original); item = {"id_sha256": _anon(row["path"]), "fp32_map_sha256": sha256(np.asarray(fp32, dtype="<f4").tobytes()).hexdigest()}
            try: item["fp16"] = parity_summary(fp32, _map(model, value, torch, original, fp16=True))
            except Exception as exc: fp16_error = type(exc).__name__; item["fp16"] = {"status": "DIAGNOSTIC_FAILED"}
            validation.append(item)
        validation_record = {"count": VALIDATION_COUNT, "selected_precision": "fp32", "fp16_status": "DIAGNOSTIC_FAILED" if fp16_error else "DIAGNOSTIC_COMPLETE", "maps": validation}
        latch = persist_initial_attempt_latch(args, binding, validation_record, admit=admit, writer=writer)
        # TEST enumeration follows the durable latch. Completed rows are never decoded again.
        entries = entries_fn(args.dataset_root)
        if len(entries) != GOOD_COUNT + BAD_COUNT: raise ChallengerBlocked("exact 114 TESTpub entries required")
        rows = _load_progress(root, args, binding)
        for index, row in enumerate(rows):
            if row.get("id_sha256") != _anon(entries[index]["image_identity"]) or row.get("label") != entries[index]["label"] or row.get("source_sha256") != entries[index]["source_sha256"] or row.get("mask_sha256") != entries[index]["mask_sha256"] or row.get("dtype") != "<f4" or row.get("byte_order") != "<" or (row.get("latency_seconds") is None and row.get("recovery") != "ORPHAN_RAW_ADOPTED_NO_SOURCE_DECODE") or (row.get("latency_seconds") is not None and (not isinstance(row.get("latency_seconds"), (int, float)) or not np.isfinite(row["latency_seconds"]))): raise ChallengerBlocked("progress entry identity/latency mismatch")
        for index, entry in enumerate(entries):
            if index < len(rows): continue
            orphan = _adopt_orphan(root, args, index, entry, latch["sha256"], binding, admit=admit, writer=writer)
            if orphan is not None:
                rows.append(orphan); _checkpoint_progress(root, args, binding, rows, admit=admit, writer=writer); continue
            begin = time.perf_counter(); value, original = _tensor_image(entry["source"], torch, device=device); mapped = np.asarray(_map(model, value, torch, original), dtype="<f4")
            if tuple(mapped.shape) != MAP_SHAPE: raise ChallengerBlocked("TESTpub map geometry differs from admitted sheet-metal contract")
            raw = mapped.tobytes(order="C"); latencies.append(time.perf_counter() - begin)
            rows.append({"index": index, "id_sha256": _anon(entry["image_identity"]), "label": entry["label"], "source_sha256": entry["source_sha256"], "mask_sha256": entry["mask_sha256"], "map_sha256": sha256(raw).hexdigest(), "dtype": "<f4", "shape": list(mapped.shape), "byte_order": "<", "latency_seconds": latencies[-1], "_bytes": raw})
            _write_envelope(root, args, index, rows[-1], latch["sha256"], binding, admit=admit, writer=writer); _write_one_map(root, args, index, rows[-1], admit=admit, writer=writer); rows[-1].pop("_bytes"); _checkpoint_progress(root, args, binding, rows, admit=admit, writer=writer)
        maps = [np.frombuffer((root / f"{MAP_PREFIX}-{i:03d}-{row['map_sha256']}.bin").read_bytes(), dtype="<f4").reshape(MAP_SHAPE) for i,row in enumerate(rows)]
        measured_latencies = [float(row["latency_seconds"]) for row in rows if isinstance(row.get("latency_seconds"), (int, float))]
        if not measured_latencies: raise ChallengerBlocked("no measured TESTpub latencies remain after recovery")
        latencies = measured_latencies
        masks = [None if entry["mask"] is None else np.asarray(Image.open(entry["mask"])) for entry in entries]
        persisted = _write_maps(root, args, rows, binding, admit=admit, writer=writer)
        metric = evaluator_fn(maps, masks, args.evaluator, include_curve=False)
        stats = [{"label": row["label"], "max": float(mapped.max())} for row, mapped in zip(rows, maps)]
        normal = max(((row, float(mapped.max())) for row, mapped in zip(rows, maps) if row["label"] == "good"), key=lambda item: item[1])[0]
        anomaly_rows = [(row, float(mapped.max())) for row, mapped in zip(rows, maps) if row["label"] == "bad"]
        low, high = min(anomaly_rows, key=lambda item: item[1])[0], max(anomaly_rows, key=lambda item: item[1])[0]
        latency = np.asarray(latencies, dtype=np.float64)
        if any(mapped.ndim != 2 or not np.isfinite(mapped).all() for mapped in maps):
            raise ChallengerBlocked("TESTpub raw maps must be finite 2D float32")
        record = {"status": READY, "comparison_status": "COMPLETE_EVIDENCE_ONLY", "selected_precision": "fp32", "determinism": deterministic, "fp16": validation_record, "latch_sha256": latch["sha256"], "bindings": binding, "counts": {"train_good": TRAIN_COUNT, "validation_good": VALIDATION_COUNT, "test_good": GOOD_COUNT, "test_bad": BAD_COUNT, "test_total": GOOD_COUNT + BAD_COUNT}, "local_au_pro_0_05": metric, "image_auroc_tie_aware": tie_aware_auroc(stats), "threshold_metrics": "NONE", "latency_seconds": {"p50": float(np.quantile(latency,.5)), "p95": float(np.quantile(latency,.95)), "p99": float(np.quantile(latency,.99)), "mean": float(latency.mean()), "throughput_images_per_second": float(len(latency)/latency.sum()), "measured_count": len(latency), "missing_count": len(rows) - len(latency)}, "resources": {"cuda_allocated_peak": int(torch.cuda.max_memory_allocated()), "cuda_reserved_peak": int(torch.cuda.max_memory_reserved()), "host_rss_bytes": host_rss_bytes(), "bank_bytes": len(bank), "raw_artifact_bytes": persisted["raw_bytes"], "storage_bytes": persisted["storage_bytes"]}, "representatives": {"highest_normal_id_sha256": normal["id_sha256"], "lowest_anomaly_id_sha256": low["id_sha256"], "highest_anomaly_id_sha256": high["id_sha256"]}, "raw_manifest_sha256": _hash(persisted["manifest"]), "runner_source_sha256": _source_sha256(), "code_git_commit": _git_head(), "limitations": ["Pinned anomalib SuperADD hardware-constrained ViT-S evidence candidate; not reproduction-equivalent to SuperAD 2505 or official SuperADD 2605 H+ paper result.", "Deterministic one-pass normal bank; no brightness sweep, morphology, binary-threshold metric, or test tuning.", "Continuous tie-aware Image AUROC and local AU-PRO only; FP32 remains selected."]}
    if not _path_free(record): raise ChallengerBlocked("durable comparison evidence is not path-free")
    payload = _canonical(record); digest = sha256(payload).hexdigest()
    source = f"immutable comparison evidence bytes={len(payload)}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", len(payload), "persistent", source, "superadd-final-evidence"), Allocation("artifact", len(payload), "transient", source, "superadd-final-evidence-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    target = root / f"superadd-vits-comparison-{args.run_id}-{digest}.json"
    if writer(target, payload, proof=proof, run_id=args.run_id, overwrite=False).get("status") != READY or target.read_bytes() != payload:
        raise ChallengerBlocked("final comparison evidence write failed")
    return {**record, "artifact_sha256": digest}


def parse_args(argv: Sequence[str] | None = None) -> ComparisonArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "preflight", "weight", "training-identity", "dataset-root", "lease-directory", "anomalib-source", "evaluator"): parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return ComparisonArgs(**{key.replace("-", "_"): value for key, value in vars(parser.parse_args(argv)).items()})


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_comparison(parse_args(argv)); print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0 if result["status"] == READY else 2
    except Exception as exc:
        print(json.dumps({"status": "STOPPED_INCOMPLETE", "comparison_status": "INCOMPLETE", "exception_type": type(exc).__name__, "exception_fingerprint": sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()[:16]}, sort_keys=True)); return 2


if __name__ == "__main__": raise SystemExit(main())
