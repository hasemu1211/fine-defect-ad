"""Produce the canonical, resource-only exact-H+ SuperADD preflight artifact."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from .g002_eval_runtime import load_training_identity
from .gpu_lock import GpuLease
from .storage import READY, StorageBlocked, atomic_write
from .superadd_preflight import (
    ANOMALIB_COMMIT,
    EXACT_HPLUS,
    PROBE_PRODUCER_MODULE,
    ChallengerBlocked,
    _admit_storage,
    _canonical,
    _dataset_category,
    _sha,
    _sha256,
    validate_fixture,
    validate_provenance,
    verify_local_weights,
)


class ReproductionFailure(RuntimeError):
    """The pinned source or runtime cannot reproduce the exact-H+ probe."""


class ResourceFailure(RuntimeError):
    """The exact-H+ probe exhausted a required resource."""


def _source_sha256() -> str:
    return _sha256(Path(__file__))


def _rss_bytes() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _verify_anomalib_source(source: Path) -> None:
    if _git_head(source) != ANOMALIB_COMMIT:
        raise ReproductionFailure("ANOMALIB_SOURCE_REVISION_MISMATCH")


def _runtime_binding(source: Path) -> str:
    import torch

    return _sha(
        {
            "anomalib_commit": _git_head(source),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "producer_source_sha256": _source_sha256(),
        }
    )


def _load_pinned_superadd(source: Path):
    """Load SuperADD only from the verified anomalib checkout, never ambient packages."""
    import importlib
    import sys

    source_root = Path(source).resolve() / "src"
    if not source_root.is_dir():
        raise ReproductionFailure("VERIFIED_SOURCE_INCOMPATIBLE")

    sys.path.insert(0, str(source_root))
    module = importlib.import_module("anomalib.models.image.super_add.torch_model")
    if not Path(module.__file__).resolve().is_relative_to(source_root):
        raise ReproductionFailure("VERIFIED_SOURCE_INCOMPATIBLE")

    import timm

    if getattr(timm, "__version__", None) != "1.0.28":
        raise ReproductionFailure("VERIFIED_SOURCE_INCOMPATIBLE")
    return module, timm


def _live_step(
    *,
    dataset_root: Path,
    fixture: Mapping[str, Any],
    weight_path: Path,
    anomalib_source: Path,
) -> dict[str, Any]:
    """Run a pinned H+ forward/index smoke over frozen normals without metrics."""
    import numpy as np
    import torch

    _verify_anomalib_source(anomalib_source)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")

    from PIL import Image, ImageOps

    torch_model, timm = _load_pinned_superadd(anomalib_source)
    original_create = timm.create_model

    def local_weight_create(*args, **kwargs):
        kwargs["pretrained"] = True
        kwargs.pop("checkpoint_path", None)
        kwargs["pretrained_cfg_overlay"] = {"file": str(weight_path)}
        return original_create(*args, **kwargs)

    torch_model.timm.create_model = local_weight_create
    try:
        try:
            model = torch_model.SuperADDModel(
                backbone=EXACT_HPLUS.backbone,
                layers=list(EXACT_HPLUS.layers),
                patch_size=EXACT_HPLUS.input_size,
                patch_overlap=EXACT_HPLUS.patch_overlap,
                max_database_size=EXACT_HPLUS.bank_per_layer,
                subsampling_iterations=EXACT_HPLUS.subsampling_iterations,
            ).cuda().train()
        except torch.cuda.OutOfMemoryError as exc:
            raise ResourceFailure("CUDA_OOM") from exc
    finally:
        torch_model.timm.create_model = original_create

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for entry in fixture["entries"]:
            image = Image.open(_dataset_category(dataset_root) / entry["path"]).convert(
                "RGB"
            )
            resized = image.resize(
                (
                    max(1, round(image.width * 0.625)),
                    max(1, round(image.height * 0.625)),
                )
            )
            width = max(EXACT_HPLUS.input_size, ((resized.width + 15) // 16) * 16)
            height = max(EXACT_HPLUS.input_size, ((resized.height + 15) // 16) * 16)
            padded = ImageOps.expand(
                resized,
                border=(0, 0, width - resized.width, height - resized.height),
                fill=0,
            )
            value = (
                torch.from_numpy(
                    np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / 255.0
                )
                .unsqueeze(0)
                .cuda()
            )
            mean = torch.tensor((0.485, 0.456, 0.406), device=value.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor((0.229, 0.224, 0.225), device=value.device).view(
                1, 3, 1, 1
            )
            model((value - mean) / std)
        model.subsample_embedding()
    except torch.cuda.OutOfMemoryError:
        raise ResourceFailure("CUDA_OOM")

    return {
        "resource": {
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_host_ram_bytes": _rss_bytes(),
            "seconds_per_image": (time.perf_counter() - started) / len(fixture["entries"]),
            "index_growth_bytes": int(
                model.memory_bank.numel() * model.memory_bank.element_size()
            ),
        }
    }


def _failure_resource() -> dict[str, Any]:
    return {
        "peak_vram_bytes": 0,
        "peak_host_ram_bytes": _rss_bytes(),
        "seconds_per_image": 0.0,
        "index_growth_bytes": 0,
    }


def produce_probe(
    plan: Mapping[str, Any],
    storage_plan: Mapping[str, Any],
    *,
    training_identity_path: Path,
    dataset_root: Path,
    lease_directory: Path,
    anomalib_source: Path,
    admit: Callable[[Mapping[str, Any]], Any] = _admit_storage,
    writer: Callable[..., Mapping[str, Any]] = atomic_write,
    lease_factory: Callable[..., Any] = GpuLease,
    step: Callable[..., Mapping[str, Any]] = _live_step,
    runtime_binding: Callable[[Path], str] = _runtime_binding,
    source_verifier: Callable[[Path], None] = _verify_anomalib_source,
) -> dict[str, Any]:
    """Run the exact-H+ smoke once and atomically persist a probe artifact."""
    if (
        set(plan) != {"run_id", "fixture", "provenance"}
        or storage_plan.get("run_id") != plan.get("run_id")
    ):
        raise ChallengerBlocked(
            "probe requires a fresh canonical preflight plan without an inline result"
        )

    proof = admit(storage_plan)
    artifact = Path(proof.roots["artifact"]).resolve()
    lease = Path(lease_directory).resolve()
    try:
        lease.relative_to(artifact)
    except ValueError as exc:
        raise ChallengerBlocked(
            "lease_directory must be an artifact-root descendant"
        ) from exc
    if lease == artifact:
        raise ChallengerBlocked("lease_directory must be an artifact-root descendant")

    identity, _ = load_training_identity(training_identity_path, artifact)
    entries = validate_fixture(
        plan["fixture"], training_identity=identity, dataset_root=dataset_root
    )
    _provenance, gated, weights = validate_provenance(plan["provenance"])
    if gated:
        raise ChallengerBlocked("WEIGHT_ACCESS_REQUIRED")
    verify_local_weights(plan["provenance"], artifact, weights)
    try:
        binding = runtime_binding(Path(anomalib_source))
    except Exception as exc:
        raise ChallengerBlocked("RUNTIME_BINDING_UNAVAILABLE") from exc

    try:
        source_verifier(Path(anomalib_source))
        with lease_factory(lease, plan["run_id"], "superadd-exact-hplus-probe"):
            result = dict(
                step(
                    dataset_root=dataset_root,
                    fixture=plan["fixture"],
                    weight_path=Path(
                        plan["provenance"]["weights"]["exact_hplus"]["local_path"]
                    ),
                    anomalib_source=Path(anomalib_source),
                )
            )
        status = "READY"
        reason = "exact H+ representative forward/index step completed"
    except Exception as exc:
        try:
            import torch

            is_cuda_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        except Exception:
            is_cuda_oom = False
        if isinstance(exc, ReproductionFailure):
            status = "REPRODUCTION_FAILURE"
        elif isinstance(exc, ResourceFailure) or is_cuda_oom:
            status = "RESOURCE_FAILURE"
        else:
            status = "STOPPED_INCOMPLETE"
        result = {"resource": _failure_resource()}
        reason = f"{type(exc).__name__}:{sha256(str(exc).encode()).hexdigest()[:16]}"

    payload = {
        "schema_version": 1,
        "status": status,
        "variant": EXACT_HPLUS.variant,
        "recipe_fingerprint": _sha(asdict(EXACT_HPLUS)),
        "fixture_entries_sha256": _sha(entries),
        "resolved_weight_sha256": weights["exact_hplus"],
        "runtime_binding_sha256": binding,
        "resource": result["resource"],
        "reason": reason,
        "producer_module": PROBE_PRODUCER_MODULE,
        "producer_source_sha256": _source_sha256(),
        "anomalib_commit": ANOMALIB_COMMIT,
    }
    raw = _canonical(payload)
    destination = artifact / f"superadd-hplus-probe-{plan['run_id']}-{sha256(raw).hexdigest()}.json"
    outcome = writer(destination, raw, proof=proof, run_id=plan["run_id"], overwrite=False)
    if (
        outcome.get("status") != READY
        or not destination.is_file()
        or destination.read_bytes() != raw
    ):
        raise ChallengerBlocked("probe evidence write failed")
    return {
        **payload,
        "artifact": str(destination),
        "artifact_sha256": sha256(raw).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "storage-plan",
        "training-identity",
        "dataset-root",
        "lease-directory",
        "anomalib-source",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = produce_probe(
            json.loads(args.plan.read_text()),
            json.loads(args.storage_plan.read_text()),
            training_identity_path=args.training_identity,
            dataset_root=args.dataset_root,
            lease_directory=args.lease_directory,
            anomalib_source=args.anomalib_source,
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "artifact"},
                sort_keys=True,
            )
        )
        return 0
    except (OSError, json.JSONDecodeError, ChallengerBlocked, StorageBlocked) as exc:
        print(
            json.dumps(
                {
                    "status": "STOPPED_INCOMPLETE",
                    "reason": "exact H+ probe failed",
                    "exception_type": type(exc).__name__,
                    "exception_fingerprint": sha256(
                        f"{type(exc).__name__}:{exc}".encode()
                    ).hexdigest()[:16],
                },
                sort_keys=True,
            )
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "STOPPED_INCOMPLETE",
                    "reason": "exact H+ probe failed unexpectedly",
                    "exception_type": type(exc).__name__,
                    "exception_fingerprint": sha256(
                        type(exc).__name__.encode()
                    ).hexdigest()[:16],
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
