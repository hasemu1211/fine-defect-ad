"""G002's local-only, train/validation-only 1,000 optimizer-step pilot.

Imports of torch, Lightning, torchvision and anomalib deliberately live inside
``run_g002_pilot``: ordinary evidence/preflight tests do not need the R1 overlay.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .evidence import immutable_json
from .gpu_lock import BusyError, GpuLease
from .pilot import (PILOT_STEPS, STOPPED_INCOMPLETE, PilotEvidence, expected_pilot_protocol_metadata,
                    host_rss_bytes, lease_events)
from .r1 import R1_SEED
from .storage import Allocation, PreflightProof, StorageBlocked, atomic_write, preflight

TEACHER_SMALL_SHA256 = "a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d"
TEACHER_SMALL_BYTES = 10_779_695
CATEGORY = "sheet_metal"


class PilotStopped(RuntimeError):
    """A controlled early stop which must never become READY."""


@dataclass(frozen=True)
class G002Args:
    dataset_root: Path
    teacher_small: Path
    imagenette_root: Path
    run_id: str
    lease_directory: Path
    category: str = CATEGORY
    expected_teacher_sha256: str = TEACHER_SMALL_SHA256


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_assets(args: G002Args) -> dict[str, Any]:
    """Verify required local inputs without downloading or touching test data."""
    if args.category != CATEGORY:
        raise ValueError("G002 is fixed to category=sheet_metal")
    teacher = args.teacher_small.resolve()
    if not teacher.is_file():
        raise FileNotFoundError(f"teacher_small missing: {teacher}")
    if teacher.stat().st_size != TEACHER_SMALL_BYTES:
        raise ValueError("teacher_small byte size does not match tracked asset")
    actual = _sha256(teacher)
    if args.expected_teacher_sha256 != TEACHER_SMALL_SHA256 or actual != TEACHER_SMALL_SHA256:
        raise ValueError("teacher_small SHA-256 does not match the tracked asset")
    imagenette = args.imagenette_root.resolve()
    if not imagenette.is_dir() or not any(path.is_dir() for path in imagenette.iterdir()):
        raise FileNotFoundError(f"Imagenette ImageFolder root missing or empty: {imagenette}")
    category_root = args.dataset_root.resolve() / CATEGORY
    train, validation = category_root / "train", category_root / "validation"
    if not train.is_dir() or not validation.is_dir():
        raise FileNotFoundError("sheet_metal train and validation directories are required")
    return {"teacher_small": {"path": str(teacher), "bytes": teacher.stat().st_size, "sha256": actual},
            "imagenette_root": str(imagenette), "dataset_root": str(args.dataset_root.resolve()),
            "category": CATEGORY, "file_identity": train_val_file_identity(category_root)}


def train_val_file_identity(category_root: Path) -> dict[str, list[dict[str, str]]]:
    """Hash only G002's train/validation files; test paths are intentionally unreachable."""
    result: dict[str, list[dict[str, str]]] = {}
    for split, dirname in (("train", "train"), ("validation", "validation")):
        base = Path(category_root) / dirname
        files = [path for path in sorted(base.rglob("*")) if path.is_file()]
        result[split] = [{"path": path.relative_to(category_root).as_posix(), "sha256": _sha256(path)} for path in files]
    return result


def _payload_plan(run_id: str, payload: bytes, *, phase: str) -> tuple[list[Allocation], int, dict[str, Any]]:
    source = f"G002 {phase} payload sha256={sha256(payload).hexdigest()} bytes={len(payload)}"
    return ([Allocation("artifact", len(payload), "persistent", source, "g002-pilot-evidence")], len(payload),
            {"max_pending_atomic_write_bytes": len(payload), "measured_high_water_bytes": 0,
             "runtime_or_source_citation": source})


def _preflight(run_id: str, payload: bytes, phase: str,
               admission: Callable[..., PreflightProof] = preflight) -> PreflightProof:
    allocations, reserve, reserve_evidence = _payload_plan(run_id, payload, phase=phase)
    return admission(run_id=run_id, allocations=allocations, reserve_bytes=reserve, reserve_evidence=reserve_evidence)


def _lazy_runtime(args: G002Args, evidence: PilotEvidence, started: float) -> tuple[Any, Any, Any]:
    """Construct no-download runtime classes only after assets and lease admission."""
    import torch
    from lightning.pytorch import Callback, Trainer, seed_everything
    from anomalib.data.datasets.image.mvtecad2 import MVTecAD2Dataset
    from anomalib.data.datamodules.base.image import AnomalibDataModule
    from anomalib.data.utils import Split
    from anomalib.models import EfficientAd

    class TrainValMVTecAD2(AnomalibDataModule):
        def __init__(self) -> None:
            super().__init__(train_batch_size=1, eval_batch_size=1, num_workers=0, seed=R1_SEED)
            self.root, self.category = args.dataset_root.resolve(), CATEGORY

        def prepare_data(self) -> None:
            # Never call the downloading MVTecAD2 implementation.
            if not (self.root / self.category).is_dir():
                raise FileNotFoundError("local sheet_metal category is missing")

        def _setup(self, _stage: str | None = None) -> None:
            raise RuntimeError("G002 setup must not call base setup")

        def setup(self, stage: str | None = None) -> None:
            # Deliberately no super().setup(): it may create test datasets/splits.
            self.train_data = MVTecAD2Dataset(root=self.root, category=CATEGORY, split=Split.TRAIN,
                                              augmentations=self.train_augmentations)
            self.val_data = MVTecAD2Dataset(root=self.root, category=CATEGORY, split=Split.VAL,
                                            augmentations=self.val_augmentations)
            if len(self.train_data) != 137 or len(self.val_data) != 19:
                raise ValueError("G002 requires exactly 137 train and 19 validation images")

        def test_dataloader(self, *unused: object, **unused_kwargs: object) -> object:
            raise RuntimeError("G002 forbids test dataloader access")

    class LocalEfficientAd(EfficientAd):
        def prepare_pretrained_model(self) -> None:
            # Parent implementation downloads; this verified path is the only allowed teacher source.
            if _sha256(args.teacher_small) != TEACHER_SMALL_SHA256:
                raise ValueError("teacher_small changed after preflight")
            self.model.teacher.load_state_dict(torch.load(args.teacher_small, map_location=torch.device(self.device), weights_only=True))

    class StopAtPilot(Callback):
        def __init__(self) -> None:
            self.first_batch = True
            self.steps = 0

        def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
            if self.first_batch:
                self.first_batch = False
                evidence.record_setup(time.monotonic() - started)

        def on_before_optimizer_step(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
            finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                         for parameter in pl_module.parameters())
            if not finite:
                evidence.gradient_finite = False
                raise PilotStopped("GRADIENT_NONFINITE")

        def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
            step = int(trainer.global_step)
            if step <= self.steps:
                return
            self.steps = step
            evidence.record_step(timestamp=time.monotonic(), gradients_finite=True, host_rss_bytes=host_rss_bytes(),
                                 gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            if step == PILOT_STEPS:
                trainer.should_stop = True
            elif step > PILOT_STEPS:
                raise PilotStopped("PILOT_OVERSHOT")

    seed_everything(R1_SEED, workers=True)
    datamodule, callback = TrainValMVTecAD2(), StopAtPilot()
    model = LocalEfficientAd(imagenet_dir=args.imagenette_root.resolve(), model_size="small", lr=1e-4,
                             weight_decay=1e-5, pre_processor=True, post_processor=False, evaluator=False,
                             visualizer=False)
    trainer = Trainer(accelerator="gpu", devices=1, precision="32-true", deterministic=True,
                      max_steps=70_000, max_epochs=1_000, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                      limit_val_batches=0, callbacks=[callback])
    return model, datamodule, trainer


def run_g002_pilot(args: G002Args, *, admission: Callable[..., PreflightProof] = preflight,
                    writer: Callable[..., Mapping[str, Any]] = atomic_write) -> dict[str, Any]:
    """Run G002 under one lease and atomically persist immutable evidence.

    BusyError deliberately propagates. Every other runtime failure becomes
    STOPPED_INCOMPLETE evidence; it never enables a READY claim.
    """
    command = "g002-pilot"
    try:
        # A source-derived, fresh storage proof is required before the GPU window.
        initial = {"run_id": args.run_id, "command": command, "protocol": "G002 local-only pilot"}
        pre_gpu_proof = _preflight(args.run_id, json.dumps(initial, sort_keys=True).encode(), "pre-gpu", admission)
    except BusyError:
        raise
    except Exception as exc:
        return {"run_id": args.run_id, "status": STOPPED_INCOMPLETE, "limitations": [f"PREFLIGHT:{type(exc).__name__}"],
                "termination_cause": f"PREFLIGHT:{type(exc).__name__}"}

    evidence = PilotEvidence(args.run_id, command, 70_000, expected_pilot_protocol_metadata())
    cause: str | None = None
    assets: dict[str, Any] = {}
    try:
        with GpuLease(args.lease_directory, args.run_id, command):
            # Asset verification is part of the same exclusive GPU-heavy window as setup/fit/validation.
            assets = verify_local_assets(args)
            started = time.monotonic()
            model, datamodule, trainer = _lazy_runtime(args, evidence, started)
            trainer.fit(model, datamodule=datamodule)
            if len(evidence.step_timestamps) != PILOT_STEPS:
                cause = f"PILOT_STEPS_{len(evidence.step_timestamps)}_OF_{PILOT_STEPS}"
            elif evidence.gradient_finite is not False:
                validated = time.monotonic()
                trainer.limit_val_batches = 1.0  # fit validation stays disabled; this is the sole explicit validation.
                trainer.validate(model, datamodule=datamodule)
                evidence.record_validation(time.monotonic() - validated)
    except BusyError:
        raise
    except PilotStopped as exc:
        cause = str(exc)
    except Exception as exc:
        cause = f"RUNNER_EXCEPTION:{type(exc).__name__}"
    evidence.record_lease_events(lease_events(args.lease_directory, args.run_id))
    record = evidence.to_record(cause)
    record.update({"g002": {"category": CATEGORY, "seed": R1_SEED, "assets": assets,
                              "pre_gpu_storage_proof": asdict(pre_gpu_proof),
                              "train_val_file_identity": assets.get("file_identity", {})}})
    payload, payload_hash = immutable_json(record)
    destination = Path(pre_gpu_proof.roots["artifact"]) / f"g002-pilot-{payload_hash}.json"
    try:
        # Payload may differ from the preliminary proof; admit the exact bytes immediately before write.
        exact_proof = _preflight(args.run_id, payload, "exact-evidence", admission)
        result = writer(destination, payload, proof=exact_proof, run_id=args.run_id, overwrite=False)
        record["artifact"] = {"sha256": payload_hash, **dict(result)}
    except Exception as exc:
        record["status"] = STOPPED_INCOMPLETE
        record["termination_cause"] = f"EVIDENCE_WRITE:{type(exc).__name__}"
        record["limitations"] = [record["termination_cause"]]
    return record


def parse_args(argv: Sequence[str] | None = None) -> G002Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-small", type=Path, required=True)
    parser.add_argument("--imagenette-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lease-directory", type=Path, required=True)
    parser.add_argument("--category", default=CATEGORY, choices=[CATEGORY])
    parser.add_argument("--expected-teacher-sha256", default=TEACHER_SMALL_SHA256)
    raw = parser.parse_args(argv)
    return G002Args(**vars(raw))


def main(argv: Sequence[str] | None = None) -> int:
    record = run_g002_pilot(parse_args(argv))
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0 if record.get("status") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
