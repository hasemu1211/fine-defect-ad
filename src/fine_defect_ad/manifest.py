"""Immutable MVTec AD 2 ``sheet_metal`` JSONL manifest builder."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
from re import fullmatch
from typing import Iterable

from .storage import Allocation, atomic_write, preflight

PRIVATE_SPLITS = frozenset({"TESTpriv", "TESTpriv,mix"})
SPLIT_COUNTS = {
    "train": {"normal": 137, "anomalous": 0},
    "validation": {"normal": 19, "anomalous": 0},
    "TESTpub": {"normal": 24, "anomalous": 90},
    "TESTpriv": {"unknown": 142},
    "TESTpriv,mix": {"unknown": 142},
}
PUBLISHED_PRIVATE_AGGREGATE_COUNTS = {
    split: {"normal": 36, "anomalous": 106} for split in PRIVATE_SPLITS
}
TEST_SPLITS = frozenset(SPLIT_COUNTS) - {"train", "validation"}
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
# Filename grammar is deliberately narrow: an unexpected acquisition variant fails closed.
_NAME = r"(?P<number>\d{3})_(?P<variant>regular|mixed|overexposed|underexposed|shift_[123])\.(?:png|jpg|jpeg)"


@dataclass(frozen=True, order=True)
class Sample:
    sample_id: str
    scene_pair_id: str
    split: str
    anomaly_status: str
    lighting_id: str
    shift_status: str
    path: str = ""
    content_hash: str = ""


def canonical_manifest_hash(samples: list[Sample]) -> str:
    payload = [asdict(sample) for sample in sorted(samples)]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _relative_path(path: str) -> bool:
    value = PurePosixPath(path)
    return bool(path) and not value.is_absolute() and ".." not in value.parts and "\\" not in path


def validate_manifest(samples: list[Sample]) -> None:
    """Validate exact roles/counts plus identities/content leakage across test barriers."""
    required = {"sample_id", "scene_pair_id", "split", "anomaly_status", "lighting_id", "shift_status"}
    for sample in samples:
        allowed_statuses = {"unknown"} if sample.split in PRIVATE_SPLITS else {"normal", "anomalous"}
        if sample.split not in SPLIT_COUNTS or sample.anomaly_status not in allowed_statuses:
            raise ValueError(f"invalid manifest role: {sample!r}")
        if not all(getattr(sample, name) for name in required) or not sample.content_hash or not _relative_path(sample.path):
            raise ValueError(f"missing or non-relative manifest field: {sample!r}")
        if not fullmatch(r"[0-9a-f]{64}", sample.content_hash):
            raise ValueError(f"content_hash must be canonical SHA-256: {sample!r}")
    by_id: dict[str, set[str]] = {}; by_pair: dict[str, set[str]] = {}; by_path: dict[str, set[str]] = {}; by_content: dict[str, set[str]] = {}
    for sample in samples:
        by_id.setdefault(sample.sample_id, set()).add(sample.split)
        by_pair.setdefault(sample.scene_pair_id, set()).add(sample.split)
        by_path.setdefault(sample.path, set()).add(sample.split)
        by_content.setdefault(sample.content_hash, set()).add(sample.split)
    if len([s.path for s in samples]) != len({s.path for s in samples}): raise ValueError("duplicate path")
    for mapping, label in ((by_id, "sample_id"), (by_path, "path"), (by_content, "content_hash")):
        for value, splits in mapping.items():
            if splits & {"train", "validation"} and splits & TEST_SPLITS:
                raise ValueError(f"leakage across test barrier ({label}={value})")
    for value, splits in by_pair.items():
        if len(splits) > 1 and splits != PRIVATE_SPLITS:
            raise ValueError(f"scene pair crosses split roles ({value}={splits})")
    train_validation = [s for s in samples if s.split in {"train", "validation"}]
    if len({s.sample_id for s in train_validation}) != len(train_validation): raise ValueError("train/validation sample IDs overlap")
    counts = {split: Counter(s.anomaly_status for s in samples if s.split == split) for split in SPLIT_COUNTS}
    actual = {split: {status: counts[split][status] for status in expected} for split, expected in SPLIT_COUNTS.items()}
    if actual != SPLIT_COUNTS: raise ValueError(f"sheet_metal role/count mismatch: {counts}")
    private_pairs = {s.scene_pair_id for s in samples if s.split == "TESTpriv"}
    mix_pairs = {s.scene_pair_id for s in samples if s.split == "TESTpriv,mix"}
    if private_pairs != mix_pairs or len(private_pairs) != 142: raise ValueError("private regular/mixed scene pairs must match exactly")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _metadata(relative: PurePosixPath) -> tuple[str, str, str]:
    match = fullmatch(_NAME, relative.name)
    if match is None: raise ValueError(f"unrecognized sheet_metal filename: {relative}")
    number, variant = match.group("number", "variant")
    if variant == "mixed": return number, "mixed", "mixed"
    if variant == "regular": return number, "regular", "not_shifted"
    if variant.startswith("shift_"): return number, "regular", variant
    return number, variant, "not_shifted"


def _add_samples(root: Path, directory: str, split: str, status: str) -> list[Sample]:
    folder = root / directory
    if not folder.is_dir(): raise ValueError(f"missing required dataset directory: {directory}")
    result: list[Sample] = []
    for file in sorted(folder.iterdir(), key=lambda item: item.name):
        if not file.is_file() or file.suffix.lower() not in _IMAGE_SUFFIXES: continue
        relative = file.relative_to(root).as_posix()
        number, lighting, shift = _metadata(PurePosixPath(relative))
        # Only private regular/mixed capture variants describe the same scene ID.
        pair = f"private:{number}" if split in PRIVATE_SPLITS else f"{split}:{status}:{number}:{lighting}:{shift}"
        result.append(Sample(f"{split}:{relative}", pair, split, status, lighting, shift, relative, _sha256(file)))
    return result


def build_sheet_metal_manifest(dataset_root: Path | str) -> list[Sample]:
    """Build the deterministic image-only manifest; masks are checked but never emitted."""
    root = Path(dataset_root).resolve()
    groups = (("train/good", "train", "normal"), ("validation/good", "validation", "normal"),
              ("test_public/good", "TESTpub", "normal"), ("test_public/bad", "TESTpub", "anomalous"),
              ("test_private", "TESTpriv", "unknown"), ("test_private_mixed", "TESTpriv,mix", "unknown"))
    samples = [sample for directory, split, status in groups for sample in _add_samples(root, directory, split, status)]
    masks = root / "test_public/ground_truth/bad"
    bad = {sample.path.rsplit("/", 1)[-1] for sample in samples if sample.split == "TESTpub" and sample.anomaly_status == "anomalous"}
    mask_names = {file.name.removesuffix("_mask.png") + ".png" for file in masks.glob("*_mask.png")} if masks.is_dir() else set()
    if bad != mask_names or len(mask_names) != 90: raise ValueError("public bad images and ground_truth masks must correspond exactly (90)")
    samples.sort()
    validate_manifest(samples)
    return samples


def manifest_jsonl(samples: Iterable[Sample]) -> bytes:
    return ("".join(json.dumps(asdict(sample), sort_keys=True, separators=(",", ":")) + "\n" for sample in sorted(samples))).encode()


def public_summary(samples: list[Sample], run_id: str) -> dict:
    return {"run_id": run_id, "dataset": "mvtec_ad_2/sheet_metal", "manifest_sha256": canonical_manifest_hash(samples),
            "counts": {split: dict(expected) for split, expected in SPLIT_COUNTS.items()}, "sample_count": len(samples)}


def write_sheet_metal_manifest(dataset_root: Path | str, run_id: str) -> dict:
    if not run_id or not fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id): raise ValueError("run_id must be a safe non-empty identifier")
    samples = build_sheet_metal_manifest(dataset_root); payload = manifest_jsonl(samples)
    artifact = Path(os.environ["FINE_DEFECT_ARTIFACT_ROOT"]).resolve()
    destination = artifact / f"{run_id}.sheet_metal.manifest.jsonl"
    if destination.exists(): raise FileExistsError(f"refusing to overwrite existing manifest: {destination.name}")
    proof = preflight(run_id=run_id, allocations=[Allocation("artifact", len(payload), "persistent", "exact JSONL byte count", "sheet-metal-manifest-jsonl")], reserve_bytes=0,
                      reserve_evidence={"max_pending_atomic_write_bytes": 0, "measured_high_water_bytes": 0, "runtime_or_source_citation": "manifest payload is atomically written once"})
    result = atomic_write(destination, payload, proof=proof, run_id=run_id)
    if result["status"] != "READY": raise RuntimeError(f"manifest write did not complete: {result['status']}")
    return public_summary(samples, run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build immutable sheet_metal manifest JSONL")
    parser.add_argument("--dataset-root", required=True, type=Path); parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_sheet_metal_manifest(args.dataset_root, args.run_id), sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
