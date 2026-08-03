"""MVTec AD 2 sheet_metal manifest contract and leakage checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json


SPLIT_COUNTS = {
    "train": {"normal": 137, "anomalous": 0},
    "validation": {"normal": 19, "anomalous": 0},
    "TESTpub": {"normal": 24, "anomalous": 90},
    "TESTpriv": {"normal": 36, "anomalous": 106},
    "TESTpriv,mix": {"normal": 36, "anomalous": 106},
}
TEST_SPLITS = frozenset(SPLIT_COUNTS) - {"train", "validation"}


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


def validate_manifest(samples: list[Sample]) -> None:
    """Validate exact roles/counts plus identities/content leakage across test barriers."""
    required = {"sample_id", "scene_pair_id", "split", "anomaly_status", "lighting_id", "shift_status"}
    for sample in samples:
        if sample.split not in SPLIT_COUNTS or sample.anomaly_status not in {"normal", "anomalous"}:
            raise ValueError(f"invalid manifest role: {sample!r}")
        if not all(getattr(sample, name) for name in required) or not sample.content_hash or not sample.path:
            raise ValueError(f"missing required manifest field: {sample!r}")
    by_id: dict[str, set[str]] = {}
    by_path: dict[str, set[str]] = {}
    by_content: dict[str, set[str]] = {}
    for sample in samples:
        by_id.setdefault(sample.sample_id, set()).add(sample.split)
        if sample.path:
            by_path.setdefault(sample.path, set()).add(sample.split)
        if sample.content_hash:
            by_content.setdefault(sample.content_hash, set()).add(sample.split)
    if len([s.path for s in samples]) != len({s.path for s in samples}): raise ValueError("duplicate path")
    barriers = ({"train", "validation"}, TEST_SPLITS)
    for mapping, label in ((by_id, "sample_id"), (by_path, "path"), (by_content, "content_hash")):
        for value, splits in mapping.items():
            if splits & barriers[0] and splits & barriers[1]:
                raise ValueError(f"leakage across test barrier ({label}={value})")
    train_validation = [s for s in samples if s.split in {"train", "validation"}]
    if len({s.sample_id for s in train_validation}) != len(train_validation):
        raise ValueError("train/validation sample IDs overlap")
    counts = {split: Counter(s.anomaly_status for s in samples if s.split == split) for split in SPLIT_COUNTS}
    actual = {split: {status: counts[split][status] for status in expected} for split, expected in SPLIT_COUNTS.items()}
    if actual != SPLIT_COUNTS:
        raise ValueError(f"sheet_metal role/count mismatch: {counts}")

    private_pairs={s.scene_pair_id for s in samples if s.split=="TESTpriv"}
    mix_pairs={s.scene_pair_id for s in samples if s.split=="TESTpriv,mix"}
    if not private_pairs & mix_pairs: raise ValueError("intentional private/mix scene pairs required")
