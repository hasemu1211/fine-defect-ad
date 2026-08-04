import os
from pathlib import Path
import tempfile

import pytest

from fine_defect_ad.manifest import (
    SPLIT_COUNTS,
    build_sheet_metal_manifest,
    canonical_manifest_hash,
    manifest_jsonl,
    public_summary,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.as_posix().encode())


def _fixture(root: Path) -> None:
    for directory, count, variant in (("train/good", 137, "regular"), ("validation/good", 19, "regular"),
                                      ("test_private", 142, "regular"), ("test_private_mixed", 142, "mixed")):
        for index in range(count): _touch(root / directory / f"{index:03}_{variant}.png")
    variants = ("regular", "shift_1", "shift_2", "shift_3", "underexposed", "overexposed")
    for directory, count in (("test_public/good", 24), ("test_public/bad", 90)):
        for index in range(count): _touch(root / directory / f"{index // 6:03}_{variants[index % 6]}.png")
    for file in (root / "test_public/bad").glob("*.png"):
        _touch(root / "test_public/ground_truth/bad" / f"{file.stem}_mask.png")


def test_builder_is_deterministic_image_only_and_private_pairs_match():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw); _fixture(root)
        samples = build_sheet_metal_manifest(root)
        assert len(samples) == 554
        assert canonical_manifest_hash(samples) == canonical_manifest_hash(build_sheet_metal_manifest(root))
        assert {sample.path for sample in samples if "ground_truth" in sample.path} == set()
        assert {s.scene_pair_id for s in samples if s.split == "TESTpriv"} == {s.scene_pair_id for s in samples if s.split == "TESTpriv,mix"}
        public_pairs = {s.scene_pair_id for s in samples if s.split == "TESTpub"}
        assert len(public_pairs) == 19
        assert all(not Path(s.path).is_absolute() for s in samples)
        assert manifest_jsonl(samples).count(b"\n") == 554
        assert public_summary(samples, "r1")["counts"] == SPLIT_COUNTS


def test_builder_fails_closed_when_bad_mask_is_missing():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw); _fixture(root)
        next((root / "test_public/ground_truth/bad").glob("*.png")).unlink()
        with pytest.raises(ValueError, match="ground_truth"):
            build_sheet_metal_manifest(root)


def test_opt_in_authorized_sheet_metal_is_read_only_buildable():
    root = os.environ.get("FINE_DEFECT_SHEET_METAL_TEST_ROOT")
    if not root: pytest.skip("set FINE_DEFECT_SHEET_METAL_TEST_ROOT for authorized read-only dataset validation")
    samples = build_sheet_metal_manifest(root)
    assert len(samples) == 554
