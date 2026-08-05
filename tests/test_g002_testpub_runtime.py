from pathlib import Path

import pytest

from fine_defect_ad.g002_testpub_runtime import BAD_COUNT, GOOD_COUNT, test_public_entries as entries


def _dataset(root: Path) -> Path:
    base = root / "sheet_metal" / "test_public"
    for label, count in (("good", GOOD_COUNT), ("bad", BAD_COUNT)):
        (base / label).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = base / label / f"{index:03}.png"; image.write_bytes(f"{label}-{index}".encode())
            if label == "bad":
                mask = base / "ground_truth" / "bad" / f"{image.stem}_mask.png"; mask.parent.mkdir(parents=True, exist_ok=True); mask.write_bytes(f"mask-{index}".encode())
    return root


def test_public_identity_set_binds_exact_counts_sources_and_bad_masks(tmp_path):
    rows = entries(_dataset(tmp_path))
    assert len(rows) == GOOD_COUNT + BAD_COUNT
    assert sum(row["label"] == "good" for row in rows) == GOOD_COUNT
    assert sum(row["label"] == "bad" for row in rows) == BAD_COUNT
    assert all(row["mask_sha256"] is None for row in rows[:GOOD_COUNT])
    assert all(row["mask_sha256"] for row in rows[GOOD_COUNT:])


def test_public_identity_set_rejects_missing_mask(tmp_path):
    root = _dataset(tmp_path)
    next((root / "sheet_metal" / "test_public" / "ground_truth" / "bad").glob("*.png")).unlink()
    with pytest.raises(ValueError, match="missing public-test mask"):
        entries(root)
