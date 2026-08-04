from pathlib import Path
import unittest
from unittest.mock import patch

from fine_defect_ad.r1_overlay import (
    DOWNLOAD_BYTES, OVERLAY_BYTES, OVERLAY_LOCK_SHA256, RESERVE_BYTES,
    expected_overlay_plan, install_after_admission, validate_overlay_plan,
)
from fine_defect_ad.storage import StorageBlocked


class R1OverlayAdmissionTests(unittest.TestCase):
    def test_plan_is_reproducible_and_rejects_one_byte_or_unverified_changes(self):
        plan = expected_overlay_plan("r1")
        validate_overlay_plan(plan)
        self.assertEqual((plan["allocations"][0]["bytes"], plan["allocations"][1]["bytes"], plan["reserve_bytes"]),
                         (OVERLAY_BYTES, DOWNLOAD_BYTES, RESERVE_BYTES))
        self.assertEqual(plan["reserve_evidence"]["lock_sha256"], OVERLAY_LOCK_SHA256)
        for key, value in (("reserve_bytes", RESERVE_BYTES + 1), ("allocations", plan["allocations"][:-1])):
            changed = {**plan, key: value}
            with self.assertRaisesRegex(StorageBlocked, "exactly match"): validate_overlay_plan(changed)

    def test_missing_or_failed_proof_prevents_pip_callback(self):
        called = []
        with patch("fine_defect_ad.r1_overlay.admit_overlay_install", side_effect=StorageBlocked("no proof")):
            with self.assertRaisesRegex(StorageBlocked, "no proof"):
                install_after_admission(Path("plan.json"), Path("overlay"), lambda: called.append("pip"))
        self.assertEqual(called, [])
        with patch("fine_defect_ad.r1_overlay.admit_overlay_install", side_effect=StorageBlocked("missing proof")):
            with self.assertRaisesRegex(StorageBlocked, "missing proof"):
                install_after_admission(Path("missing-plan.json"), Path("overlay"), lambda: called.append("pip"))
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
