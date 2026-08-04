import errno
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fine_defect_ad.r1_overlay import (
    DOWNLOAD_BYTES, OVERLAY_BYTES, OVERLAY_LOCK_SHA256, RESERVE_BYTES,
    expected_overlay_plan, install_after_admission, main, validate_overlay_plan,
)
from fine_defect_ad.storage import PreflightProof, RunInvalidated, StorageBlocked


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

    def test_enospc_invalidates_admitted_run_and_preserves_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"; artifact.mkdir()
            overlay = Path(tmp) / "venv" / "r1-overlay"
            proof = PreflightProof("run", {"artifact": str(artifact)}, "x", "2026-08-04T00:00:00+00:00", {}, [{"component_id": "x"}], {})
            with patch("fine_defect_ad.r1_overlay.admit_overlay_install", return_value=proof):
                with self.assertRaisesRegex(RunInvalidated, "new run ID"):
                    install_after_admission(Path("plan.json"), overlay, lambda: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
            marker = artifact / ".invalidated-run.json"
            self.assertTrue(overlay.is_dir())
            self.assertEqual(__import__("json").loads(marker.read_text())["workflow_status"], "STOPPED_INCOMPLETE")

    def test_pip_enospc_invalidates_and_cli_is_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"; artifact.mkdir()
            overlay = Path(tmp) / "venv" / "r1-overlay"
            proof = PreflightProof("run", {"artifact": str(artifact)}, "x", "2026-08-04T00:00:00+00:00", {}, [{"component_id": "x"}], {})
            error = subprocess.CalledProcessError(1, ["pip"], stderr=b"OSError: [Errno 28] No space left on device")
            with patch("fine_defect_ad.r1_overlay.admit_overlay_install", return_value=proof):
                with self.assertRaises(RunInvalidated):
                    install_after_admission(Path("plan.json"), overlay, lambda: (_ for _ in ()).throw(error))
            self.assertTrue((artifact / ".invalidated-run.json").exists())
            with patch("fine_defect_ad.r1_overlay.install_command", side_effect=RunInvalidated("new run ID")):
                self.assertEqual(main(["--plan", "plan.json", "--overlay", "overlay", "--", "pip"]), 2)


if __name__ == "__main__":
    unittest.main()
