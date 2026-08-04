from pathlib import Path
import unittest
from unittest.mock import patch

from fine_defect_ad.r1_overlay import install_after_admission
from fine_defect_ad.storage import StorageBlocked


class R1OverlayAdmissionTests(unittest.TestCase):
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
