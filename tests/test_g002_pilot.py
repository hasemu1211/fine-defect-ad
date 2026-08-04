import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_defect_ad.g002_pilot import G002Args, train_val_file_identity, verify_local_assets


class G002PreflightTests(unittest.TestCase):
    def _args(self, root: Path, teacher: Path) -> G002Args:
        return G002Args(root, teacher, root / "imagenette", "g002-test", root / "lease", expected_teacher_sha256="expected")

    def test_hash_mismatch_stops_before_any_runtime_download(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); teacher = root / "teacher.pth"; teacher.write_bytes(b"bad")
            (root / "imagenette" / "class").mkdir(parents=True)
            (root / "sheet_metal" / "train").mkdir(parents=True)
            (root / "sheet_metal" / "validation").mkdir(parents=True)
            with patch("fine_defect_ad.g002_pilot.TEACHER_SMALL_BYTES", 3), patch(
                "fine_defect_ad.g002_pilot.TEACHER_SMALL_SHA256", "expected"
            ), patch("fine_defect_ad.g002_pilot._sha256", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    verify_local_assets(self._args(root, teacher))

    def test_identity_never_enumerates_test(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for relative in ("train/good/a.png", "validation/good/b.png", "test_public/bad/secret.png"):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(relative.encode())
            identity = train_val_file_identity(root)
            self.assertEqual([item["path"] for item in identity["train"]], ["train/good/a.png"])
            self.assertEqual([item["path"] for item in identity["validation"]], ["validation/good/b.png"])
            self.assertNotIn("test", repr(identity))


if __name__ == "__main__":
    unittest.main()
