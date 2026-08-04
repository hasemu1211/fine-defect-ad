import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fine_defect_ad.acquire_r1_assets import ASSETS, StorageBlocked, download_plan, extract, inspect
from fine_defect_ad.storage import PreflightProof


class R1AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name) / "data"; self.root.mkdir()
        self.asset = {"archive": "tiny.zip", "url": "https://example.invalid/tiny.zip", "bytes": 0, "sha256": "", "format": "zip"}
        raw = self.root / "efficientad-upstream/raw"; raw.mkdir(parents=True); self.archive = raw / self.asset["archive"]
        with zipfile.ZipFile(self.archive, "w") as zipped: zipped.writestr("weights/small.pth", b"small")
        self.asset["bytes"] = self.archive.stat().st_size; self.asset["sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.proof = PreflightProof("run", {}, "x", "2026-08-04T00:00:00+00:00", {}, [{"component_id": "x"}], {})

    def tearDown(self): self.tmp.cleanup()

    def test_plan_has_exact_pinned_compressed_bytes(self):
        plan = download_plan("run")
        self.assertEqual(sum(x["bytes"] for x in plan["allocations"]), 1_597_121_733)

    def test_inspect_and_extract_are_streamed_and_contained(self):
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True):
            details = inspect("teacher", self.root)
            self.assertEqual((details["uncompressed_bytes"], details["maximum_file_bytes"]), (5, 5))
            roots = {"data": self.root}
            with patch("fine_defect_ad.acquire_r1_assets.preflight", return_value=self.proof), patch("fine_defect_ad.acquire_r1_assets.require_proof"):
                result = extract(name="teacher", run_id="run", roots=roots)
        self.assertEqual(result["status"], "READY")
        self.assertEqual((self.root / "efficientad-upstream/extracted/teacher/weights/small.pth").read_bytes(), b"small")

    def test_inspection_rejects_path_traversal(self):
        with zipfile.ZipFile(self.archive, "w") as zipped: zipped.writestr("../escape", b"x")
        self.asset["bytes"] = self.archive.stat().st_size; self.asset["sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True):
            with self.assertRaises(StorageBlocked): inspect("teacher", self.root)


if __name__ == "__main__": unittest.main()
