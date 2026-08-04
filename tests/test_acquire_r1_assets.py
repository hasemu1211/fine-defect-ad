import errno
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fine_defect_ad.acquire_r1_assets import ASSETS, StorageBlocked, download, download_plan, extract, inspect
from fine_defect_ad.storage import PreflightProof


class R1AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name) / "data"; self.root.mkdir()
        self.artifact = Path(self.tmp.name) / "artifact"; self.artifact.mkdir()
        self.roots = {"data": self.root, "artifact": self.artifact}
        self.asset = {"archive": "tiny.zip", "url": "https://example.invalid/tiny.zip", "bytes": 0, "sha256": "", "format": "zip"}
        raw = self.root / "efficientad-upstream/raw"; raw.mkdir(parents=True); self.archive = raw / self.asset["archive"]
        with zipfile.ZipFile(self.archive, "w") as zipped: zipped.writestr("weights/small.pth", b"small")
        self.asset["bytes"] = self.archive.stat().st_size; self.asset["sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.proof = PreflightProof("run", {"artifact": str(self.artifact)}, "x", "2026-08-04T00:00:00+00:00", {}, [{"component_id": "x"}], {})

    def tearDown(self): self.tmp.cleanup()

    def test_plan_has_exact_pinned_compressed_bytes(self):
        plan = download_plan("run")
        self.assertEqual(sum(x["bytes"] for x in plan["allocations"]), 1_597_121_733)

    def test_inspect_and_extract_are_streamed_and_contained(self):
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True):
            details = inspect("teacher", self.root)
            self.assertEqual((details["uncompressed_bytes"], details["maximum_file_bytes"]), (5, 5))
            with patch("fine_defect_ad.acquire_r1_assets.preflight", return_value=self.proof), patch("fine_defect_ad.acquire_r1_assets.require_proof"):
                result = extract(name="teacher", run_id="run", roots=self.roots)
        self.assertEqual(result["status"], "READY")
        self.assertEqual((self.root / "efficientad-upstream/extracted/teacher/weights/small.pth").read_bytes(), b"small")

    def test_inspection_rejects_path_traversal(self):
        with zipfile.ZipFile(self.archive, "w") as zipped: zipped.writestr("../escape", b"x")
        self.asset["bytes"] = self.archive.stat().st_size; self.asset["sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True):
            with self.assertRaises(StorageBlocked): inspect("teacher", self.root)

    def test_complete_partial_finalizes_without_a_range_request(self):
        partial = self.archive.with_name("." + self.archive.name + ".partial")
        self.archive.replace(partial)
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True), \
             patch("fine_defect_ad.acquire_r1_assets.preflight", return_value=self.proof), \
             patch("fine_defect_ad.acquire_r1_assets.require_proof"), \
             patch("fine_defect_ad.acquire_r1_assets.urllib.request.urlopen", side_effect=AssertionError("must not request a complete partial")):
            result = download(name="teacher", run_id="run", roots=self.roots)
        self.assertEqual(result["status"], "READY")
        self.assertTrue(self.archive.is_file()); self.assertFalse(partial.exists())

    def test_complete_mismatched_partial_is_preserved_and_blocked(self):
        partial = self.archive.with_name("." + self.archive.name + ".partial")
        payload = b"x" * self.asset["bytes"]; partial.write_bytes(payload); self.archive.unlink()
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True), \
             patch("fine_defect_ad.acquire_r1_assets.preflight", return_value=self.proof), \
             patch("fine_defect_ad.acquire_r1_assets.require_proof"):
            with self.assertRaisesRegex(StorageBlocked, "complete partial archive hash mismatch"):
                download(name="teacher", run_id="run", roots=self.roots)
        self.assertEqual(partial.read_bytes(), payload); self.assertFalse(self.archive.exists())

    def test_enospc_invalidates_run_with_artifact_marker(self):
        payload = b"payload"
        self.archive.unlink()
        asset = {"archive": self.asset["archive"], "url": "https://example.invalid/tiny.zip", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "format": "zip"}
        partial = self.archive.with_name("." + self.archive.name + ".partial")

        class Response:
            status = 200; headers = {"Content-Length": str(len(payload))}
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _):
                if getattr(self, "sent", False): return b""
                self.sent = True; return payload

        class NoSpace:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def write(self, _): raise OSError(errno.ENOSPC, "no space")

        real_open = Path.open
        def fake_open(path, mode="r", *args, **kwargs):
            if path == partial and mode == "wb": return NoSpace()
            return real_open(path, mode, *args, **kwargs)
        with patch.dict(ASSETS, {"teacher": asset}, clear=True), \
             patch("fine_defect_ad.acquire_r1_assets.preflight", return_value=self.proof), \
             patch("fine_defect_ad.acquire_r1_assets.require_proof"), \
             patch("fine_defect_ad.acquire_r1_assets.urllib.request.urlopen", return_value=Response()), \
             patch.object(Path, "open", fake_open):
            result = download(name="teacher", run_id="run", roots=self.roots)
        marker = self.artifact / ".invalidated-run.json"
        self.assertEqual(result["status"], "INVALIDATED")
        self.assertEqual(json.loads(marker.read_text()), {"cause": "ENOSPC", "partial_path": str(partial), "run_id": "run", "status": "INVALIDATED", "workflow_status": "STOPPED_INCOMPLETE"})

    def test_archive_final_rename_enospc_invalidates_and_preserves_partial(self):
        partial = self.archive.with_name("." + self.archive.name + ".partial")
        self.archive.replace(partial)
        real_replace = __import__("os").replace
        def no_space_at_final(source, destination):
            if Path(source) == partial: raise OSError(errno.ENOSPC, "full")
            return real_replace(source, destination)
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True), \
             patch("fine_defect_ad.acquire_r1_assets.preflight", return_value=self.proof), \
             patch("fine_defect_ad.acquire_r1_assets.require_proof"), \
             patch("fine_defect_ad.acquire_r1_assets.os.replace", side_effect=no_space_at_final):
            result = download(name="teacher", run_id="run", roots=self.roots)
        self.assertEqual(result["status"], "INVALIDATED")
        self.assertEqual(result["workflow_status"], "STOPPED_INCOMPLETE")
        self.assertTrue(partial.exists())
        self.assertTrue((self.artifact / ".invalidated-run.json").exists())

    def test_extracted_member_rename_enospc_invalidates_and_preserves_partial(self):
        real_replace = __import__("os").replace
        def no_space_at_final(source, destination):
            if str(source).endswith(".partial"): raise OSError(errno.ENOSPC, "full")
            return real_replace(source, destination)
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True), \
             patch("fine_defect_ad.acquire_r1_assets.preflight", return_value=self.proof), \
             patch("fine_defect_ad.acquire_r1_assets.require_proof"), \
             patch("fine_defect_ad.acquire_r1_assets.os.replace", side_effect=no_space_at_final):
            result = extract(name="teacher", run_id="run", roots=self.roots)
        partial = self.root / "efficientad-upstream/extracted/teacher/weights/.small.pth.partial"
        self.assertEqual(result["status"], "INVALIDATED")
        self.assertEqual(result["workflow_status"], "STOPPED_INCOMPLETE")
        self.assertTrue(partial.exists())
        self.assertTrue((self.artifact / ".invalidated-run.json").exists())

    def test_download_requires_artifact_root(self):
        with patch.dict(ASSETS, {"teacher": self.asset}, clear=True):
            with self.assertRaisesRegex(StorageBlocked, "artifact roots"):
                download(name="teacher", run_id="run", roots={"data": self.root})


if __name__ == "__main__": unittest.main()
