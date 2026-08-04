import hashlib
import json
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from fine_defect_ad.acquire_mvtecad2 import ARCHIVE_NAME, StorageBlocked, archive_plan, download, load_terms_ack
from fine_defect_ad.storage import PreflightProof


class _RangeServer(BaseHTTPRequestHandler):
    payload = b""
    bad_range = False
    def do_GET(self):
        start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        if self.bad_range and start:
            self.send_response(200); self.send_header("Content-Length", str(len(self.payload))); self.end_headers(); self.wfile.write(self.payload); return
        body = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start: self.send_header("Content-Range", f"bytes {start}-{len(self.payload)-1}/{len(self.payload)}")
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.data, self.artifact = root / "data", root / "artifact"; self.data.mkdir(); self.artifact.mkdir()
        self.ack = root / "terms.json"; self.ack.write_text(json.dumps({"status":"ACCEPTED", "official_form_url":"https://www.mvtec.com/company/research/datasets/mvtec-ad-2/downloads", "license":"CC-BY-NC-SA-4.0", "noncommercial":True, "accepted_at":"2026-08-04T00:00:00+00:00"}))
        self.payload = b"tiny-mvtec-archive"; _RangeServer.payload = self.payload; _RangeServer.bad_range = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeServer); self.thread = threading.Thread(target=self.server.serve_forever); self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/archive"
        self.proof = PreflightProof("run", {"artifact":str(self.artifact)}, "x", "2026-08-04T00:00:00+00:00", {}, [{"component_id":"x"}], {})

    def tearDown(self): self.server.shutdown(); self.thread.join(); self.tmp.cleanup()

    def _run(self):
        return download(run_id="run", terms_ack=self.ack, url=self.url)

    def test_fresh_download_and_plan_count_one_atomic_archive(self):
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest()), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof) as pre, patch("fine_defect_ad.acquire_mvtecad2.require_proof") as require:
            result = self._run()
        self.assertEqual(result["status"], "READY"); self.assertEqual((self.data / ARCHIVE_NAME).read_bytes(), self.payload)
        self.assertFalse((self.data / ("." + ARCHIVE_NAME + ".partial")).exists()); self.assertTrue(require.called)
        self.assertEqual(pre.call_args.kwargs["allocations"][0].bytes, len(self.payload)); self.assertEqual(len(archive_plan("run")["allocations"]), 1)

    def test_resume_validates_range_and_hashes_existing_prefix(self):
        partial = self.data / ("." + ARCHIVE_NAME + ".partial"); partial.write_bytes(self.payload[:4])
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest()), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof), patch("fine_defect_ad.acquire_mvtecad2.require_proof"):
            self._run()
        self.assertEqual((self.data / ARCHIVE_NAME).read_bytes(), self.payload)

    def test_bad_range_and_bad_hash_preserve_partial(self):
        partial = self.data / ("." + ARCHIVE_NAME + ".partial"); partial.write_bytes(self.payload[:3]); _RangeServer.bad_range = True
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest()), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof):
            with self.assertRaises(StorageBlocked): self._run()
        self.assertEqual(partial.read_bytes(), self.payload[:3]); _RangeServer.bad_range = False
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", "0" * 64), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof), patch("fine_defect_ad.acquire_mvtecad2.require_proof"):
            with self.assertRaises(StorageBlocked): self._run()
        self.assertTrue(partial.exists())

    def test_missing_ack_and_pii_ack_are_rejected(self):
        with self.assertRaises(StorageBlocked): load_terms_ack(self.ack.with_name("missing.json"))
        self.ack.write_text(json.dumps({"status":"ACCEPTED", "official_form_url":"https://www.mvtec.com/company/research/datasets/mvtec-ad-2/downloads", "license":"CC-BY-NC-SA-4.0", "noncommercial":True, "accepted_at":"2026-08-04T00:00:00+00:00", "name":"no"}))
        with self.assertRaises(StorageBlocked): load_terms_ack(self.ack)


if __name__ == "__main__": unittest.main()
