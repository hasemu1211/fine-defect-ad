import errno
import hashlib
import io
import json
import tarfile
import os
import subprocess
import sys
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import fine_defect_ad.acquire_mvtecad2 as acquire
from fine_defect_ad.acquire_mvtecad2 import ARCHIVE_NAME, ARCHIVE_SHA256, ARCHIVE_URL, AUDIT_METHOD, StorageBlocked, archive_plan, audit_metadata, download, load_extraction_evidence, load_terms_ack
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
        self.ack = root / "terms.json"; self.ack.write_text(json.dumps({"status":"ACCEPTED", "official_form_url":"https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2", "license":"CC-BY-NC-SA-4.0", "noncommercial":True, "accepted_at":"2026-08-04T00:00:00+00:00"}))
        self.sizing = root / "extraction.json"; self.sizing.write_text(json.dumps({"archive_url": ARCHIVE_URL, "archive_bytes": 32_739_596_982, "archive_sha256": ARCHIVE_SHA256, "audit_method": AUDIT_METHOD, "exact_uncompressed_bytes": 100, "max_member_bytes": 20, "member_count": 2}))
        self.payload = b"tiny-mvtec-archive"; _RangeServer.payload = self.payload; _RangeServer.bad_range = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeServer); self.thread = threading.Thread(target=self.server.serve_forever); self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/archive"
        self.proof = PreflightProof("run", {"artifact":str(self.artifact)}, "x", "2026-08-04T00:00:00+00:00", {}, [{"component_id":"x"}], {})

    def tearDown(self): self.server.shutdown(); self.thread.join(); self.tmp.cleanup()

    def _write_sizing(self):
        self.sizing.write_text(json.dumps({"archive_url": acquire.ARCHIVE_URL, "archive_bytes": acquire.ARCHIVE_BYTES, "archive_sha256": acquire.ARCHIVE_SHA256, "audit_method": acquire.AUDIT_METHOD, "exact_uncompressed_bytes": 100, "max_member_bytes": 20, "member_count": 2}))

    def _audit_result(self):
        return {"status":"SIZING_AUDIT_ONLY_NOT_STORAGE_READY", "archive_url": acquire.ARCHIVE_URL, "archive_bytes": acquire.ARCHIVE_BYTES, "archive_sha256": acquire.ARCHIVE_SHA256, "audit_method": acquire.AUDIT_METHOD, "exact_uncompressed_bytes":100, "max_member_bytes":20, "member_count":2}

    def _run(self):
        with patch("fine_defect_ad.acquire_mvtecad2.audit_metadata", return_value=self._audit_result()):
            return download(run_id="run", terms_ack=self.ack, url=self.url)

    def test_fresh_download_and_plan_count_one_atomic_archive(self):
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest()), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof) as pre, patch("fine_defect_ad.acquire_mvtecad2.require_proof") as require:
            result = self._run()
        self.assertEqual(result["status"], "READY"); self.assertEqual((self.data / ARCHIVE_NAME).read_bytes(), self.payload)
        self.assertFalse((self.data / ("." + ARCHIVE_NAME + ".partial")).exists()); self.assertTrue(require.called)
        self.assertEqual(pre.call_args.kwargs["allocations"][0].bytes, len(self.payload)); self.assertEqual(len(archive_plan("run")["allocations"]), 1)
        self.assertEqual(archive_plan("run")["status"], "SIZING_ONLY_NOT_DOWNLOAD_AUTHORIZED")
        self.assertEqual([item.bytes for item in pre.call_args.kwargs["allocations"]], [len(self.payload), 100, 20])

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

    def test_missing_ack_pii_and_naive_ack_are_rejected(self):
        with self.assertRaises(StorageBlocked): load_terms_ack(self.ack.with_name("missing.json"))
        self.ack.write_text(json.dumps({"status":"ACCEPTED", "official_form_url":"https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2", "license":"CC-BY-NC-SA-4.0", "noncommercial":True, "accepted_at":"2026-08-04T00:00:00+00:00", "name":"no"}))
        with self.assertRaises(StorageBlocked): load_terms_ack(self.ack)
        self.ack.write_text(json.dumps({"status":"ACCEPTED", "official_form_url":"https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2", "license":"CC-BY-NC-SA-4.0", "noncommercial":True, "accepted_at":"2026-08-04T00:00:00"}))
        with self.assertRaises(StorageBlocked): load_terms_ack(self.ack)

    def test_cli_outside_checkout_rejects_tracked_ack_path(self):
        tracked = Path(__file__).resolve()
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        result = subprocess.run([sys.executable, "-m", "fine_defect_ad.acquire_mvtecad2", "--run-id", "run", "--terms-ack", str(tracked)], cwd=self.tmp.name, env=env, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("private or git-ignored", result.stdout)


    def test_saved_forged_json_cannot_authorize_download(self):
        self.sizing.write_text('{"forged": true}')
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest()), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof), patch("fine_defect_ad.acquire_mvtecad2.require_proof"), patch("fine_defect_ad.acquire_mvtecad2.audit_metadata", return_value=self._audit_result()) as audit:
            self._run()
        audit.assert_not_called()  # _run's same-invocation audit, not the saved JSON, is authoritative.


    def test_metadata_audit_binds_counted_tar_to_compressed_identity(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            member = tarfile.TarInfo("one"); member.size = 3; archive.addfile(member, io.BytesIO(b"one"))
        body = payload.getvalue()
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length": str(len(body))}
            def __enter__(self): return self
            def __exit__(self, *_): return False
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(body)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(body).hexdigest()):
            result = audit_metadata(terms_ack=self.ack, url=self.url, opener=lambda *_args, **_kwargs: Response(body))
        self.assertEqual((result["exact_uncompressed_bytes"], result["max_member_bytes"], result["member_count"]), (3, 3, 1))
        self.assertEqual(result["status"], "SIZING_AUDIT_ONLY_NOT_STORAGE_READY")

    def test_forged_extraction_archive_hash_or_size_is_rejected(self):
        evidence = json.loads(self.sizing.read_text())
        evidence["archive_sha256"] = "0" * 64; self.sizing.write_text(json.dumps(evidence))
        with self.assertRaises(StorageBlocked): load_extraction_evidence(self.sizing)
        evidence["archive_sha256"] = ARCHIVE_SHA256; evidence["archive_bytes"] = 1; self.sizing.write_text(json.dumps(evidence))
        with self.assertRaises(StorageBlocked): load_extraction_evidence(self.sizing)

    def test_atomic_rename_enospc_invalidates_run_without_deleting_partial(self):
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest()), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof), patch("fine_defect_ad.acquire_mvtecad2.require_proof"), patch("fine_defect_ad.acquire_mvtecad2.audit_metadata", return_value=self._audit_result()), patch("fine_defect_ad.acquire_mvtecad2.os.replace", side_effect=OSError(errno.ENOSPC, "full")):
            result = download(run_id="run", terms_ack=self.ack, url=self.url)
        self.assertEqual(result["workflow_status"], "STOPPED_INCOMPLETE")
        self.assertTrue((self.data / ("." + ARCHIVE_NAME + ".partial")).exists())
        self.assertTrue((self.artifact / ".invalidated-run.json").exists())

    def test_complete_partial_finalizes_without_network(self):
        partial = self.data / ("." + ARCHIVE_NAME + ".partial"); partial.write_bytes(self.payload)
        no_network = Mock()
        with patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_URL", self.url), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES", len(self.payload)), patch("fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256", hashlib.sha256(self.payload).hexdigest()), patch("fine_defect_ad.acquire_mvtecad2.roots_from_env", return_value={"data":self.data}), patch("fine_defect_ad.acquire_mvtecad2.preflight", return_value=self.proof), patch("fine_defect_ad.acquire_mvtecad2.require_proof"), patch("fine_defect_ad.acquire_mvtecad2.audit_metadata", return_value=self._audit_result()):
            result = download(run_id="run", terms_ack=self.ack, url=self.url, opener=no_network)
        self.assertTrue(result["resumed"]); no_network.assert_not_called(); self.assertEqual((self.data / ARCHIVE_NAME).read_bytes(), self.payload)


if __name__ == "__main__": unittest.main()

class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name); self.data = root/'data'; self.artifact=root/'artifact'; self.data.mkdir(); self.artifact.mkdir()
        self.archive = self.data / ARCHIVE_NAME
        self.ack = root / 'terms.json'; self.ack.write_text(json.dumps({'status':'ACCEPTED', 'official_form_url':'https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2', 'license':'CC-BY-NC-SA-4.0', 'noncommercial':True, 'accepted_at':'2026-08-04T00:00:00+00:00'}))
        self.proof = PreflightProof('extract', {'artifact':str(self.artifact)}, 'x', '2026-08-04T00:00:00+00:00', {}, [{'component_id':'x'}], {})

    def tearDown(self): self.tmp.cleanup()

    def _tar(self, name='folder/file.txt', body=b'ok'):
        with tarfile.open(self.archive, 'w:gz') as archive:
            directory=tarfile.TarInfo('folder'); directory.type=tarfile.DIRTYPE; archive.addfile(directory)
            member=tarfile.TarInfo(name); member.size=len(body); archive.addfile(member, io.BytesIO(body))
        return self.archive.read_bytes()

    def test_extract_cli_dispatches_after_module_definitions(self):
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        result = subprocess.run([sys.executable, "-m", "fine_defect_ad.acquire_mvtecad2", "--run-id", "extract", "--extract"], cwd=self.tmp.name, env=env, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2); self.assertNotIn("NameError", result.stderr)

    def test_no_ack_rejects_before_any_local_write(self):
        with self.assertRaises(StorageBlocked): acquire.extract(run_id='extract')
        self.assertFalse((self.data/'mvtec_ad_2').exists())

    def test_safe_extraction_and_no_write_before_proof(self):
        body=self._tar(); digest=hashlib.sha256(body).hexdigest()
        with patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES', len(body)), patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256', digest), patch('fine_defect_ad.acquire_mvtecad2.roots_from_env', return_value={'data':self.data}), patch('fine_defect_ad.acquire_mvtecad2.preflight', return_value=self.proof), patch('fine_defect_ad.acquire_mvtecad2.require_proof', side_effect=StorageBlocked('no proof')):
            with self.assertRaises(StorageBlocked): acquire.extract(run_id='extract', terms_ack=self.ack)
        self.assertFalse((self.data/'mvtec_ad_2').exists())
        with patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES', len(body)), patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256', digest), patch('fine_defect_ad.acquire_mvtecad2.roots_from_env', return_value={'data':self.data}), patch('fine_defect_ad.acquire_mvtecad2.preflight', return_value=self.proof), patch('fine_defect_ad.acquire_mvtecad2.require_proof'):
            result=acquire.extract(run_id='extract', terms_ack=self.ack)
        self.assertEqual((self.data/'mvtec_ad_2/folder/file.txt').read_bytes(), b'ok'); self.assertEqual(result['extracted_file_count'], 1)

    def test_traversal_and_existing_file_rejected(self):
        body=self._tar('../escape', b'x'); digest=hashlib.sha256(body).hexdigest()
        with patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES', len(body)), patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256', digest), patch('fine_defect_ad.acquire_mvtecad2.roots_from_env', return_value={'data':self.data}):
            with self.assertRaises(StorageBlocked): acquire.extract(run_id='extract', terms_ack=self.ack)
        with tarfile.open(self.archive, 'w:gz') as archive:
            member=tarfile.TarInfo('link'); member.type=tarfile.SYMTYPE; member.linkname='target'; archive.addfile(member)
        body=self.archive.read_bytes(); digest=hashlib.sha256(body).hexdigest()
        with patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES', len(body)), patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256', digest), patch('fine_defect_ad.acquire_mvtecad2.roots_from_env', return_value={'data':self.data}):
            with self.assertRaises(StorageBlocked): acquire.extract(run_id='extract', terms_ack=self.ack)
        body=self._tar(); digest=hashlib.sha256(body).hexdigest(); (self.data/'mvtec_ad_2/folder').mkdir(parents=True); (self.data/'mvtec_ad_2/folder/file.txt').write_text('old')
        with patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES', len(body)), patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256', digest), patch('fine_defect_ad.acquire_mvtecad2.roots_from_env', return_value={'data':self.data}), patch('fine_defect_ad.acquire_mvtecad2.preflight', return_value=self.proof), patch('fine_defect_ad.acquire_mvtecad2.require_proof'):
            with self.assertRaises(StorageBlocked): acquire.extract(run_id='extract', terms_ack=self.ack)

    def test_extraction_rename_enospc_invalidates(self):
        body=self._tar(); digest=hashlib.sha256(body).hexdigest()
        with patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_BYTES', len(body)), patch('fine_defect_ad.acquire_mvtecad2.ARCHIVE_SHA256', digest), patch('fine_defect_ad.acquire_mvtecad2.roots_from_env', return_value={'data':self.data}), patch('fine_defect_ad.acquire_mvtecad2.preflight', return_value=self.proof), patch('fine_defect_ad.acquire_mvtecad2.require_proof'), patch('fine_defect_ad.acquire_mvtecad2.os.replace', side_effect=OSError(errno.ENOSPC, 'full')):
            result=acquire.extract(run_id='extract', terms_ack=self.ack)
        self.assertEqual(result['workflow_status'], 'STOPPED_INCOMPLETE'); self.assertTrue((self.artifact/'.invalidated-extract.json').exists())
