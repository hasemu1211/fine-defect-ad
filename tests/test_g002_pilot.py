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

class G002BoundaryTests(unittest.TestCase):
    def test_scoped_parser_never_recurses_or_touches_test(self):
        from fine_defect_ad.g002_pilot import scoped_normal_images
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); normal = root / "sheet_metal" / "train" / "good"; normal.mkdir(parents=True)
            (normal / "a.png").write_bytes(b"x")
            (root / "sheet_metal" / "test_public" / "bad").mkdir(parents=True)
            calls = []
            original = Path.glob
            def spy(path, pattern):
                calls.append((str(path), pattern)); return original(path, pattern)
            with patch.object(Path, "glob", spy):
                self.assertEqual([p.name for p in scoped_normal_images(normal)], ["a.png"])
            self.assertEqual(calls, [(str(normal), "*.png")])
            self.assertNotIn("test", repr(calls))

    def test_lease_lifecycle_fails_closed_on_duplicate_or_bad_outcome(self):
        from fine_defect_ad.g002_pilot import validate_lease_events
        events = [
            {"state": "acquired", "run_id": "r", "command": "g002-pilot", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"state": "released", "run_id": "r", "command": "g002-pilot", "timestamp": "2026-01-01T00:00:01+00:00", "outcome": "normal"},
        ]
        self.assertEqual(validate_lease_events(events, "r", "g002-pilot")[1]["outcome"], "normal")
        with self.assertRaises(ValueError): validate_lease_events(events * 2, "r", "g002-pilot")
        events[-1]["outcome"] = "exception"
        with self.assertRaises(ValueError): validate_lease_events(events, "r", "g002-pilot")

    def test_non_ready_writer_stops_and_payload_has_no_local_paths(self):
        from datetime import datetime, timezone
        from fine_defect_ad.g002_pilot import run_g002_pilot
        from fine_defect_ad.storage import PreflightProof
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); artifact = root / "artifacts"; lease = artifact / "lease"; artifact.mkdir()
            args = G002Args(root, root / "unused.pth", root / "imagenette", "safe-run", lease,
                            expected_teacher_sha256="expected")
            proof = PreflightProof("safe-run", {"artifact": str(artifact)}, "fingerprint", datetime.now(timezone.utc).isoformat(),
                                   {"devices": {}}, [], {"reserve_bytes": 0})
            captured = []
            class Fit:
                def fit(self, *unused, **kwargs): return None
            class Validate:
                def validate(self, *unused, **kwargs): return None
            def runtime(_args, evidence, _started):
                evidence.record_setup(1); evidence.record_validation(1)
                for step in range(1000): evidence.record_step(timestamp=step, gradients_finite=True, host_rss_bytes=1, gpu_allocated_bytes=1, gpu_reserved_bytes=1)
                return object(), object(), Fit(), Validate()
            def writer(_destination, payload, **kwargs):
                captured.append(payload); return {"status": "INVALIDATED", "path": str(root / "secret")}
            with patch("fine_defect_ad.g002_pilot.verify_local_assets", return_value={"teacher_small": {"sha256": "h", "bytes": 1}, "file_identity": {}}), \
                 patch("fine_defect_ad.g002_pilot._lazy_runtime", runtime):
                record = run_g002_pilot(args, admission=lambda **kwargs: proof, writer=writer)
            self.assertEqual(record["status"], "STOPPED_INCOMPLETE")
            self.assertTrue(captured)
            self.assertNotIn(str(root).encode(), captured[0])
            self.assertNotIn(b'"path"', captured[0])
