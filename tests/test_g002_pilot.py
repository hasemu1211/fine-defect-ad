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

class G002OverlayDataSmokeTests(unittest.TestCase):
    def test_scoped_samples_transform_and_collate_without_test_access(self):
        """Runs only when the separately installed overlay is available; never fits."""
        import os, shutil, subprocess, textwrap
        python = Path(os.environ.get("FINE_DEFECT_OVERLAY_PYTHON", ""))
        if not python.is_file() or not Path(".internal/venv/r1-overlay/anomalib").is_dir():
            self.skipTest("R1 overlay is unavailable")
        script = r'''
import base64, tempfile
from pathlib import Path
from types import SimpleNamespace
from fine_defect_ad.g002_pilot import G002Args, _lazy_runtime
from fine_defect_ad.pilot import PilotEvidence, expected_pilot_protocol_metadata
png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
with tempfile.TemporaryDirectory() as raw:
    root = Path(raw); category = root / "sheet_metal"
    for split, count in (("train", 137), ("validation", 19)):
        leaf = category / split / "good"; leaf.mkdir(parents=True)
        for index in range(count): (leaf / f"{index:03d}.png").write_bytes(png)
    forbidden = category / "test_public" / "bad"; forbidden.mkdir(parents=True); (forbidden / "x.png").write_bytes(png)
    args = G002Args(root, root / "teacher.pth", root / "imagenette", "g002-smoke", root / "lease")
    model, module, _, _ = _lazy_runtime(args, PilotEvidence("g002-smoke", "smoke", 70000, expected_pilot_protocol_metadata()), 0)
    module.trainer = SimpleNamespace(model=model)
    calls, original = [], Path.glob
    def spy(path, pattern): calls.append((str(path), pattern)); return original(path, pattern)
    Path.glob = spy
    try: module.setup("fit")
    finally: Path.glob = original
    assert not any("test" in path or pattern == "**/*" for path, pattern in calls)
    assert type(module.train_data.augmentations).__name__ == "Resize"
    assert tuple(module.train_data.augmentations.size) == (256, 256)
    batch = module.train_data.collate_fn([module.train_data[0]])
    assert tuple(batch.image.shape[-2:]) == (256, 256)
'''
        env = {**os.environ, "PYTHONPATH": f"{Path.cwd() / 'src'}:{Path.cwd() / '.internal/venv/r1-overlay'}"}
        subprocess.run([str(python), "-c", textwrap.dedent(script)], check=True, env=env, capture_output=True)
