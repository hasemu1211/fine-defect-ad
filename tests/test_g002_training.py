import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fine_defect_ad.g002_training import (PILOT_SHA256, TrainingBlocked, admit_pilot,
                                          checkpoint_interval_steps, resume_sidecar, validate_resume)
from fine_defect_ad.pilot import expected_pilot_protocol_metadata

class G002TrainingTests(TestCase):
    def pilot(self):
        return {"status":"READY", "completed_steps":1000, "gradient_finite":True, "termination_cause":None,
                "median_seconds_per_step":.142, **expected_pilot_protocol_metadata()}
    def test_exact_ready_pilot_gate_and_rpo_cadence(self):
        with TemporaryDirectory() as raw:
            path=Path(raw)/'pilot.json'; path.write_text(json.dumps(self.pilot()))
            with patch('fine_defect_ad.g002_training.PILOT_SHA256', sha256(path.read_bytes()).hexdigest()):
                self.assertEqual(admit_pilot(path)['status'], 'READY')
                self.assertEqual(checkpoint_interval_steps(self.pilot()), int(300/.142))
            with self.assertRaises(TrainingBlocked): admit_pilot(path)
    def test_resume_requires_checkpoint_hash_sidecar_and_declares_not_established(self):
        with TemporaryDirectory() as raw:
            checkpoint=Path(raw)/'last.ckpt'; checkpoint.write_bytes(b'checkpoint')
            sidecar=checkpoint.with_suffix('.ckpt.json'); sidecar.write_text(json.dumps(resume_sidecar(checkpoint, PILOT_SHA256)))
            self.assertEqual(validate_resume(checkpoint, sidecar)['resume_exactness'], 'NOT_ESTABLISHED')
            sidecar.write_text('{}')
            with self.assertRaises(TrainingBlocked): validate_resume(checkpoint, sidecar)
