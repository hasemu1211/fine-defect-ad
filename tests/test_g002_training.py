import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fine_defect_ad.g002_training import (PILOT_SHA256, TrainingBlocked, admit_pilot,
                                          checkpoint_interval_steps)
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

    def test_deferred_lease_signal_is_recorded_until_callback_boundary(self):
        from fine_defect_ad.gpu_lock import GpuLease
        with TemporaryDirectory() as raw:
            lease = GpuLease(Path(raw), "run", "command", defer_signals=True)
            with lease:
                lease._signal(15, None)
                self.assertEqual(lease.pending_signal, 15)

class TrainingArtifactTests(TestCase):
    def test_exact_bytes_sidecar_metrics_and_immutable_writer(self):
        from datetime import datetime, timezone
        from fine_defect_ad.g002_training import TrainingArtifacts
        from fine_defect_ad.storage import PreflightProof
        with TemporaryDirectory() as raw:
            root=Path(raw); writes=[]
            def admit(**kwargs): return PreflightProof('run',{'artifact':str(root)},'f',datetime.now(timezone.utc).isoformat(),{},[],{'reserve_bytes':0})
            def writer(path,payload,**kwargs):
                Path(path).write_bytes(payload); writes.append(bytes(payload)); return {'status':'READY'}
            a=TrainingArtifacts(root,'run',{'identity':'x'},admit,writer)
            cp, side=a.checkpoint(2,lambda:b'checkpoint')
            self.assertTrue(cp.is_file() and side.is_file()); self.assertTrue(a.metrics([{'loss':1}]).is_file())
            self.assertEqual(len(writes),3)

    def test_two_checkpoint_commits_use_independent_slots(self):
        from datetime import datetime, timezone
        from fine_defect_ad.g002_training import TrainingArtifacts
        from fine_defect_ad.storage import PreflightProof
        with TemporaryDirectory() as raw:
            root=Path(raw)
            def admit(**kwargs): return PreflightProof('run',{'artifact':str(root)},'f',datetime.now(timezone.utc).isoformat(),{},[],{'reserve_bytes':kwargs['reserve_bytes']})
            def writer(path,payload,**kwargs): Path(path).write_bytes(payload); return {'status':'READY'}
            a=TrainingArtifacts(root,'run',{'i':1},admit,writer)
            first,_=a.checkpoint(1,lambda:b'a'); second,_=a.checkpoint(2,lambda:b'b')
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b'a')

    def test_slot_resume_and_lease_lifecycle_reject_corruption(self):
        from fine_defect_ad.g002_training import validate_slot_resume, validate_training_lease
        with TemporaryDirectory() as raw:
            cp=Path(raw)/'x.ckpt'; cp.write_bytes(b'x'); identity={'a':1}
            side={'checkpoint_name':cp.name,'checkpoint_sha256':sha256(b'x').hexdigest(),'identity_sha256':sha256(json.dumps(identity,separators=(',',':'),sort_keys=True).encode()).hexdigest(),'pilot_sha256':PILOT_SHA256,'global_step':3,'lineage':'r','resume_exactness':'NOT_ESTABLISHED'}
            cp.with_suffix('.ckpt.json').write_text(json.dumps(side)); self.assertEqual(validate_slot_resume(cp,identity)['global_step'],3)
            side['global_step']=70000; cp.with_suffix('.ckpt.json').write_text(json.dumps(side))
            with self.assertRaises(TrainingBlocked): validate_slot_resume(cp,identity)
        events=[{'state':'acquired','run_id':'r','command':'g002-training'},{'state':'released','run_id':'r','command':'g002-training','outcome':'normal'}]
        validate_training_lease(events,'r','normal')
        with self.assertRaises(TrainingBlocked): validate_training_lease(events,'r','signal:15')

    def test_resume_falls_back_only_to_valid_sibling_slot(self):
        from fine_defect_ad.g002_training import select_resume_slot, file_sha256
        with TemporaryDirectory() as raw:
            root=Path(raw); identity={'a':1}; ih=sha256(json.dumps(identity,separators=(',',':'),sort_keys=True).encode()).hexdigest()
            for slot, step in ((0,2),(1,3)):
                cp=root/f'g002-last-run-{slot}.ckpt'; cp.write_bytes(str(slot).encode())
                cp.with_suffix('.ckpt.json').write_text(json.dumps({'checkpoint_name':cp.name,'checkpoint_sha256':file_sha256(cp),'identity_sha256':ih,'pilot_sha256':PILOT_SHA256,'global_step':step,'lineage':'run','resume_exactness':'NOT_ESTABLISHED'}))
            bad=root/'g002-last-run-1.ckpt'; bad.write_bytes(b'corrupt')
            self.assertTrue(select_resume_slot(bad,identity).name.endswith('-0.ckpt'))
            (root/'g002-last-run-0.ckpt.json').unlink()
            with self.assertRaises(TrainingBlocked): select_resume_slot(bad,identity)

    def test_cli_help_imports_all_runtime_symbols(self):
        import os, subprocess, sys
        result = subprocess.run([sys.executable, '-m', 'fine_defect_ad.g002_training', '--help'], env={**os.environ, 'PYTHONPATH':'src'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--pilot-evidence', result.stdout)
