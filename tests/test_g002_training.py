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

    def test_fake_exact_70k_run_is_ready_with_post_release_hashes(self):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from fine_defect_ad.g002_training import run_training, TrainingArgs
        from fine_defect_ad.g002_pilot import G002Args
        from fine_defect_ad.storage import PreflightProof
        with TemporaryDirectory() as raw:
            root=Path(raw); pilot={'median_seconds_per_step':1}; g=G002Args(root,root/'t',root/'i','run',root/'lease')
            args=TrainingArgs(root/'pilot','run',root,root/'metrics',g)
            class Lease:
                pending_signal=None
                def __init__(self,*a,**k): pass
                def __enter__(self): return self
                def __exit__(self,*a): pass
            class Art:
                def __init__(self,*a): self.root=root
                def checkpoint(self,step,serializer):
                    cp=root/'c'; sc=root/'s'; cp.write_bytes(b'c');sc.write_bytes(b's'); return cp,sc
                def metrics(self,rows): p=root/'m';p.write_bytes(b'm');return p
                def final(self,r): p=root/'f';p.write_bytes(b'f');return p
            class Trainer:
                global_step=70000; callbacks=[]; current_epoch=0; optimizers=[SimpleNamespace(param_groups=[{'lr':1.0}])]; callback_metrics={}
                def fit(self,*a,**k): pass
            runtime=lambda *a,**k:(object(),object(),Trainer(),None)
            proof=PreflightProof('run',{'artifact':str(root)},'f',datetime.now(timezone.utc).isoformat(),{},[],{'reserve_bytes':0})
            torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0),isfinite=lambda x:x)
            with patch('fine_defect_ad.g002_training.training_identity',return_value={'x':1}), patch('fine_defect_ad.g002_training.validate_training_lease'):
                record=run_training(args,admit_pilot_fn=lambda _:pilot,preflight_fn=lambda **k:proof,lease_factory=Lease,runtime_factory=runtime,artifacts_factory=Art,lease_event_loader=lambda *_:[{'state':'acquired','run_id':'run','command':'g002-training'},{'state':'released','run_id':'run','command':'g002-training','outcome':'normal'}],torch_module=torch,callback_base=object)
            self.assertEqual(record['status'],'READY'); self.assertEqual(set(record['artifacts']),{'checkpoint','sidecar','metrics','final'})

    def test_fake_69999_run_stops_with_normal_lease(self):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from fine_defect_ad.g002_training import run_training, TrainingArgs
        from fine_defect_ad.g002_pilot import G002Args
        from fine_defect_ad.storage import PreflightProof
        with TemporaryDirectory() as raw:
            root=Path(raw); pilot={'median_seconds_per_step':1}; g=G002Args(root,root/'t',root/'i','run',root/'lease')
            args=TrainingArgs(root/'pilot','run',root,root/'metrics',g)
            class Lease:
                pending_signal=None
                def __init__(self,*a,**k): pass
                def __enter__(self): return self
                def __exit__(self,*a): pass
            class Art:
                def __init__(self,*a): self.root=root
                def checkpoint(self,step,serializer):
                    cp=root/'c'; sc=root/'s'; cp.write_bytes(b'c');sc.write_bytes(b's'); return cp,sc
                def metrics(self,rows): p=root/'m';p.write_bytes(b'm');return p
                def final(self,r): p=root/'f';p.write_bytes(b'f');return p
            class Trainer:
                global_step=69999; callbacks=[]; current_epoch=0; optimizers=[SimpleNamespace(param_groups=[{'lr':1.0}])]; callback_metrics={}
                def fit(self,*a,**k): pass
            runtime=lambda *a,**k:(object(),object(),Trainer(),None)
            proof=PreflightProof('run',{'artifact':str(root)},'f',datetime.now(timezone.utc).isoformat(),{},[],{'reserve_bytes':0})
            torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0),isfinite=lambda x:x)
            with patch('fine_defect_ad.g002_training.training_identity',return_value={'x':1}), patch('fine_defect_ad.g002_training.validate_training_lease'):
                record=run_training(args,admit_pilot_fn=lambda _:pilot,preflight_fn=lambda **k:proof,lease_factory=Lease,runtime_factory=runtime,artifacts_factory=Art,lease_event_loader=lambda *_:[{'state':'acquired','run_id':'run','command':'g002-training'},{'state':'released','run_id':'run','command':'g002-training','outcome':'normal'}],torch_module=torch,callback_base=object)
            self.assertEqual(record['status'],'STOPPED_INCOMPLETE'); self.assertEqual(record['termination_cause'],'INCOMPLETE_STEPS:69999_OF_70000'); self.assertIn('final',record['artifacts'])

    def test_fake_signal_run_stops_resumable(self):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from fine_defect_ad.g002_training import run_training, TrainingArgs
        from fine_defect_ad.g002_pilot import G002Args
        from fine_defect_ad.storage import PreflightProof
        with TemporaryDirectory() as raw:
            root=Path(raw); pilot={'median_seconds_per_step':1}; g=G002Args(root,root/'t',root/'i','run',root/'lease')
            args=TrainingArgs(root/'pilot','run',root,root/'metrics',g)
            class Lease:
                pending_signal=15
                def __init__(self,*a,**k): pass
                def __enter__(self): return self
                def __exit__(self,*a): pass
            class Art:
                def __init__(self,*a): self.root=root
                def checkpoint(self,step,serializer):
                    cp=root/'c'; sc=root/'s'; cp.write_bytes(b'c');sc.write_bytes(b's'); return cp,sc
                def metrics(self,rows): p=root/'m';p.write_bytes(b'm');return p
                def final(self,r): p=root/'f';p.write_bytes(b'f');return p
            class Trainer:
                global_step=70000; callbacks=[]; current_epoch=0; optimizers=[SimpleNamespace(param_groups=[{'lr':1.0}])]; callback_metrics={}
                def fit(self,*a,**k):
                    [cb.on_train_batch_end(self,None,None,None,0) for cb in self.callbacks if hasattr(cb,"on_train_batch_end")]
            runtime=lambda *a,**k:(object(),object(),Trainer(),None)
            proof=PreflightProof('run',{'artifact':str(root)},'f',datetime.now(timezone.utc).isoformat(),{},[],{'reserve_bytes':0})
            torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0),isfinite=lambda x:x)
            with patch('fine_defect_ad.g002_training.training_identity',return_value={'x':1}), patch('fine_defect_ad.g002_training.validate_training_lease'):
                record=run_training(args,admit_pilot_fn=lambda _:pilot,preflight_fn=lambda **k:proof,lease_factory=Lease,runtime_factory=runtime,artifacts_factory=Art,lease_event_loader=lambda *_:[{'state':'acquired','run_id':'run','command':'g002-training'},{'state':'released','run_id':'run','command':'g002-training','outcome':'signal:15'}],torch_module=torch,callback_base=object)
            self.assertEqual(record['status'],'STOPPED_INCOMPLETE'); self.assertEqual(record['termination_cause'],'INTERRUPTED_RESUMABLE'); self.assertIn('final',record['artifacts'])
