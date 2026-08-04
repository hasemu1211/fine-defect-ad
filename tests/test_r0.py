from dataclasses import replace
from pathlib import Path
import errno, json, os, signal, subprocess, sys, tempfile, time, unittest
from unittest.mock import patch
from fine_defect_ad.evidence import DER_REQUIRED, validate_decision_register, validate_der, validate_evidence
from fine_defect_ad.gpu_lock import BusyError, GpuLease, benchmark_window
from fine_defect_ad.manifest import SPLIT_COUNTS, Sample, canonical_manifest_hash, validate_manifest
from fine_defect_ad.runtime import collect_runtime_evidence, container_gpu_spike, discover_docker_storage
from fine_defect_ad.storage import Allocation, ROOT_ENV, StorageBlocked, atomic_write, preflight, require_proof
from fine_defect_ad.preflight import main as preflight_main

class R0Tests(unittest.TestCase):
 def samples(self):
  out=[]
  for split,counts in SPLIT_COUNTS.items():
   for status,n in counts.items():
    for i in range(n): out.append(Sample(f'{split}-{status}-{i}',f'pair-{i}' if split in {'TESTpriv','TESTpriv,mix'} else f'{split}-{i}',split,status,'regular','id',f'{split}/{status}/{i}',f'h-{split}-{status}-{i}'))
  return out
 def test_manifest_roles_hash_path_and_leakage(self):
  xs=self.samples(); validate_manifest(xs); self.assertEqual(canonical_manifest_hash(xs),canonical_manifest_hash(xs[::-1]))
  i=next(i for i,x in enumerate(xs) if x.split=='TESTpub'); xs[i]=replace(xs[i],sample_id=xs[0].sample_id)
  with self.assertRaisesRegex(ValueError,'leakage'): validate_manifest(xs)
 def test_der_actual_register_and_schema_contract(self):
  item={k:'x' for k in DER_REQUIRED};item.update(decision_id='DEC-R0-1',status='proposed',drivers=[],alternatives=[]);validate_der(item)
  rows=validate_decision_register(Path('evidence/decision-register.yaml')); self.assertEqual(len(rows),2)
  schema=json.loads(Path('evidence/schemas/decision-evidence-register.schema.json').read_text()); self.assertEqual(schema['type'],'array'); self.assertEqual(schema['items']['type'],'object')
  validate_evidence({'run_id':'x','timestamp':'t','command':'x','status':'INVALIDATED','limitations':['ENOSPC']})
 def test_lock_contention_exception_and_window(self):
  with tempfile.TemporaryDirectory() as raw:
   d=Path(raw)
   with GpuLease(d,'one','train'):
    with self.assertRaisesRegex(BusyError,'BUSY.*one'):
     with GpuLease(d,'two','eval'): pass
   self.assertEqual(benchmark_window(d,'three',[lambda:'start',lambda:'measure',lambda:'stop']),['start','measure','stop'])
   with self.assertRaises(RuntimeError):
    with GpuLease(d,'exception','train'): raise RuntimeError('simulated failure')
   events=list((d/'gpu-heavy-events').glob('*.json')); self.assertTrue(events)
   latest=json.loads((d/'gpu-heavy-holder.json').read_text()); self.assertEqual(latest['state'],'released'); self.assertIn('timestamp',latest)
 def test_lock_signal_release_artifact(self):
  with tempfile.TemporaryDirectory() as raw:
   code="from pathlib import Path; from fine_defect_ad.gpu_lock import GpuLease; import time;\nwith GpuLease(Path(r'%s'),'sig','test'): time.sleep(30)" % raw
   child=subprocess.Popen([sys.executable,'-c',code], env={**os.environ,'PYTHONPATH':'src'}); time.sleep(.3); child.send_signal(signal.SIGTERM); child.wait(timeout=5)
   holder=json.loads((Path(raw)/'gpu-heavy-holder.json').read_text()); self.assertEqual(holder['outcome'],f'signal:{signal.SIGTERM}')
 def _storage(self):
  raw=tempfile.TemporaryDirectory(); base=Path(raw.name); roots={name:base/name for name in ('data','artifact','cache','package_cache','temp','venv','docker_root','source')}
  for p in roots.values():p.mkdir()
  (roots['source']/'.git').mkdir()
  env={'FINE_DEFECT_AUTHORIZED_NTFS_ROOT':str(base), **{ROOT_ENV[k]:str(v) for k,v in roots.items()}}
  def mount(p): return ('/ntfs','ntfs3','rw') if Path(p) in {roots['data'],roots['artifact']} else ('/','ext4','rw')
  docker={'status':'READY','root':str(roots['docker_root']),'driver':'overlayfs','driver_type':'io.containerd.snapshotter.v1','backing_fs':'ext4','image_store_path':'UNKNOWN_CONTAINED_BY_DOCKER_ROOT','evidence':{'command':['docker','info'],'returncode':0}}
  return raw,roots,env,mount,docker
 def test_storage_honesty_components_reserve_capacity_and_write(self):
  raw,roots,env,mount,docker=self._storage()
  with raw,patch.dict(os.environ,env,clear=True),patch('fine_defect_ad.storage._mount',side_effect=mount),patch('fine_defect_ad.runtime.discover_docker_storage',return_value=docker):
   reserve={'max_pending_atomic_write_bytes':2,'measured_high_water_bytes':1,'runtime_or_source_citation':'measured:test'}
   proof=preflight(run_id='r',allocations=[Allocation('data',1,'persistent','header','dataset-header'),Allocation('artifact',2,'transient','measured','artifact-temp'),Allocation('artifact',2,'transient','measured','artifact-temp')],reserve_bytes=3,reserve_evidence=reserve)
   self.assertEqual(proof.filesystems['data']['dirty_state'],'UNKNOWN_WHILE_MOUNTED'); self.assertTrue(proof.filesystems['data']['mount_rw_accepted']); self.assertEqual(len(proof.components),2)
   self.assertEqual(atomic_write(roots['artifact']/'x',b'ok',proof=proof,run_id='r')['status'],'READY')
   with self.assertRaises(StorageBlocked): preflight(run_id='x',allocations=[Allocation('data',1,'persistent','a','same'),Allocation('artifact',1,'persistent','b','same')],reserve_bytes=3,reserve_evidence=reserve)
   with self.assertRaises(StorageBlocked): preflight(run_id='x',allocations=[],reserve_bytes=3)
   with patch('fine_defect_ad.storage.os.statvfs') as sf:
    sf.return_value=type('S',(),{'f_bavail':0,'f_frsize':1})()
    with self.assertRaises(StorageBlocked): require_proof(proof,run_id='r')
 def test_daemon_docker_root_needs_no_user_write_and_cli_plan(self):
  raw,roots,env,mount,docker=self._storage()
  with raw,patch.dict(os.environ,env,clear=True),patch('fine_defect_ad.storage._mount',side_effect=mount),patch('fine_defect_ad.runtime.discover_docker_storage',return_value=docker):
   real_access=os.access
   with patch('fine_defect_ad.storage.os.access',side_effect=lambda path,mode: False if Path(path)==roots['docker_root'] else real_access(path,mode)):
    plan={'run_id':'cli','allocations':[{'root':'artifact','bytes':1,'kind':'persistent','source':'tiny-r0-evidence','component_id':'cli'}],'reserve_bytes':1,'reserve_evidence':{'max_pending_atomic_write_bytes':1,'measured_high_water_bytes':0,'runtime_or_source_citation':'tiny-plan'}}
    p=roots['temp']/'plan.json'; p.write_text(json.dumps(plan)); self.assertEqual(preflight_main(['--plan',str(p)]),0)
    plan.pop('reserve_evidence'); p.write_text(json.dumps(plan)); self.assertEqual(preflight_main(['--plan',str(p)]),2)
 def test_proof_uses_stable_docker_storage_identity(self):
  raw,roots,env,mount,docker=self._storage()
  with raw,patch.dict(os.environ,env,clear=True),patch('fine_defect_ad.storage._mount',side_effect=mount),patch('fine_defect_ad.runtime.discover_docker_storage',side_effect=[docker,{**docker,'evidence':{**docker['evidence'],'stdout':'new docker info SystemTime'}}]):
   proof=preflight(run_id='r',allocations=[Allocation('artifact',1,'persistent','x','x')],reserve_bytes=0,reserve_evidence={'max_pending_atomic_write_bytes':0,'measured_high_water_bytes':0,'runtime_or_source_citation':'test'})
   require_proof(proof,run_id='r')
 def test_storage_enospc_invalidates_same_run(self):
  raw,roots,env,mount,docker=self._storage()
  with raw,patch.dict(os.environ,env,clear=True),patch('fine_defect_ad.storage._mount',side_effect=mount),patch('fine_defect_ad.runtime.discover_docker_storage',return_value=docker):
   proof=preflight(run_id='r',allocations=[Allocation('artifact',1,'persistent','x','x')],reserve_bytes=0,reserve_evidence={'max_pending_atomic_write_bytes':0,'measured_high_water_bytes':0,'runtime_or_source_citation':'test'})
   real_replace=os.replace
   def full_on_material(src,dst):
    if str(src).endswith('.partial'): raise OSError(errno.ENOSPC,'full')
    return real_replace(src,dst)
   with patch('fine_defect_ad.storage.os.replace',side_effect=full_on_material):
    self.assertEqual(atomic_write(roots['artifact']/'x',b'ok',proof=proof,run_id='r')['status'],'INVALIDATED')
   with self.assertRaises(Exception): require_proof(proof,run_id='r')
 def test_docker_discovery_mismatch_and_cdi_failure(self):
  with patch('fine_defect_ad.runtime._run',return_value={'command':['docker','info'],'returncode':0,'stdout':'{"DockerRootDir":"/d","Driver":"overlayfs","DriverStatus":[["driver-type","io.containerd.snapshotter.v1"],["Backing Filesystem","ext4"]]}','stderr':''}):
   found=discover_docker_storage(); self.assertEqual(found['status'],'READY'); self.assertEqual(found['driver_type'],'io.containerd.snapshotter.v1')
  with patch('fine_defect_ad.runtime._run',return_value={'command':['docker','run'],'returncode':1,'stdout':'','stderr':'CDI device error'}):
   self.assertEqual(container_gpu_spike('approved:image')['reason'],'DOCKER_CDI_GPU_UNAVAILABLE')
  self.assertEqual(collect_runtime_evidence()['status'],'STOPPED_INCOMPLETE')
if __name__=='__main__': unittest.main()
