"""One-shot-free TensorRT backend A/B TESTpub evaluation; never calibrates or selects."""
from __future__ import annotations
import argparse, json, subprocess, time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from .tensorrt_promotion import IMAGE, MODEL_NAME, _combined, _cleanup_model_repository, _prepare_model_repository
from .triton_promotion import OUTPUT_NAMES, binary_infer
from .g002_testpub_runtime import GOOD_COUNT, BAD_COUNT, test_public_entries
from .evaluation_history import _image_auroc

READY='READY'; UNAVAILABLE='INSPECTION_UNAVAILABLE'
@dataclass(frozen=True)
class Args:
    artifact_root:Path; checkpoint:Path; metrics:Path; final_attempt:Path; training_identity:Path; dataset_root:Path; teacher_small:Path; imagenette_root:Path; lease_directory:Path; run_id:str; plan:Path; split_freeze:Path; evaluator:Path; http_port:int=18000

def parse_args(argv:Sequence[str]|None=None)->Args:
    from .g002_e2_split_runner import add_common_arguments
    p=argparse.ArgumentParser(description=__doc__); add_common_arguments(p)
    for key in ('plan','split-freeze','evaluator'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--http-port',type=int,default=18000); return Args(**vars(p.parse_args(argv)))

def _hash(path:Path)->str:return sha256(path.read_bytes()).hexdigest()
def _write(path:Path,data:bytes,proof:Any,run_id:str,writer:Callable[...,Mapping[str,Any]]):
    from .storage import READY as STORED
    if writer(path,data,proof=proof,run_id=run_id,overwrite=False).get('status')!=STORED or _hash(path)!=sha256(data).hexdigest(): raise RuntimeError('IMMUTABLE_WRITE_FAILED')

def _attempt_binding(*, checkpoint_sha256: str, split_freeze_sha256: str, plan_sha256: str) -> dict[str, str]:
    return {'checkpoint_sha256': checkpoint_sha256, 'split_freeze_sha256': split_freeze_sha256, 'plan_sha256': plan_sha256}

def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise RuntimeError('TESTPUB_EVIDENCE_INVALID')
    return value

def _existing_testpub_binding(root: Path, run_id: str) -> tuple[dict[str, str] | None, list[Path]]:
    evidence = list(root.glob(f'tensorrt-testpub-attempt-{run_id}-*.json')) + list(root.glob(f'tensorrt-testpub-manifest-{run_id}-*.json'))
    binding = None
    for path in evidence:
        value = _read_json(path); candidate = value.get('binding')
        if not isinstance(candidate, dict): raise RuntimeError('TESTPUB_EVIDENCE_BINDING_UNAVAILABLE')
        candidate = {key: candidate.get(key) for key in ('checkpoint_sha256','split_freeze_sha256','plan_sha256')}
        if not all(isinstance(value, str) and value for value in candidate.values()): raise RuntimeError('TESTPUB_EVIDENCE_BINDING_UNAVAILABLE')
        if binding is not None and candidate != binding: raise RuntimeError('TESTPUB_EVIDENCE_BINDING_CONFLICT')
        binding = candidate
    # Raw-map-only evidence is deliberately non-recoverable: it cannot prove a binding.
    if not evidence and list(root.glob(f'tensorrt-testpub-raw-*.bin')): raise RuntimeError('TESTPUB_EVIDENCE_BINDING_UNAVAILABLE')
    return binding, evidence

def _recover_legacy_authorized(root: Path, args: Args, binding: dict[str, str] | None, evidence: list[Path]) -> dict[str, Any] | None:
    if args.run_id != 'tensorrt-testpub-ab-20260810a' or binding is None or not any('manifest' in path.name for path in evidence): return None
    manifests = [path for path in evidence if 'manifest' in path.name]
    manifest = _read_json(manifests[0]); rows = manifest.get('maps')
    if not isinstance(rows, list) or len(rows) != GOOD_COUNT + BAD_COUNT: raise RuntimeError('LEGACY_TESTPUB_RECOVERY_INVALID')
    for row in rows:
        digest = row.get('map_sha256') if isinstance(row, dict) else None
        matches = list(root.glob(f'tensorrt-testpub-raw-*-{digest}.bin')) if isinstance(digest, str) else []
        if len(matches) != 1 or _hash(matches[0]) != digest: raise RuntimeError('LEGACY_TESTPUB_RECOVERY_INVALID')
    return {'status': READY, 'promotion_eligible': False, 'adoption_recommendation': 'METRICS_REPORTED_REVIEW_REQUIRED', 'recovery': 'READ_ONLY_PERSISTED_MAPS_MANIFEST', 'initial_attempt_latch': 'NOT_AVAILABLE_LEGACY_AUTHORIZED_COMPARISON', 'manifest_sha256': _hash(manifests[0]), 'counts': {'total': len(rows)}}

def _establish_attempt_latch(root: Path, args: Args, binding: dict[str, str], proof: Any, writer: Callable[..., Mapping[str, Any]]) -> str:
    existing, _ = _existing_testpub_binding(root, args.run_id)
    if existing is not None:
        if existing != binding: raise RuntimeError('TESTPUB_ATTEMPT_BINDING_MISMATCH')
        return 'EXISTING_BLOCKED'
    payload = json.dumps({'status':'TENSORRT_TESTPUB_INITIAL_ATTEMPT_LATCH','run_id':args.run_id,'binding':binding}, sort_keys=True, separators=(',',':')).encode()
    _write(root / f'tensorrt-testpub-attempt-{args.run_id}-{sha256(payload).hexdigest()}.json', payload, proof, args.run_id, writer)
    return 'NEW'

def raw_manifest(*, run_id:str, checkpoint_sha256:str, split_freeze_sha256:str, plan_sha256:str, rows:list[Mapping[str,Any]], total_seconds:float)->dict[str,Any]:
    """Canonical host-path-free backend A/B raw-map manifest."""
    if len(rows) != GOOD_COUNT + BAD_COUNT: raise ValueError('exact 114 TESTpub map rows required')
    public=[dict(row) for row in rows]
    if any('source' in row or 'path' in row for row in public): raise ValueError('manifest must not contain host paths')
    return {'status':'TENSORRT_E2_SPLIT_TEST_PUBLIC_RAW_MAPS','run_id':run_id,'protocol':'BACKEND_AB_EVALUATION_NO_TUNING_OR_RECALIBRATION','binding':{'checkpoint_sha256':checkpoint_sha256,'split_freeze_sha256':split_freeze_sha256,'plan_sha256':plan_sha256},'maps':public,'timing':{'total_seconds':total_seconds,'per_image_seconds':[r['seconds'] for r in public]}}

def _metrics(maps:list[Any],entries:list[Mapping[str,Any]],evaluator:Path)->dict[str,Any]:
    import numpy as np
    from anomalib.data.utils.image import read_mask
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import Resize
    from .mvtec_aupro import local_au_pro_0_05
    shape=maps[0].shape; resize=Resize(shape,interpolation=InterpolationMode.NEAREST)
    masks=[None if e['mask'] is None else np.asarray(resize(read_mask(e['mask'],as_tensor=True))).squeeze() for e in entries]
    stats={e['image_identity']:{'label':e['label'],'max':float(np.asarray(m).max())} for e,m in zip(entries,maps)}
    return {'local_au_pro':local_au_pro_0_05(maps,masks,evaluator,include_curve=False),'image_auroc':_image_auroc(stats,normal_label='good',anomaly_label='bad')}

def run(args:Args,*, lease_factory:Callable[...,Any]=None, popen:Callable[...,Any]=subprocess.Popen, runner:Callable[...,Any]=subprocess.run, client_factory:Callable[...,Any]=None, metric_fn:Callable[...,dict[str,Any]]=_metrics, entries_fn:Callable[...,list[dict[str,Any]]]=test_public_entries)->dict[str,Any]:
    """Runs all 114 public inputs once, bound by an immutable pre-decode latch."""
    import numpy as np, torch
    from urllib.request import urlopen
    from .gpu_lock import GpuLease
    from .storage import Allocation, atomic_write, preflight
    from .g002_e2_split_runner import _model, verify_split_lineage
    from .g002_e2_runtime import SPLIT_TARGET_SHAPE, decode_rgb01, verify_split_freeze
    root=Path(args.artifact_root).resolve(); plan=Path(args.plan).resolve(); freeze_path=Path(args.split_freeze).resolve()
    try:
        if not root.is_dir() or not plan.is_file() or not freeze_path.is_file() or not Path(args.evaluator).is_file(): raise ValueError('required input missing')
        freeze=json.loads(freeze_path.read_text()); verify_split_freeze(freeze)
        raw_bytes=(GOOD_COUNT+BAD_COUNT)*SPLIT_TARGET_SHAPE[0]*SPLIT_TARGET_SHAPE[1]*4; manifest_bytes=2*1024*1024; source=f'TensorRT TESTpub raw maps and evidence for {GOOD_COUNT+BAD_COUNT} images'; proof=preflight(run_id=args.run_id,allocations=[Allocation('artifact',plan.stat().st_size+raw_bytes+manifest_bytes,'persistent',source,'trt-testpub'),Allocation('artifact',max(SPLIT_TARGET_SHAPE[0]*SPLIT_TARGET_SHAPE[1]*4,manifest_bytes),'transient',source,'trt-testpub-incoming')],reserve_bytes=max(SPLIT_TARGET_SHAPE[0]*SPLIT_TARGET_SHAPE[1]*4,manifest_bytes),reserve_evidence={'max_pending_atomic_write_bytes':max(SPLIT_TARGET_SHAPE[0]*SPLIT_TARGET_SHAPE[1]*4,manifest_bytes),'measured_high_water_bytes':0,'runtime_or_source_citation':source})
        existing_binding, existing_evidence = _existing_testpub_binding(root, args.run_id)
        recovered = _recover_legacy_authorized(root, args, existing_binding, existing_evidence)
        if recovered is not None: return recovered
        binding = _attempt_binding(checkpoint_sha256=_hash(Path(args.checkpoint).resolve()), split_freeze_sha256=_hash(freeze_path), plan_sha256=_hash(plan))
        if _establish_attempt_latch(root, args, binding, proof, atomic_write) != 'NEW': raise RuntimeError('TESTPUB_ATTEMPT_ALREADY_EXISTS')
        entries=entries_fn(args.dataset_root)
        if len(entries)!=GOOD_COUNT+BAD_COUNT: raise ValueError('exact 114 TESTpub identities required')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA_UNAVAILABLE')
        repo: Path | None = None
        lease=lease_factory or GpuLease; name=f'fine-defect-trt-testpub-{args.run_id}'; server=None; logs=''; rows=[]; maps=[]; result={}
        with lease(args.lease_directory,args.run_id,'tensorrt-testpub-backend-ab'):
            admitted,model=_model(args,torch); verify_split_lineage(freeze,admitted)
            if admitted.checkpoint_sha256 != binding['checkpoint_sha256']: raise RuntimeError('TESTPUB_CHECKPOINT_BINDING_MISMATCH')
            repo, _, _ = _prepare_model_repository(root, plan, proof=proof, run_id=args.run_id, prefix='tensorrt-testpub')
            device=torch.device('cuda:0'); endpoint=f'http://127.0.0.1:{args.http_port}'
            try:
                server=popen(['docker','run','--rm','--gpus','all','--name',name,'--network','host','--log-driver=none','-v',f'{repo}:/models:ro',IMAGE,'tritonserver','--model-repository=/models',f'--http-port={args.http_port}','--log-info=false'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for _ in range(120):
                    if server.poll() is not None: raise RuntimeError('TRITON_EXITED')
                    try:
                        if urlopen(endpoint+'/v2/health/ready',timeout=.5).status==200: break
                    except Exception: time.sleep(.25)
                else: raise RuntimeError('TRITON_READY_TIMEOUT')
                if client_factory is None: client_factory=__import__('tritonclient.http',fromlist=['InferenceServerClient']).InferenceServerClient
                client=client_factory(url=f'127.0.0.1:{args.http_port}')
                def infer(batch:Any): return binary_infer(client,batch.detach().cpu().numpy().astype(np.float32,copy=False))[0]
                total_started=time.perf_counter()
                for index,entry in enumerate(entries):
                    began=time.perf_counter(); value=_combined(decode_rgb01(entry['source']),infer,freeze,torch,device); raw=np.asarray(value,dtype='<f4').tobytes(); digest=sha256(raw).hexdigest(); _write(root/f'tensorrt-testpub-raw-{index:03d}-{digest}.bin',raw,proof,args.run_id,atomic_write)
                    maps.append(value); rows.append({'image_identity':entry['image_identity'],'label':entry['label'],'source_sha256':entry['source_sha256'],'mask_sha256':entry['mask_sha256'],'map_sha256':digest,'dtype':'<f4','byte_order':'<','shape':list(np.asarray(value).shape),'seconds':time.perf_counter()-began})
                metrics=metric_fn(maps,entries,Path(args.evaluator)); manifest=raw_manifest(run_id=args.run_id,checkpoint_sha256=admitted.checkpoint_sha256,split_freeze_sha256=_hash(freeze_path),plan_sha256=_hash(plan),rows=rows,total_seconds=time.perf_counter()-total_started)
                payload=json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode(); manifest_path=root/f'tensorrt-testpub-manifest-{args.run_id}-{sha256(payload).hexdigest()}.json'; _write(manifest_path,payload,proof,args.run_id,atomic_write)
                result={'status':READY,'promotion_eligible':False,'adoption_recommendation':'METRICS_REPORTED_REVIEW_REQUIRED','metrics':metrics,'counts':{'good':GOOD_COUNT,'bad':BAD_COUNT,'total':len(rows)},'manifest_sha256':sha256(payload).hexdigest(),'timing':manifest['timing']}
            finally:
                try:
                    if server is not None:
                        stopped=runner(['docker','stop','-t','10',name],text=True,capture_output=True,check=False)
                        if stopped.returncode:
                            removed=runner(['docker','rm','-f',name],text=True,capture_output=True,check=False)
                            inspected=runner(['docker','inspect',name],text=True,capture_output=True,check=False)
                            if removed.returncode or inspected.returncode == 0: raise RuntimeError('TRITON_CLEANUP_FAILED')
                        try: logs,_=server.communicate(timeout=15)
                        except Exception: server.kill(); logs,_=server.communicate()
                finally:
                    _cleanup_model_repository(repo)
        if logs: result['server_log_sha256']=sha256(logs.encode()).hexdigest()
        return result
    except Exception as exc:
        return {'status':UNAVAILABLE,'promotion_eligible':False,'cause':f'TRT_TESTPUB:{type(exc).__name__}:{sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()[:16]}'}

def main(argv:Sequence[str]|None=None)->int:
    result=run(parse_args(argv)); print(json.dumps(result,sort_keys=True)); return 0 if result['status']==READY else 2
if __name__=='__main__':raise SystemExit(main())
