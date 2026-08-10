"""Fail-closed TensorRT Triton candidate launcher (never publishes TESTpub)."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, tempfile, time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .triton_promotion import (MODEL_NAME, INPUT_NAME, OUTPUT_NAMES, MAX_BATCH_SIZE, TILE_BATCH_SIZE, batch_groups,
    binary_infer, combined_parity, parity_manifest_entries, strict_decision, _error_fingerprint, perf_analyzer_identity)

IMAGE = "nvcr.io/nvidia/tritonserver:26.06-py3@sha256:a40838bb4587d2aceb46b1e7fd144afb24c9016c219dd3eba31716e4e28dbfc7"
UNAVAILABLE = "INSPECTION_UNAVAILABLE"

@dataclass(frozen=True)
class PromotionArgs:
    artifact_root: Path; checkpoint: Path; metrics: Path; final_attempt: Path; training_identity: Path
    dataset_root: Path; teacher_small: Path; imagenette_root: Path; lease_directory: Path; run_id: str
    plan: Path; split_freeze: Path; parity_manifest: Path; calibration_artifact: Path; source_image: Path
    http_port: int; perf_analyzer: Path; perf_wheel_version: str

def _hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()

def _prepare_model_repository(root: Path, plan: Path, *, proof: Any, run_id: str, prefix: str) -> tuple[Path, str, str]:
    """Publish a complete, hash-named ephemeral Triton repository atomically."""
    from .storage import READY, atomic_write
    plan_bytes = Path(plan).read_bytes(); config_bytes = config_pbtxt().encode()
    plan_hash, config_hash = _hash_bytes(plan_bytes), _hash_bytes(config_bytes)
    repo = Path(root) / f"{prefix}-{run_id}-{plan_hash[:16]}-{config_hash[:16]}-repo"
    if repo.exists(): raise RuntimeError("MODEL_REPOSITORY_ALREADY_EXISTS")
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{repo.name}.", suffix=".staging"))
    try:
        version = staging / MODEL_NAME / "1"; version.mkdir(parents=True)
        for target, payload, expected in ((version / "model.plan", plan_bytes, plan_hash), (staging / MODEL_NAME / "config.pbtxt", config_bytes, config_hash)):
            if atomic_write(target, payload, proof=proof, run_id=run_id, overwrite=False).get("status") != READY or _hash_bytes(target.read_bytes()) != expected: raise RuntimeError("MODEL_REPOSITORY_WRITE_FAILED")
        os.rename(staging, repo)
        return repo, plan_hash, config_hash
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise

def _cleanup_model_repository(repo: Path | None) -> None:
    if repo is None: return
    shutil.rmtree(repo, ignore_errors=False)
    if repo.exists(): raise RuntimeError("MODEL_REPOSITORY_CLEANUP_FAILED")

def config_pbtxt() -> str:
    return "\n".join((f'name: "{MODEL_NAME}"', 'platform: "tensorrt_plan"', 'max_batch_size: 8',
        f'input [{{ name: "{INPUT_NAME}" data_type: TYPE_FP32 dims: [3, 256, 256] }}]',
        f'output [{{ name: "{OUTPUT_NAMES[0]}" data_type: TYPE_FP32 dims: [1, 256, 256] }}]',
        f'output [{{ name: "{OUTPUT_NAMES[1]}" data_type: TYPE_FP32 dims: [1, 256, 256] }}]',
        'instance_group [{ kind: KIND_GPU count: 1 }]', ''))

def add_promotion_arguments(parser: argparse.ArgumentParser) -> None:
    from .g002_e2_split_runner import add_common_arguments
    add_common_arguments(parser)
    for name in ('plan','split-freeze','parity-manifest','calibration-artifact','source-image','perf-analyzer'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--http-port', type=int, default=18000); parser.add_argument('--perf-wheel-version', required=True)

def parse_args(argv: Sequence[str] | None = None) -> PromotionArgs:
    parser=argparse.ArgumentParser(description=__doc__); add_promotion_arguments(parser); return PromotionArgs(**vars(parser.parse_args(argv)))

def path_free_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively remove host path-bearing fields from persisted evidence."""
    forbidden={"path","source","artifact","plan"}
    def clean(item: Any) -> Any:
        if isinstance(item, Mapping): return {key:clean(val) for key,val in item.items() if key not in forbidden}
        if isinstance(item, list): return [clean(v) for v in item]
        return item
    return clean(value)

def _admit_source(args: PromotionArgs) -> Path:
    source=Path(args.source_image).resolve(); category=Path(args.dataset_root).resolve()/"sheet_metal"
    allowed=tuple((category/s).resolve() for s in ("train","validation"))
    if not source.is_file() or not any(source.is_relative_to(root) for root in allowed): raise ValueError("source image must be a train or validation image")
    return source

def _preflight(args: PromotionArgs):
    from .storage import Allocation, preflight
    root=Path(args.artifact_root).resolve(); plan=Path(args.plan).resolve(); freeze=Path(args.split_freeze).resolve()
    if not root.is_dir() or not plan.is_file() or not freeze.is_file(): raise ValueError("artifact root, plan, and split freeze must exist")
    size=plan.stat().st_size + len(config_pbtxt().encode()) + 2_097_152
    return preflight(run_id=args.run_id, allocations=[Allocation("artifact",size,"persistent","TensorRT model repo and immutable evidence","trt-candidate"), Allocation("artifact",size,"transient","TensorRT model repo and immutable evidence","trt-candidate-incoming")], reserve_bytes=size, reserve_evidence={"max_pending_atomic_write_bytes":size,"measured_high_water_bytes":0,"runtime_or_source_citation":"TensorRT candidate repository"})

def perf_analyzer_commands(executable: Path, endpoint: str, output_root: Path) -> list[list[str]]:
    base=[str(executable),'-m',MODEL_NAME,'-i','http','-u',endpoint.removeprefix('http://'),'--input-data','zero','--input-tensor-format','binary','--output-tensor-format','binary','--shape',f'{INPUT_NAME}:1,3,256,256','--measurement-mode','time_windows','--measurement-interval','5000','--max-trials','5','--warmup-request-count','5','--percentile','99']
    return [base+['--concurrency-range',f'{c}:{c}:1','-f',str(output_root/f'perf-c{c}.csv')] for c in (1,2,4)]

def _combined(full: Any, mapper: Callable[[Any], tuple[Any,Any]], freeze: Mapping[str,Any], torch: Any, device: Any) -> Any:
    import numpy as np
    from .g002_e2_runtime import _split_boxes, canonical_256, combine_split_maps, periodic_hann_weights
    h,w=full.shape[:2]; sums=np.zeros((h,w),np.float64); weights=np.zeros((h,w),np.float64)
    for group in batch_groups(_split_boxes((h,w)), TILE_BATCH_SIZE):
        tiles=np.stack([np.ascontiguousarray(full[y:y2,x:x2].transpose(2,0,1)) for y,x,y2,x2 in group]); first,_=mapper(torch.from_numpy(tiles).to(device))
        for i,box in enumerate(group):
            y,x,y2,x2=box; weight=periodic_hann_weights(box,(h,w)); sums[y:y2,x:x2]+=np.asarray(first)[i,0]*weight; weights[y:y2,x:x2]+=weight
    _,global_second=mapper(canonical_256(full,torch,device=device))
    return np.asarray(combine_split_maps((sums/weights).astype('<f4'),np.asarray(global_second)[0,0],freeze['quantiles'],torch))

def _live(args: PromotionArgs, proof: Any, *, popen: Callable[...,Any], runner: Callable[...,Any], client_factory: Callable[...,Any] | None) -> dict[str,Any]:
    import numpy as np, torch
    from urllib.request import urlopen
    from .g002_e2_split_runner import _model, verify_split_lineage
    from .g002_e2_runtime import _split_boxes, decode_rgb01, verify_split_freeze
    from .storage import READY, atomic_write
    root=Path(proof.roots['artifact']).resolve(); repo: Path | None = None
    plan=Path(args.plan).resolve()
    freeze=json.loads(Path(args.split_freeze).read_text()); verify_split_freeze(freeze); source=_admit_source(args); entries=parity_manifest_entries(args.parity_manifest,args.dataset_root); admitted,model=_model(args,torch); verify_split_lineage(freeze,admitted); from .split_calibration import load_calibration_artifact; threshold, calibration_sha256=load_calibration_artifact(args.calibration_artifact,split_freeze=args.split_freeze,checkpoint_sha256=admitted.checkpoint_sha256); device=torch.device('cuda:0')
    endpoint=f'http://127.0.0.1:{args.http_port}'; name=f'fine-defect-trt-{args.run_id}'; server=None; logs=''; result: dict[str,Any]={}
    try:
        repo, plan_hash, config_hash = _prepare_model_repository(root, plan, proof=proof, run_id=args.run_id, prefix='tensorrt')
        target=repo/MODEL_NAME/'1'/'model.plan'; config=repo/MODEL_NAME/'config.pbtxt'
        server=popen(['docker','run','--rm','--gpus','all','--name',name,'--network','host','--log-driver=none','-v',f'{repo}:/models:ro',IMAGE,'tritonserver','--model-repository=/models',f'--http-port={args.http_port}','--log-info=false'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for _ in range(120):
            if server.poll() is not None: raise RuntimeError('TRITON_EXITED')
            try:
                if urlopen(endpoint+'/v2/health/ready',timeout=.5).status==200: break
            except Exception: time.sleep(.25)
        else: raise RuntimeError('TRITON_READY_TIMEOUT')
        if client_factory is None: client_factory=__import__('tritonclient.http',fromlist=['InferenceServerClient']).InferenceServerClient
        client=client_factory(url=f'127.0.0.1:{args.http_port}')
        torch.cuda.reset_peak_memory_stats(device)
        def triton(batch: Any):
            return binary_infer(client,batch.detach().cpu().numpy().astype(np.float32,copy=False))[0]
        def eager(batch: Any):
            with torch.inference_mode(): return tuple(x.detach().cpu().numpy() for x in model.model.get_maps(batch,normalize=False))
        eager_maps={}; triton_maps={}
        for entry in entries:
            image=decode_rgb01(entry['source']); eager_maps[entry['path']]=_combined(image,eager,freeze,torch,device); triton_maps[entry['path']]=_combined(image,triton,freeze,torch,device)
        parity=combined_parity(entries,eager=lambda e:eager_maps[e['path']],triton=lambda e:triton_maps[e['path']],threshold=threshold)
        if parity['status']!='PARITY_PASS': raise RuntimeError('FINAL_E2_SPLIT_PARITY_FAILED')
        perf_identity=perf_analyzer_identity(Path(args.perf_analyzer),runner=runner); perf=[]
        for cmd in perf_analyzer_commands(args.perf_analyzer,endpoint,root):
            completed=runner(cmd,text=True,capture_output=True,check=False); csv=Path(cmd[cmd.index('-f')+1])
            if completed.returncode or not csv.is_file() or not csv.read_bytes(): raise RuntimeError('PERF_ANALYZER_FAILED')
            perf.append({'concurrency':int(cmd[cmd.index('--concurrency-range')+1].split(':')[0]),'csv_sha256':sha256(csv.read_bytes()).hexdigest(),'stdout_sha256':sha256(completed.stdout.encode()).hexdigest(),'stderr_sha256':sha256(completed.stderr.encode()).hexdigest()})
        source_started=time.perf_counter(); source_rgb=decode_rgb01(source); source_boxes=_split_boxes(source_rgb.shape[:2]); combined=_combined(source_rgb,triton,freeze,torch,device); decision={k:v for k,v in strict_decision(combined,threshold).items() if k!='mask'}
        result={'status':'READY','promotion_eligible':False,'promotion_reason':'TESTPUB_METRICS_DEFERRED','model_sha256':sha256(target.read_bytes()).hexdigest(),'config_sha256':sha256(config.read_bytes()).hexdigest(),'parity':parity,'perf_analyzer':{**perf_identity,'package_version':args.perf_wheel_version,'runs':perf},'source_e2e':{'total_seconds':time.perf_counter()-source_started,'tile_count':len(source_boxes),'tile_batch_size':TILE_BATCH_SIZE,'triton_call_count':len(batch_groups(source_boxes,TILE_BATCH_SIZE))+1,'triton_transport':'tritonclient.http.binary','raw_map_sha256':sha256(combined.tobytes()).hexdigest(),'decision':decision},'gpu_peak_since_server_ready_bytes':{'allocated':int(torch.cuda.max_memory_allocated()),'reserved':int(torch.cuda.max_memory_reserved())}}
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
            if logs: result['server_log_sha256']=sha256(logs.encode()).hexdigest()
        finally:
            _cleanup_model_repository(repo)
    return result

def run_promotion(args: PromotionArgs, *, lease_factory: Callable[...,Any]=None, runner: Callable[...,Any]=subprocess.run, popen: Callable[...,Any]=subprocess.Popen, client_factory: Callable[...,Any]=None) -> dict[str,Any]:
    from .gpu_lock import GpuLease
    from .storage import READY, atomic_write
    import torch
    try:
        source=_admit_source(args); entries=parity_manifest_entries(args.parity_manifest,args.dataset_root); proof=_preflight(args)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA_UNAVAILABLE')
        factory=lease_factory or GpuLease
        with factory(args.lease_directory,args.run_id,'tensorrt-triton-candidate'):
            live=_live(args,proof,popen=popen,runner=runner,client_factory=client_factory)
        result={'status':live['status'],'promotion_eligible':False,'reason':'TESTPUB_METRICS_DEFERRED', 'binding':{'plan_sha256':sha256(Path(args.plan).read_bytes()).hexdigest(),'split_freeze_sha256':sha256(Path(args.split_freeze).read_bytes()).hexdigest(),'calibration_artifact_sha256':sha256(Path(args.calibration_artifact).read_bytes()).hexdigest(),'parity_manifest_sha256':sha256(Path(args.parity_manifest).read_bytes()).hexdigest(),'source_image_sha256':sha256(source.read_bytes()).hexdigest(),'parity_image_count':len(entries),'image':IMAGE},'steps':live}
    except Exception as exc:
        result={'status':UNAVAILABLE,'promotion_eligible':False,'cause':f'TRT:{type(exc).__name__}:{_error_fingerprint(exc)}'}
    # Persist only immutable, path-free evidence after the lease exits.
    root=Path(args.artifact_root).resolve(); raw=json.dumps(path_free_evidence(result),sort_keys=True,separators=(',',':')).encode(); dest=root/f'tensorrt-promotion-{args.run_id}-{sha256(raw).hexdigest()}.json'
    try:
        proof_locals=locals(); proof=proof_locals.get('proof')
        if proof is None or atomic_write(dest,raw,proof=proof,run_id=args.run_id,overwrite=False).get('status')!=READY: return {'status':UNAVAILABLE,'promotion_eligible':False,'cause':'EVIDENCE_WRITE_FAILED'}
        return {**result,'artifact_sha256':sha256(raw).hexdigest()}
    except Exception: return {'status':UNAVAILABLE,'promotion_eligible':False,'cause':'EVIDENCE_WRITE_FAILED'}

def main(argv: Sequence[str]|None=None)->int:
    result=path_free_evidence(run_promotion(parse_args(argv)))
    print(json.dumps(result,sort_keys=True)); return 0 if result.get('status') == 'READY' else 2
if __name__=='__main__': raise SystemExit(main())
