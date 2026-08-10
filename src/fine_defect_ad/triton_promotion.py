"""R2 Triton candidate utilities.

This is deliberately a candidate gate, not a promotion: the pinned 26.06 image
has no ONNX backend or Perf Analyzer, so its direct V2 calls are recorded as a
stdlib-client substitute and cannot satisfy the R2 promotion requirement.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.request import Request, urlopen

from .gpu_lock import GpuLease
from .storage import Allocation, PreflightProof, READY, atomic_write, preflight

IMAGE = "nvcr.io/nvidia/tritonserver@sha256:80caf7d0be25520d39c5162cdeec1f6b2febe4ab774d7b25138cd602d624db3a"
MODEL_NAME = "efficientad"
INPUT_NAME = "INPUT__0"
OUTPUT_NAMES = ("OUTPUT__0", "OUTPUT__1")
OUTPUT_SEMANTICS = {"OUTPUT__0": "map_st", "OUTPUT__1": "map_stae"}
INPUT_SHAPE = (1, 3, 256, 256)
INSPECTION_UNAVAILABLE = "INSPECTION_UNAVAILABLE"


@dataclass(frozen=True)
class TritonConfig:
    model_name: str = MODEL_NAME
    input_name: str = INPUT_NAME
    output_names: tuple[str, str] = OUTPUT_NAMES
    input_shape: tuple[int, int, int, int] = INPUT_SHAPE
    max_batch_size: int = 0
    instance_count: int = 1
    dynamic_batching: bool = False

    def __post_init__(self) -> None:
        if self.input_shape != INPUT_SHAPE or self.max_batch_size != 0 or self.instance_count != 1 or self.dynamic_batching:
            raise ValueError("R2 requires fixed FP32 NCHW 1x3x256x256, one GPU instance, and no dynamic batching")
        if len(self.output_names) != 2 or not all(self.output_names):
            raise ValueError("two raw EfficientAD branches are required")


def onnx_fallback_evidence(*, host_onnx_available: bool, image_backends: Iterable[str]) -> dict[str, Any]:
    """Return the auditable export choice; ONNX is never silently substituted."""
    backends = tuple(sorted(set(image_backends)))
    # Triton calls the backend "onnxruntime"; accept neither an image nor host guess.
    image_missing = "onnxruntime" not in backends
    if not (not host_onnx_available and image_missing):
        raise ValueError("TorchScript fallback is only valid after both ONNX gaps are evidenced")
    return {
        "onnx": {"host_module_available": host_onnx_available, "image_backend_available": not image_missing, "status": "UNAVAILABLE"},
        "selected_export": "torchscript",
        "selected_backend": "pytorch_libtorch",
        "reason": "host onnx missing and pinned Triton image lacks ONNX Runtime backend",
        "image_backends": list(backends),
        "output_semantics": OUTPUT_SEMANTICS,
    }


def config_pbtxt(config: TritonConfig = TritonConfig()) -> str:
    """Minimal immutable model contract for the PyTorch backend."""
    return "\n".join((
        f'name: "{config.model_name}"', 'platform: "pytorch_libtorch"', 'max_batch_size: 0',
        f'input [{{ name: "{config.input_name}" data_type: TYPE_FP32 dims: [1, 3, 256, 256] }}]',
        f'output [{{ name: "{config.output_names[0]}" data_type: TYPE_FP32 dims: [1, 1, 256, 256] }}]',
        f'output [{{ name: "{config.output_names[1]}" data_type: TYPE_FP32 dims: [1, 1, 256, 256] }}]',
        'instance_group [{ kind: KIND_GPU count: 1 }]', '',
    ))


def raw_maps_source(model: Any) -> Any:
    """Select the EfficientAD core without registering Lightning's trainer wrapper."""
    return getattr(model, "model", model)


def fixed_256_adapter(model: Any) -> Any:
    """Fixed 256px EfficientAD inference graph with literal decoder sizes.

    This mirrors ``get_maps(normalize=False)`` but bypasses the decoder's
    tensor-shape-derived interpolation sizes, which CUDA tracing cannot replay.
    """
    import torch
    import torch.nn.functional as F

    source = raw_maps_source(model)
    def require_small_pdn(network: Any) -> None:
        names = ("conv1", "avgpool1", "conv2", "avgpool2", "conv3", "conv4")
        if (not all(hasattr(network, name) for name in names) or network.conv1.in_channels != 3
                or network.conv1.out_channels != 128 or network.conv2.in_channels != 128
                or network.conv2.out_channels != 256 or network.conv3.in_channels != 256):
            raise ValueError("R2 fixed adapter only accepts EfficientAD-S SmallPatchDescriptionNetwork")
    require_small_pdn(source.teacher); require_small_pdn(source.student)
    teacher_normalized = any(bool(value.detach().sum().item() != 0) for value in source.mean_std.values())
    pad_maps = bool(source.pad_maps)
    last_upsample = (64, 64) if bool(source.ae.decoder.padding) else (56, 56)

    class Fixed256RawMaps(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.source = source
            self.teacher_normalized = teacher_normalized
            self.pad_maps = pad_maps
            # Buffers replace upstream's per-call CPU tensor/device transfer.
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        def forward(self, image: Any) -> tuple[Any, Any]:
            normalized = (image - self.imagenet_mean) / self.imagenet_std
            # Inline EfficientAD-S Small PDN: upstream forward creates CPU
            # constants then transfers them, which gets baked into CPU traces.
            teacher_network = self.source.teacher
            teacher = teacher_network.conv4(F.relu(teacher_network.conv3(teacher_network.avgpool2(F.relu(teacher_network.conv2(teacher_network.avgpool1(F.relu(teacher_network.conv1(normalized)))))))))
            if self.teacher_normalized:
                teacher = (teacher - self.source.mean_std["mean"]) / self.source.mean_std["std"]
            student_network = self.source.student
            student = student_network.conv4(F.relu(student_network.conv3(student_network.avgpool2(F.relu(student_network.conv2(student_network.avgpool1(F.relu(student_network.conv1(normalized)))))))))
            distance_st = torch.pow(teacher - student[:, : self.source.teacher_out_channels, :, :], 2)

            ae = self.source.ae
            decoded = ae.encoder(normalized)
            decoder = ae.decoder
            decoded = F.interpolate(decoded, size=(3, 3), mode="bilinear")
            decoded = decoder.dropout1(F.relu(decoder.deconv1(decoded)))
            decoded = F.interpolate(decoded, size=(8, 8), mode="bilinear")
            decoded = decoder.dropout2(F.relu(decoder.deconv2(decoded)))
            decoded = F.interpolate(decoded, size=(15, 15), mode="bilinear")
            decoded = decoder.dropout3(F.relu(decoder.deconv3(decoded)))
            decoded = F.interpolate(decoded, size=(32, 32), mode="bilinear")
            decoded = decoder.dropout4(F.relu(decoder.deconv4(decoded)))
            decoded = F.interpolate(decoded, size=(63, 63), mode="bilinear")
            decoded = decoder.dropout5(F.relu(decoder.deconv5(decoded)))
            decoded = F.interpolate(decoded, size=(127, 127), mode="bilinear")
            decoded = decoder.dropout6(F.relu(decoder.deconv6(decoded)))
            decoded = F.interpolate(decoded, size=last_upsample, mode="bilinear")
            ae_output = decoder.deconv8(F.relu(decoder.deconv7(decoded)))

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean((ae_output - student[:, self.source.teacher_out_channels :, :, :]) ** 2, dim=1, keepdim=True)
            if self.pad_maps:
                map_st, map_stae = F.pad(map_st, (4, 4, 4, 4)), F.pad(map_stae, (4, 4, 4, 4))
            return F.interpolate(map_st, size=(256, 256), mode="bilinear"), F.interpolate(map_stae, size=(256, 256), mode="bilinear")

    return Fixed256RawMaps().eval()


def _persist_torchscript(payload: bytes, destination: Path, *, proof: PreflightProof, run_id: str,
                         writer: Callable[..., Mapping[str, Any]] = atomic_write) -> dict[str, Any]:
    """Proof-bound persistence for the material model artifact."""
    destination = Path(destination)
    digest = sha256(payload).hexdigest()
    outcome = writer(destination, payload, proof=proof, run_id=run_id, overwrite=False)
    if outcome.get("status") != READY or not destination.is_file() or destination.read_bytes() != payload:
        raise RuntimeError("TorchScript artifact persistence failed")
    return {"path": str(destination), "sha256": digest, "bytes": len(payload)}


def save_torchscript(model: Any, destination: Path, example: Any, *, proof: PreflightProof, run_id: str,
                     writer: Callable[..., Mapping[str, Any]] = atomic_write) -> dict[str, Any]:
    """Trace fixed-size raw branches, then atomically persist under an admitted proof."""
    import torch
    if tuple(example.shape) != INPUT_SHAPE:
        raise ValueError("TorchScript example must be fixed 1x3x256x256")
    # CUDA tracing in torch 2.7 decomposes interpolate into device-mismatched
    # clamp constants. Trace an isolated CPU copy; TorchScript remains movable.
    adapter = copy.deepcopy(fixed_256_adapter(model)).cpu()
    traced = torch.jit.trace(adapter, example.detach().cpu(), check_trace=False)
    serialized = io.BytesIO(); torch.jit.save(traced, serialized)
    return _persist_torchscript(serialized.getvalue(), destination, proof=proof, run_id=run_id, writer=writer)


def parity_report(eager: Iterable[Any], served: Iterable[Any], *, tolerance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare both raw branches using an evidence-backed tolerance envelope."""
    if not isinstance(tolerance, Mapping):
        return {"status": INSPECTION_UNAVAILABLE, "cause": "TOLERANCE_EVIDENCE_REQUIRED", "promotion_eligible": False}
    atol, rtol, provenance = tolerance.get("atol"), tolerance.get("rtol"), tolerance.get("provenance")
    if (not isinstance(atol, (int, float)) or not isinstance(rtol, (int, float)) or atol < 0 or rtol < 0
            or not isinstance(provenance, str) or not provenance):
        return {"status": INSPECTION_UNAVAILABLE, "cause": "INVALID_TOLERANCE_EVIDENCE", "promotion_eligible": False}
    eager, served = tuple(eager), tuple(served)
    if len(eager) != 2 or len(served) != 2:
        return {"status": INSPECTION_UNAVAILABLE, "cause": "MALFORMED_OUTPUT", "promotion_eligible": False}
    def pairs(left: Any, right: Any):
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right): raise ValueError("shape mismatch")
            for item_left, item_right in zip(left, right): yield from pairs(item_left, item_right)
        else: yield float(left), float(right)
    try:
        deltas, equal = [], []
        for left, right in zip(eager, served):
            values = list(pairs(left, right))
            if not values: raise ValueError("empty output")
            deltas.append(max(abs(a - b) for a, b in values))
            equal.append(all(abs(a - b) <= atol + rtol * abs(b) for a, b in values))
    except Exception as exc:
        return {"status": INSPECTION_UNAVAILABLE, "cause": f"PARITY:{type(exc).__name__}:{_error_fingerprint(exc)}", "promotion_eligible": False}
    return {"status": "PARITY_PASS" if all(equal) else INSPECTION_UNAVAILABLE, "max_abs_error": deltas,
            "tolerance": {"atol": atol, "rtol": rtol, "provenance": provenance}, "promotion_eligible": False}


def _error_fingerprint(exc: Exception) -> str:
    return sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()[:16]


def _percentile(values: list[float], percentile: float) -> float:
    if not values: raise ValueError("latencies required")
    ordered = sorted(values); index = (len(ordered) - 1) * percentile / 100
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _infer_v2(endpoint: str, tensor: list[float], *, timeout: float) -> dict[str, Any]:
    if len(tensor) != 1 * 3 * 256 * 256:
        raise ValueError("direct V2 tensor must contain exactly 1x3x256x256 FP32 values")
    payload = {"inputs": [{"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "datatype": "FP32", "data": tensor}],
               "outputs": [{"name": name} for name in OUTPUT_NAMES]}
    request = Request(endpoint.rstrip("/") + f"/v2/models/{MODEL_NAME}/infer", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    began = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    return {"seconds": time.perf_counter() - began, "response": value}


def direct_client_benchmark(endpoint: str, tensor: list[float], *, concurrencies: tuple[int, ...] = (1, 2, 4), timeout: float = 30.0, warmup_requests: int = 1, samples_per_concurrency: int = 5, caller: Callable[..., Mapping[str, Any]] = _infer_v2) -> dict[str, Any]:
    """Bounded V2 probe with percentile latency; explicitly not Perf Analyzer."""
    if warmup_requests < 1 or samples_per_concurrency < 1:
        raise ValueError("positive warmup and sample count required")
    rows = []
    for concurrency in concurrencies:
        if concurrency not in (1, 2, 4): raise ValueError("R2 benchmark is fixed to c1/c2/c4")
        for _ in range(warmup_requests): caller(endpoint, tensor, timeout=timeout)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            values = list(pool.map(lambda _: caller(endpoint, tensor, timeout=timeout), range(concurrency * samples_per_concurrency)))
        elapsed = time.perf_counter() - started; latencies = [float(v["seconds"]) for v in values]
        rows.append({"concurrency": concurrency, "requests": len(values), "warmup_requests": warmup_requests,
                     "sample_count": samples_per_concurrency, "elapsed_seconds": elapsed,
                     "latency_seconds": latencies, "p50_seconds": _percentile(latencies, 50),
                     "p95_seconds": _percentile(latencies, 95), "p99_seconds": _percentile(latencies, 99),
                     "requests_per_second": len(values) / elapsed})
    return {"tool": "stdlib_http_direct_client_substitute", "not_perf_analyzer": True, "queue_time": "UNAVAILABLE", "rows": rows,
            "promotion_eligible": False, "limitation": "Perf Analyzer unavailable in pinned image"}


def unavailable_result(cause: str) -> dict[str, Any]:
    return {"status": INSPECTION_UNAVAILABLE, "cause": cause, "verdict": None, "promotion_eligible": False}


# Public, deliberately narrow R2 entrypoint.  Hardware steps remain injected so
# its admission/failure contract is testable without Docker or CUDA.
@dataclass(frozen=True)
class PromotionArgs:
    artifact_root: Path; checkpoint: Path; metrics: Path; final_attempt: Path; training_identity: Path
    dataset_root: Path; teacher_small: Path; imagenette_root: Path; lease_directory: Path
    source_image: Path; split_freeze: Path; run_id: str; http_port: int = 18000


def add_promotion_arguments(parser: Any) -> None:
    # Exact reviewed G002 admission fields; R2 adds only source/freeze/port.
    from .g002_e2_split_runner import add_common_arguments
    add_common_arguments(parser)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--split-freeze", type=Path, required=True)
    parser.add_argument("--http-port", type=int, default=18000)


def parse_promotion_args(argv: list[str] | None = None) -> PromotionArgs:
    import argparse
    parser = argparse.ArgumentParser(description="Fail-closed EfficientAD-S R2 Triton candidate runner")
    add_promotion_arguments(parser)
    value = parser.parse_args(argv)
    return PromotionArgs(**vars(value))


def g002_runtime_args(args: PromotionArgs) -> Any:
    """Reuse the reviewed G002 runtime argument contract; no parallel config."""
    from .g002_pilot import G002Args
    return G002Args(Path(args.dataset_root), Path(args.teacher_small), Path(args.imagenette_root), args.run_id, Path(args.lease_directory))


def _admit_source_image(args: PromotionArgs) -> Path:
    category = (Path(args.dataset_root).resolve() / "sheet_metal")
    source = Path(args.source_image).resolve()
    allowed = tuple((category / split).resolve() for split in ("train", "validation"))
    if not source.is_file() or not any(source.is_relative_to(root) for root in allowed):
        raise ValueError("source image must be an existing sheet_metal train or validation image; TEST partitions are forbidden")
    return source


def runtime_probes(*, runner: Callable[..., Any]) -> dict[str, Any]:
    """Evidence the actual host ONNX module and pinned container backend inventory."""
    import importlib.util
    command = ["docker", "run", "--rm", "--entrypoint", "sh", IMAGE, "-c", "find /opt/tritonserver/backends -mindepth 1 -maxdepth 1 -printf '%f\\n' | sort"]
    completed = runner(command, text=True, capture_output=True, check=False)
    backends = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if completed.returncode != 0 or "pytorch" not in backends:
        raise RuntimeError("pinned Triton backend probe did not verify pytorch")
    return {"host_onnx_module": bool(importlib.util.find_spec("onnx")), "image": IMAGE,
            "backend_probe": {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
            "image_backends": backends}


def promotion_preflight(args: PromotionArgs, *, admit: Callable[..., PreflightProof] = preflight) -> PreflightProof:
    """Reserve all known R2 model/config/evidence writes before directory or GPU work."""
    root, checkpoint = Path(args.artifact_root).resolve(), Path(args.checkpoint).resolve()
    if not root.is_dir() or not checkpoint.is_file():
        raise ValueError("existing artifact root and checkpoint are required before R2 admission")
    # Source-backed envelope covers the fixed TorchScript serialization plus
    # config/evidence records without relying on an output-size assumption.
    model_bytes, metadata_bytes = 2 * checkpoint.stat().st_size + 1_048_576, 2_097_152
    source = f"R2 preflight TorchScript envelope={model_bytes}; config/evidence/log bytes={metadata_bytes}"
    return admit(run_id=args.run_id, allocations=[
        Allocation("artifact", model_bytes, "persistent", source, "triton-r2-model"),
        Allocation("artifact", metadata_bytes, "persistent", source, "triton-r2-config-and-evidence"),
        Allocation("artifact", max(model_bytes, metadata_bytes), "transient", source, "triton-r2-atomic-incoming"),
    ], reserve_bytes=max(model_bytes, metadata_bytes), reserve_evidence={"max_pending_atomic_write_bytes": max(model_bytes, metadata_bytes), "measured_high_water_bytes": 0, "runtime_or_source_citation": source})


def _live_steps(args: PromotionArgs, proof: PreflightProof, probes: Mapping[str, Any]) -> Mapping[str, Any]:
    """The single real R2 path: export, Triton, V2 benchmark, and one E2E image."""
    import subprocess, threading
    import numpy as np
    import torch
    from urllib.request import Request, urlopen
    from .g002_e2_split_runner import _model
    from .g002_e2_runtime import _split_boxes, canonical_256, combine_split_maps, decode_rgb01, periodic_hann_weights

    root = Path(proof.roots["artifact"]); repo = root / f"triton-r2-{args.run_id}-model-repo"; version = repo / MODEL_NAME / "1"
    version.mkdir(parents=True, exist_ok=True)  # admission already succeeded
    model_path, config_path = version / "model.pt", repo / MODEL_NAME / "config.pbtxt"
    freeze = json.loads(Path(args.split_freeze).read_text()); source = _admit_source_image(args)
    _admitted, model = _model(args, torch); device = torch.device("cuda:0")
    rgb = decode_rgb01(source); tensor = canonical_256(rgb, torch, device=device)
    with torch.inference_mode(): eager = model.model.get_maps(tensor, normalize=False)
    exported = save_torchscript(model, model_path, tensor, proof=proof, run_id=args.run_id)
    config_raw = config_pbtxt().encode(); outcome = atomic_write(config_path, config_raw, proof=proof, run_id=args.run_id, overwrite=False)
    if outcome.get("status") != READY or config_path.read_bytes() != config_raw: raise RuntimeError("CONFIG_WRITE_FAILED")
    name = f"fine-defect-r2-{args.run_id}"; logs = ""; server = None
    endpoint = f"http://127.0.0.1:{args.http_port}"
    def request(tile: Any) -> tuple[dict[str, Any], float]:
        data = tile.detach().cpu().numpy().astype(np.float32, copy=False).ravel().tolist()
        payload = {"inputs":[{"name":INPUT_NAME,"shape":list(INPUT_SHAPE),"datatype":"FP32","data":data}],"outputs":[{"name":n} for n in OUTPUT_NAMES]}
        began=time.perf_counter(); response=json.loads(urlopen(Request(endpoint+f"/v2/models/{MODEL_NAME}/infer",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"}),timeout=30).read()); elapsed=time.perf_counter()-began
        return {row["name"]:np.asarray(row["data"],dtype=np.float32).reshape(row["shape"]) for row in response["outputs"]}, elapsed
    try:
        server = subprocess.Popen(["docker","run","--rm","--gpus","all","--name",name,"--network","host","--log-driver=none","-v",f"{repo}:/models:ro",IMAGE,"tritonserver","--model-repository=/models",f"--http-port={args.http_port}","--log-info=false"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for _ in range(120):
            if server.poll() is not None: raise RuntimeError("TRITON_EXITED")
            try:
                if urlopen(endpoint+"/v2/health/ready",timeout=.5).status == 200: break
            except Exception: time.sleep(.25)
        else: raise RuntimeError("TRITON_READY_TIMEOUT")
        served, first = request(tensor)
        tolerance={"atol":1e-5,"rtol":1.3e-6,"provenance":"PyTorch torch.testing.assert_close documented FP32 defaults"}
        parity=parity_report(tuple(v.detach().cpu().numpy().tolist() for v in eager),tuple(served[n].tolist() for n in OUTPUT_NAMES),tolerance=tolerance)
        base = {"model": exported, "config_sha256": sha256(config_raw).hexdigest(), "parity": {**parity, "first_request_seconds": first}}
        if parity["status"] != "PARITY_PASS":
            result = {**base, "status": INSPECTION_UNAVAILABLE, "promotion_eligible": False,
                      "cause": "TRITON_PARITY_FAILED", "benchmark": "SKIPPED_PARITY_FAILURE", "original_image_e2e": "SKIPPED_PARITY_FAILURE"}
            return result
        flat=tensor.detach().cpu().numpy().astype(np.float32,copy=False).ravel().tolist()
        benchmark=direct_client_benchmark(endpoint,flat,warmup_requests=2,samples_per_concurrency=10)
        began=time.perf_counter(); full=decode_rgb01(source); h,w=full.shape[:2]; sums=np.zeros((h,w),np.float64); weights=np.zeros((h,w),np.float64); infer=0.0
        boxes=_split_boxes((h,w))
        for y,x,y2,x2 in boxes:
            tile=torch.from_numpy(np.ascontiguousarray(full[y:y2,x:x2].transpose(2,0,1))).unsqueeze(0).to(device); output,elapsed=request(tile); infer += elapsed; weight=periodic_hann_weights((y,x,y2,x2),(h,w)); sums[y:y2,x:x2]+=output[OUTPUT_NAMES[0]][0,0]*weight; weights[y:y2,x:x2]+=weight
        global_output,elapsed=request(canonical_256(full,torch,device=device)); infer += elapsed
        combined=combine_split_maps((sums/weights).astype("<f4"),global_output[OUTPUT_NAMES[1]][0,0],freeze["quantiles"],torch)
        result = {**base,"direct_client_benchmark":benchmark,"original_image_e2e":{"source_shape":[h,w,3],"tile_count":len(boxes),"triton_request_seconds_sum":infer,"total_seconds":time.perf_counter()-began,"raw_map_sha256":sha256(combined.tobytes()).hexdigest(),"threshold":"UNAVAILABLE","result":INSPECTION_UNAVAILABLE}}
    finally:
        if server is not None:
            subprocess.run(["docker","stop","-t","10",name],capture_output=True,text=True)
            try: logs,_=server.communicate(timeout=15)
            except subprocess.TimeoutExpired: server.kill(); logs,_=server.communicate()
        if logs and "result" in locals(): result["server_log_sha256"] = sha256(logs.encode()).hexdigest()
    return result


def run_promotion(args: PromotionArgs, *, steps: Callable[[PromotionArgs, PreflightProof, Mapping[str, Any]], Mapping[str, Any]] | None = None,
                  admit: Callable[..., PreflightProof] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write,
                  lease_factory: Callable[..., Any] = GpuLease, runner: Callable[..., Any]) -> dict[str, Any]:
    """Run one admitted lease window; absent hardware steps fail closed and persist evidence."""
    source = _admit_source_image(args)
    checkpoint, freeze = Path(args.checkpoint).resolve(), Path(args.split_freeze).resolve()
    if not freeze.is_file(): raise ValueError("split freeze is required")
    binding = {"run_id": args.run_id, "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
               "source_image": str(source.relative_to(Path(args.dataset_root).resolve())), "source_image_sha256": sha256(source.read_bytes()).hexdigest(),
               "split_freeze_sha256": sha256(freeze.read_bytes()).hexdigest(), "mode": {"input_shape": list(INPUT_SHAPE), "image": IMAGE, "max_batch_size": 0, "instance_count": 1, "dynamic_batching": False}}
    proof = promotion_preflight(args, admit=admit)  # must precede mkdir, probes, or GPU lease
    root = Path(proof.roots["artifact"]).resolve()
    if root != Path(args.artifact_root).resolve():
        raise ValueError("preflight artifact root changed")
    lease_directory = Path(args.lease_directory).resolve()
    if not lease_directory.is_relative_to(root):
        raise ValueError("lease directory must be under admitted artifact root")
    result: dict[str, Any] = {"run_id": args.run_id, "status": INSPECTION_UNAVAILABLE, "promotion_eligible": False, "stage": "probe", "binding": binding,
                               "limitations": ["direct stdlib client is not Perf Analyzer", "official comparator unavailable: no NORMAL/ANOMALOUS"]}
    try:
        probes = runtime_probes(runner=runner)
        result["runtime_probes"] = probes
        result["export_fallback"] = onnx_fallback_evidence(host_onnx_available=probes["host_onnx_module"], image_backends=probes["image_backends"])
        result["stage"] = "hardware_steps"
        with lease_factory(lease_directory, args.run_id, "triton-r2-promotion"):
            step_result = dict((steps or _live_steps)(args, proof, probes))
            model = step_result.get("model")
            if isinstance(model, Mapping):
                step_result["model"] = {key: model[key] for key in ("sha256", "bytes") if key in model}
            result["steps"] = step_result
    except Exception as exc:
        result["exception"] = {"type": type(exc).__name__, "fingerprint": _error_fingerprint(exc)}
        result.update(unavailable_result(f"R2:{type(exc).__name__}:{_error_fingerprint(exc)}"))
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(); destination = root / f"triton-r2-promotion-{args.run_id}-{sha256(raw).hexdigest()}.json"
    try:
        outcome = writer(destination, raw, proof=proof, run_id=args.run_id, overwrite=False)
        if outcome.get("status") != READY or destination.read_bytes() != raw:
            return unavailable_result("EVIDENCE_WRITE_FAILED")
    except Exception as exc:
        return unavailable_result(f"EVIDENCE:{type(exc).__name__}:{_error_fingerprint(exc)}")
    return {**result, "artifact": str(destination), "artifact_sha256": sha256(raw).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    """Public parser; the full hardware step function is intentionally library-injected."""
    args = parse_promotion_args(argv)
    result = run_promotion(args, runner=__import__("subprocess").run)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
