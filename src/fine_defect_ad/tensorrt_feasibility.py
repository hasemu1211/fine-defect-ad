"""Fail-closed FP32 TensorRT feasibility probe; never a serving promotion."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

from .gpu_lock import GpuLease
from .storage import Allocation, READY, PreflightProof, atomic_write, preflight
from .triton_promotion import (INPUT_SHAPE, MAX_BATCH_SIZE, _error_fingerprint,
                               IMAGE, combined_parity, fixed_256_adapter, numerical_stage)

TRTEXEC = "/usr/bin/trtexec"


@dataclass(frozen=True)
class TensorRTArgs:
    artifact_root: Path
    onnx_path: Path
    plan_path: Path
    input_path: Path
    output_path: Path
    parity_manifest: Path
    dataset_root: Path
    threshold: float
    run_id: str
    trtexec: Path = Path(TRTEXEC)


def _container_paths(args: TensorRTArgs) -> dict[str, str]:
    """Only producer-owned direct children may cross the fixed container mount."""
    root = Path(args.artifact_root).resolve()
    names = {"onnx": args.onnx_path, "plan": args.plan_path, "input": args.input_path, "output": args.output_path}
    paths = {key: Path(value).resolve() for key, value in names.items()}
    if not root.is_dir() or any(path.parent != root for path in paths.values()):
        raise ValueError("TensorRT artifacts must be direct children of artifact root")
    return {key: f"/work/{path.name}" for key, path in paths.items()}


def _trtexec_container(args: TensorRTArgs, flags: list[str]) -> list[str]:
    # Do not pass host paths (or a host trtexec binary) across this boundary.
    root = Path(args.artifact_root).resolve()
    _container_paths(args)
    return ["docker", "run", "--rm", "--gpus", "all", "-v", f"{root}:/work", IMAGE, TRTEXEC, *flags]


def trtexec_command(args: TensorRTArgs) -> list[str]:
    """Pinned-container FP32 engine build for the reviewed dynamic batch envelope."""
    paths = _container_paths(args)
    return _trtexec_container(args, [f"--onnx={paths['onnx']}", f"--saveEngine={paths['plan']}",
            "--minShapes=INPUT__0:1x3x256x256", "--optShapes=INPUT__0:4x3x256x256",
            "--maxShapes=INPUT__0:8x3x256x256"])


def trtexec_compare_command(args: TensorRTArgs) -> list[str]:
    paths = _container_paths(args)
    return _trtexec_container(args, [f"--loadEngine={paths['plan']}", f"--loadInputs=INPUT__0:{paths['input']}",
            f"--exportOutput={paths['output']}", "--shapes=INPUT__0:1x3x256x256"])


def parse_trtexec_output(raw: str | bytes | Path) -> dict[str, Any]:
    """Decode trtexec's JSON output into the two named FP32 raw branches."""
    import numpy as np
    if isinstance(raw, Path): raw = raw.read_bytes()
    if isinstance(raw, bytes): raw = raw.decode("utf-8")
    value = json.loads(raw)
    rows = value.get("outputs", value) if isinstance(value, Mapping) else value
    if isinstance(rows, Mapping):
        rows = [{"name": name, **(item if isinstance(item, Mapping) else {"data": item})}
                for name, item in rows.items()]
    if not isinstance(rows, list): raise ValueError("trtexec output JSON must contain outputs")
    found: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping): continue
        name = row.get("name") or row.get("tensorName")
        data = row.get("data", row.get("values"))
        shape = row.get("shape", row.get("dims", row.get("dimensions")))
        if name not in ("OUTPUT__0", "OUTPUT__1") or data is None: continue
        array = np.asarray(data, dtype=np.float32)
        if shape is not None:
            dimensions = shape.split("x") if isinstance(shape, str) else shape
            if not isinstance(dimensions, (list, tuple)) or not dimensions: raise ValueError("trtexec output dimensions required")
            try: normalized = tuple(int(x) for x in dimensions)
            except (TypeError, ValueError) as exc: raise ValueError("trtexec output dimensions must be positive integers") from exc
            if any(x <= 0 for x in normalized): raise ValueError("trtexec output dimensions must be positive integers")
            array = array.reshape(normalized)
        found[str(name)] = array
    if set(found) != {"OUTPUT__0", "OUTPUT__1"}: raise ValueError("trtexec output missing raw branches")
    return found


def trtexec_benchmark(stdout: str) -> dict[str, float]:
    """Keep stable, numeric timing facts only; absent fields remain unavailable."""
    fields: dict[str, float] = {}
    for label, key in (("Throughput", "throughput_qps"), ("GPU Compute Time", "gpu_compute_ms"),
                       ("Host Latency", "host_latency_ms"), ("Latency", "latency_ms")):
        match = re.search(rf"{re.escape(label)}[^0-9]*([0-9]+(?:\.[0-9]+)?)", stdout, re.I)
        if match: fields[key] = float(match.group(1))
    return fields


def fp32_parity(raw_st: Any, raw_stae: Any, trt_st: Any, trt_stae: Any, *, entries: list[Mapping[str, Any]],
                eager_final: Callable[[Mapping[str, Any]], Any], trt_final: Callable[[Mapping[str, Any]], Any], threshold: float) -> dict[str, Any]:
    """Compatibility helper for deferred final-map analysis; not a feasibility gate."""
    return {"raw": {"st": numerical_stage(raw_st, trt_st), "stae": numerical_stage(raw_stae, trt_stae)},
            "final": combined_parity(entries, eager=eager_final, triton=trt_final, threshold=threshold)}


def export_onnx(model: Any, example: Any, destination: Path) -> None:
    """ONNX opset-18 dynamic-batch export of the reviewed fixed adapter."""
    import torch
    if example.ndim != 4 or tuple(example.shape[1:]) != INPUT_SHAPE[1:]:
        raise ValueError("FP32 Bx3x256x256 example required")
    torch.onnx.export(fixed_256_adapter(model).cpu(), example.detach().cpu(), str(destination),
                      input_names=["INPUT__0"], output_names=["OUTPUT__0", "OUTPUT__1"], opset_version=18,
                      dynamo=False, dynamic_axes={"INPUT__0": {0: "batch"}, "OUTPUT__0": {0: "batch"}, "OUTPUT__1": {0: "batch"}})


def export_from_admitted_model(model: Any, source_tensor: Any, onnx_path: Path, *, proof: PreflightProof,
                               run_id: str, writer: Callable[..., Mapping[str, Any]] = atomic_write) -> dict[str, Any]:
    """Proof-bound in-memory ONNX serialization, without retaining source images."""
    if source_tensor.ndim != 4 or tuple(source_tensor.shape[1:]) != INPUT_SHAPE[1:]:
        raise ValueError("admitted canonical Bx3x256x256 tensor required")
    import io
    import torch
    stream = io.BytesIO()
    torch.onnx.export(fixed_256_adapter(model).cpu(), source_tensor.detach().cpu(), stream,
                      input_names=["INPUT__0"], output_names=["OUTPUT__0", "OUTPUT__1"], opset_version=18,
                      dynamo=False, dynamic_axes={"INPUT__0": {0: "batch"}, "OUTPUT__0": {0: "batch"}, "OUTPUT__1": {0: "batch"}})
    payload = stream.getvalue(); outcome = writer(Path(onnx_path), payload, proof=proof, run_id=run_id, overwrite=False)
    if outcome.get("status") != READY or Path(onnx_path).read_bytes() != payload: raise RuntimeError("ONNX_WRITE_FAILED")
    return {"sha256": sha256(payload).hexdigest(), "bytes": len(payload)}


def _write_bytes(path: Path, payload: bytes, *, proof: PreflightProof, run_id: str, writer: Callable[..., Mapping[str, Any]]) -> str:
    outcome = writer(path, payload, proof=proof, run_id=run_id, overwrite=False)
    if outcome.get("status") != READY or path.read_bytes() != payload: raise RuntimeError("ARTIFACT_WRITE_FAILED")
    return sha256(payload).hexdigest()


def run_feasibility(args: TensorRTArgs, *, runner: Callable[..., Any] = subprocess.run,
                    admit: Callable[..., PreflightProof] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write,
                    lease_factory: Callable[..., Any] = GpuLease, eager: tuple[Any, Any] | None = None) -> dict[str, Any]:
    """Build and compare one FP32 engine; FEASIBLE requires valid raw comparison."""
    root, onnx = Path(args.artifact_root).resolve(), Path(args.onnx_path).resolve()
    if not root.is_dir() or not onnx.is_file(): raise ValueError("admitted artifact root and ONNX file required")
    size = max(onnx.stat().st_size * 2, 1_048_576); source = f"TensorRT FP32 plan/input/output/evidence envelope={size}"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", size, "persistent", source, "trt-fp32"), Allocation("artifact", size, "transient", source, "trt-fp32-incoming")], reserve_bytes=size, reserve_evidence={"max_pending_atomic_write_bytes": size, "measured_high_water_bytes": 0, "runtime_or_source_citation": source})
    record: dict[str, Any] = {"run_id": args.run_id, "status": "INSPECTION_UNAVAILABLE", "promotion_eligible": False, "mode": "FP32_ONLY", "final_e2_split": "DEFERRED", "onnx_sha256": sha256(onnx.read_bytes()).hexdigest()}
    try:
        with lease_factory(root, args.run_id, "tensorrt-fp32-feasibility"):
            build = runner(trtexec_command(args), text=True, capture_output=True, check=False)
            if build.returncode: raise RuntimeError("TRTEXEC_BUILD_FAILED")
            compare = runner(trtexec_compare_command(args), text=True, capture_output=True, check=False)
            if compare.returncode: raise RuntimeError("TRTEXEC_COMPARE_FAILED")
            if not Path(args.plan_path).is_file() or not Path(args.output_path).is_file(): raise RuntimeError("TRTEXEC_OUTPUT_MISSING")
            outputs = parse_trtexec_output(Path(args.output_path))
            diagnostics = None if eager is None else {"st": numerical_stage(eager[0], outputs["OUTPUT__0"]), "stae": numerical_stage(eager[1], outputs["OUTPUT__1"])}
            if diagnostics is not None and any(v["status"] != "NUMERICAL_DIAGNOSTIC" for v in diagnostics.values()): raise RuntimeError("TRT_RAW_COMPARISON_INVALID")
            record.update({"status": "FEASIBLE", "plan_sha256": sha256(Path(args.plan_path).read_bytes()).hexdigest(), "output_sha256": sha256(Path(args.output_path).read_bytes()).hexdigest(), "raw_diagnostics": diagnostics, "trtexec_benchmark": {"build": trtexec_benchmark(build.stdout), "infer": trtexec_benchmark(compare.stdout)}})
    except Exception as exc:
        record.update({"exception_type": type(exc).__name__, "exception_fingerprint": _error_fingerprint(exc)})
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode(); path = root / f"tensorrt-feasibility-{args.run_id}-{sha256(raw).hexdigest()}.json"
    outcome = writer(path, raw, proof=proof, run_id=args.run_id, overwrite=False)
    if outcome.get("status") != READY or path.read_bytes() != raw: return {"status": "INSPECTION_UNAVAILABLE", "reason": "EVIDENCE_WRITE_FAILED"}
    return {**record, "artifact": str(path)}


def add_feasibility_arguments(parser: argparse.ArgumentParser) -> None:
    """Use G002's exact admission arguments; this probe adds no parallel config."""
    from .g002_e2_split_runner import add_common_arguments
    add_common_arguments(parser)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--split-freeze", type=Path)  # accepted for operational continuity; intentionally unused
    parser.add_argument("--parity-manifest", type=Path)  # likewise: final E2 parity is deferred
    parser.add_argument("--trtexec", type=Path, default=Path(TRTEXEC))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); add_feasibility_arguments(parser)
    return parser.parse_args(argv)


def _admit_source(args: Any) -> Path:
    source, category = Path(args.source_image).resolve(), (Path(args.dataset_root).resolve() / "sheet_metal")
    allowed = tuple((category / split).resolve() for split in ("train", "validation"))
    if not source.is_file() or not any(source.is_relative_to(root) for root in allowed):
        raise ValueError("source image must be an admitted sheet_metal train or validation image")
    return source


def run_live_feasibility(args: Any, *, runner: Callable[..., Any] = subprocess.run,
                         admit: Callable[..., PreflightProof] = preflight, writer: Callable[..., Mapping[str, Any]] = atomic_write,
                         lease_factory: Callable[..., Any] = GpuLease) -> dict[str, Any]:
    """Single-lease live path: G002 admission, canonical tile, ONNX, FP32 TRT, raw comparison."""
    import numpy as np
    import torch
    from .g002_e2_runtime import canonical_256, decode_rgb01
    from .g002_e2_split_runner import _model
    root, source = Path(args.artifact_root).resolve(), _admit_source(args)
    if not root.is_dir() or not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
    # _model performs checkpoint/metrics/final-attempt/training-identity admission.
    envelope = max(Path(args.checkpoint).stat().st_size * 2, 4 * 3 * 256 * 256 + 2_097_152)
    citation = f"TensorRT FP32 live envelope={envelope}; canonical tile and raw branches held in memory"
    proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", envelope, "persistent", citation, "trt-live"), Allocation("artifact", envelope, "transient", citation, "trt-live-incoming")], reserve_bytes=envelope, reserve_evidence={"max_pending_atomic_write_bytes": envelope, "measured_high_water_bytes": 0, "runtime_or_source_citation": citation})
    stem = f"tensorrt-fp32-{args.run_id}"; paths = TensorRTArgs(root, root / f"{stem}.onnx", root / f"{stem}.plan", root / f"{stem}.input.bin", root / f"{stem}.outputs.json", Path(args.parity_manifest or root / "deferred"), Path(args.dataset_root), 0.0, args.run_id, Path(args.trtexec))
    record: dict[str, Any] = {"run_id": args.run_id, "status": "INSPECTION_UNAVAILABLE", "promotion_eligible": False, "mode": "FP32_ONLY", "final_e2_split": "DEFERRED"}
    try:
        with lease_factory(args.lease_directory, args.run_id, "tensorrt-fp32-feasibility"):
            admitted, model = _model(args, torch)
            tensor = canonical_256(decode_rgb01(source), torch, device=torch.device("cuda:0"))
            with torch.inference_mode(): eager = tuple(value.detach().cpu().numpy().astype(np.float32, copy=False) for value in fixed_256_adapter(model).to("cuda:0")(tensor))
            export_from_admitted_model(model, tensor, paths.onnx_path, proof=proof, run_id=args.run_id, writer=writer)
            _write_bytes(paths.input_path, tensor.detach().cpu().numpy().astype(np.float32, copy=False).tobytes(), proof=proof, run_id=args.run_id, writer=writer)
            build = runner(trtexec_command(paths), text=True, capture_output=True, check=False)
            if build.returncode: raise RuntimeError("TRTEXEC_BUILD_FAILED")
            compare = runner(trtexec_compare_command(paths), text=True, capture_output=True, check=False)
            if compare.returncode: raise RuntimeError("TRTEXEC_COMPARE_FAILED")
            outputs = parse_trtexec_output(paths.output_path)
            diagnostics = {"st": numerical_stage(eager[0], outputs["OUTPUT__0"]), "stae": numerical_stage(eager[1], outputs["OUTPUT__1"])}
            if any(value["status"] != "NUMERICAL_DIAGNOSTIC" for value in diagnostics.values()): raise RuntimeError("TRT_RAW_COMPARISON_INVALID")
            record.update({"status": "FEASIBLE", "checkpoint_sha256": admitted.checkpoint_sha256, "onnx_sha256": sha256(paths.onnx_path.read_bytes()).hexdigest(), "plan_sha256": sha256(paths.plan_path.read_bytes()).hexdigest(), "raw_diagnostics": diagnostics, "trtexec_benchmark": {"build": trtexec_benchmark(build.stdout), "infer": trtexec_benchmark(compare.stdout)}})
    except Exception as exc:
        record.update({"exception_type": type(exc).__name__, "exception_fingerprint": _error_fingerprint(exc)})
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode(); artifact = root / f"tensorrt-feasibility-{args.run_id}-{sha256(raw).hexdigest()}.json"
    if writer(artifact, raw, proof=proof, run_id=args.run_id, overwrite=False).get("status") != READY or artifact.read_bytes() != raw: return {"status": "INSPECTION_UNAVAILABLE", "reason": "EVIDENCE_WRITE_FAILED"}
    return {**record, "artifact": str(artifact)}


def main(argv: list[str] | None = None) -> int:
    result = run_live_feasibility(parse_args(argv)); print(json.dumps(result, sort_keys=True)); return 0 if result.get("status") == "FEASIBLE" else 2


if __name__ == "__main__": raise SystemExit(main())
