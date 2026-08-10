"""Public E2-Split high-resolution inference and evaluation entry point.

This emits raw anomaly maps only; it is not an operational deployment or
verdict interface.  It reuses immutable split validation and TESTpub contracts.
"""
from __future__ import annotations

import argparse
import io
import time
from statistics import median
import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from . import g002_e2_split_runner as _runner
from .g002_e2_runtime import SPLIT_DECISION_ID, canonical_256, combine_split_maps, decode_rgb01, split_branch_raw_maps, verify_split_freeze
from .g002_evaluate import _hash, raw_map
from .g002_testpub_runtime import _canon
from .gpu_lock import GpuLease

PUBLIC_PIPELINE = "FineDefect raw-map inference pipeline"
FAILURE_SCHEMA_VERSION = "1.0"
INFER_COMMAND = "highres-split-infer"


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT") or "unknown"


def _input_bindings(args: Any) -> dict[str, str]:
    names = ("checkpoint", "metrics", "final_attempt", "training_identity", "split_freeze", "evaluator", "input_image")
    return {name: _hash(Path(value)) for name in names if (value := getattr(args, name, None)) and Path(value).is_file()}


def _failure_payload(args: Any, exc: Exception) -> bytes:
    return _canon({"schema_version": FAILURE_SCHEMA_VERSION, "public_pipeline": PUBLIC_PIPELINE,
                   "operation": args.operation, "run_id": args.run_id, "decision_id": _decision_id(getattr(args, "mode", "e2-split")),
                   "stage": getattr(args, "_stage", "admission"), "configuration": {"mode": getattr(args, "mode", None), "repeat": getattr(args, "repeat", None)},
                   **_runner.safe_failure_fields(exc), "git_commit": _git_commit(), "input_sha256": _input_bindings(args)})


def _write_failure_record(args: Any, exc: Exception) -> Path:
    payload = _failure_payload(args, exc); digest = sha256(payload).hexdigest()
    return _runner._write(Path(args.artifact_root).resolve(), args.run_id, f"highres-split-FAILED-{args.run_id}-{digest}.json", payload)


def _reject_dataset_input(args: Any) -> Path:
    image, dataset = Path(args.input_image).resolve(), Path(args.dataset_root).resolve()
    try: image.relative_to(dataset)
    except ValueError: return image
    raise ValueError("dataset-root input is not accepted by public infer")


def _heatmap_bytes(value: Any) -> tuple[bytes, dict[str, float]]:
    import numpy as np
    from PIL import Image
    array = np.asarray(value, dtype='<f4').squeeze()
    if array.ndim != 2: raise ValueError("raw anomaly map must be 2D after squeeze")
    low, high = float(array.min()), float(array.max())
    pixels = np.zeros(array.shape, dtype=np.uint8) if high == low else ((array - low) * (255 / (high - low))).clip(0, 255).astype(np.uint8)
    buffer = io.BytesIO(); Image.fromarray(pixels, mode='L').save(buffer, format='PNG')
    return buffer.getvalue(), {"min": low, "max": high}


def _sync(torch: Any) -> None:
    sync = getattr(getattr(torch, "cuda", None), "synchronize", None)
    if sync: sync()


def _e1_map(rgb: Any, model: Any, torch: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    st, stae = getattr(model, "model", model).get_maps(canonical_256(rgb, torch, device=device), normalize=False)
    return raw_map(st, stae), {"coverage": "not_applicable"}


def _decision_id(mode: str) -> str:
    return SPLIT_DECISION_ID if mode == "e2-split" else "DEC-GEO-002"


def _write_or_reuse(root: Path, run_id: str, name: str, data: bytes) -> tuple[Path, bool]:
    path = root / name
    if path.exists():
        if path.read_bytes() != data: raise ValueError("content-addressed artifact collision")
        return path, False
    return _runner._write(root, run_id, name, data), True


def run_inference(args: Any) -> dict[str, Any]:
    args._stage = "admission"
    if args.mode not in {"e2-split", "e1"} or args.repeat < 2: raise ValueError("infer mode must be e2-split/e1 and repeat >= 2")
    import torch
    root, image = Path(args.artifact_root).resolve(), _reject_dataset_input(args)
    freeze, freeze_path = None, None
    if args.mode == "e2-split":
        if not args.split_freeze: raise ValueError("--split-freeze is required for e2-split")
        args._stage = "freeze"; freeze_path = Path(args.split_freeze).resolve(); freeze = json.loads(freeze_path.read_text()); verify_split_freeze(freeze)
    elif args.split_freeze: raise ValueError("--split-freeze is only accepted for e2-split")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
    with GpuLease(args.lease_directory, args.run_id, INFER_COMMAND):
        admitted, model = _runner._model(args, torch)
        if freeze is not None: _runner.verify_split_lineage(freeze, admitted)
        reset = getattr(torch.cuda, "reset_peak_memory_stats", None)
        if reset: reset()
        args._stage = "decode"; rgb = decode_rgb01(image); device = torch.device("cuda:0")
        outputs, timings, geometry = [], [], None
        for _ in range(args.repeat):
            args._stage = "inference"; _sync(torch); began = time.perf_counter()
            if args.mode == "e2-split":
                local, global_map, geometry = split_branch_raw_maps(rgb, model, torch, device=device); output = combine_split_maps(local, global_map, freeze["quantiles"], torch)
            else: output, geometry = _e1_map(rgb, model, torch, device)
            _sync(torch); timings.append((time.perf_counter() - began) * 1000); outputs.append(output)
        raws = [bytes(value.tobytes()) for value in outputs]; hashes = [sha256(raw).hexdigest() for raw in raws]
        if len(set(hashes)) != 1: raise ValueError("repeat output hashes differ")
        if args.mode == "e2-split" and float(geometry.get("weight_min", 0)) <= 0: raise ValueError("split coverage is non-positive")
        args._stage = "render"; png, bounds = _heatmap_bytes(outputs[0]); raw, output_sha256 = raws[0], hashes[0]; png_sha256 = sha256(png).hexdigest()
        args._stage = "persist"; created: list[Path] = []
        try:
            raw_path, made = _write_or_reuse(root, args.run_id, f"highres-split-raw-{args.mode}-{output_sha256}.bin", raw)
            if made: created.append(raw_path)
            png_path, made = _write_or_reuse(root, args.run_id, f"highres-split-heatmap-{args.mode}-{png_sha256}.png", png)
            if made: created.append(png_path)
        except Exception:
            for path in created: path.unlink(missing_ok=True)
            raise
        peak = getattr(torch.cuda, "max_memory_allocated", lambda: 0)(), getattr(torch.cuda, "max_memory_reserved", lambda: 0)()
        smoke = {"repeat": args.repeat, "map_latency_ms": timings, "median_map_latency_ms": median(timings), "output_hashes": hashes, "all_equal": True,
                 "cuda_peak_allocated_bytes": peak[0], "cuda_peak_reserved_bytes": peak[1], "scope": "decode/model-load excluded; map computation only"}
        if args.mode == "e2-split": smoke["coverage"] = {"weight_min": geometry["weight_min"], "weight_max": geometry["weight_max"], "tile": geometry["tile"], "stride": geometry["stride"], "box_count": len(geometry["boxes"])}
        else: smoke["coverage"] = "not_applicable"
        manifest = {"schema_version": "1.0", "public_pipeline": PUBLIC_PIPELINE, "operation": "infer", "mode": args.mode, "run_id": args.run_id, "decision_id": _decision_id(args.mode),
                    "input_sha256": _hash(image), "checkpoint_sha256": admitted.checkpoint_sha256, "split_freeze_sha256": None if freeze_path is None else _hash(freeze_path),
                    "raw": {"basename": raw_path.name, "sha256": output_sha256, "dtype": "<f4", "shape": list(outputs[0].shape)}, "heatmap": {"basename": png_path.name, "sha256": png_sha256, "method": "minmax_grayscale", "bounds": bounds}, "smoke": smoke}
        payload = _canon(manifest)
        try:
            manifest_path, _ = _write_or_reuse(root, args.run_id, f"highres-split-manifest-{args.mode}-{sha256(payload).hexdigest()}.json", payload)
        except Exception:
            for path in created: path.unlink(missing_ok=True)
            raise
    return {"status": "READY", "raw_map": str(raw_path), "heatmap": str(png_path), "manifest": str(manifest_path), "output_sha256": output_sha256}


def run_validation(args: Any) -> dict[str, Any]:
    args._stage = "validation"
    return _runner.run_validation(args)


def run_test_public(args: Any) -> dict[str, Any]:
    args._stage = "testpub"
    return _runner.run_test_public_once(args)


def run(args: Any) -> dict[str, Any]:
    runners = {"infer": run_inference, "validation": run_validation, "testpub": run_test_public}
    if args.operation not in runners: raise ValueError(f"unsupported public operation: {args.operation}")
    try:
        return runners[args.operation](args)
    except Exception as exc:
        try: _write_failure_record(args, exc)
        except Exception: pass  # Best-effort provenance must not mask the evaluation error.
        raise


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("infer", "validation", "testpub"):
        command = sub.add_parser(name, help=("Infer one raw E2-Split anomaly map" if name == "infer" else f"Run E2-Split {name} evaluation"))
        _runner.add_common_arguments(command)
        if name == "infer": command.add_argument("--input-image", type=Path, required=True); command.add_argument("--mode", choices=("e2-split", "e1"), default="e2-split"); command.add_argument("--repeat", type=int, default=2); command.add_argument("--split-freeze", type=Path)
        if name == "testpub": command.add_argument("--split-freeze", type=Path, required=True); command.add_argument("--evaluator", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
