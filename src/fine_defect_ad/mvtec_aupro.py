"""Hash-verified local wrapper around the official MVTec AD v1 evaluator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


EVALUATOR_VERSION = "1.0"
INTEGRATION_LIMIT = 0.05
RESULT_LABEL = "LOCAL_MVTEC_AD_V1_SOURCE_NOT_AD2_SERVER_EQUIVALENT"
EMPTY_MASK_STATUS = "LOCAL_MVTEC_AD_V1_EMPTY_MASK_UNDEFINED"
_ROOT = "mvtec_ad_evaluation"
_REQUIRED_SOURCES = ("README.md", "evaluate_experiment.py", "LICENSE.txt", "generic_util.py", "pro_curve_util.py")


class EvaluatorVerificationError(ValueError):
    """The supplied local evaluator is not the exact inspected MVTec source."""


@dataclass(frozen=True)
class EvaluatorSource:
    path: Path
    kind: str
    archive_sha256: str | None
    source_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_identity() -> tuple[str, dict[str, str]]:
    """Read only the evaluator identity recorded in repository evidence."""
    try:
        evidence = json.loads((Path(__file__).resolve().parents[2] / "evidence" / "mvtec-metric-provenance.json").read_text())
        source = evidence["sources"]["mvtec_ad_evaluator_v1"]
        hashes = {Path(item["path"]).name: item["sha256"] for item in source["inspected_members"]}
        hashes = {name: hashes[name] for name in _REQUIRED_SOURCES}
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise EvaluatorVerificationError("complete evaluator identity evidence is required") from exc
    return source["archive_sha256"], hashes


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = Path(member.name)
    return not member.name.startswith("/") and ".." not in path.parts and not (member.issym() or member.islnk() or member.isdev())


def _check_hashes(read: Any, sources: dict[str, str]) -> dict[str, str]:
    hashes = {}
    for name, expected in sources.items():
        body = read(name)
        actual = hashlib.sha256(body).hexdigest()
        if actual != expected:
            raise EvaluatorVerificationError(f"official evaluator source hash mismatch: {name}")
        hashes[name] = actual
    return hashes


def verify_evaluator(source: Path) -> EvaluatorSource:
    """Verify a local v1 archive or extracted root without downloading anything."""
    source = Path(source).expanduser().resolve()
    archive_sha256, sources = _pinned_identity()
    if source.is_file():
        if _sha256(source) != archive_sha256:
            raise EvaluatorVerificationError("official evaluator archive hash mismatch")
        try:
            with tarfile.open(source, "r:xz") as archive:
                members = archive.getmembers()
                if any(not _safe_member(member) for member in members):
                    raise EvaluatorVerificationError("official evaluator archive has unsafe member")
                entries = {member.name: member for member in members if member.isfile()}
                hashes = _check_hashes(lambda name: _archive_bytes(archive, entries, name), sources)
        except tarfile.TarError as exc:
            raise EvaluatorVerificationError("invalid official evaluator archive") from exc
        return EvaluatorSource(source, "archive", archive_sha256, hashes)
    root = source / _ROOT if (source / _ROOT).is_dir() else source
    if not root.is_dir() or root.is_symlink():
        raise EvaluatorVerificationError("evaluator source must be an extracted root or archive")
    hashes = _check_hashes(lambda name: _regular_file_bytes(root / name), sources)
    return EvaluatorSource(root, "root", archive_sha256, hashes)


def _archive_bytes(archive: tarfile.TarFile, entries: dict[str, tarfile.TarInfo], name: str) -> bytes:
    member = entries.get(f"{_ROOT}/{name}")
    if member is None:
        raise EvaluatorVerificationError(f"official evaluator source missing: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise EvaluatorVerificationError(f"official evaluator source unreadable: {name}")
    return stream.read()


def _regular_file_bytes(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise EvaluatorVerificationError(f"official evaluator source missing or unsafe: {path.name}")
    return path.read_bytes()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluatorVerificationError(f"cannot import official evaluator source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    # generic_util imports tifffile for image I/O although its official
    # trapezoid helper does not use it.  Do not make that unused optional I/O
    # dependency a requirement for array-only local PRO evaluation.
    placeholder = None
    if path.name == "generic_util.py":
        try:
            import tifffile  # noqa: F401
        except ModuleNotFoundError:
            placeholder = types.ModuleType("tifffile")
            placeholder.imread = None
            sys.modules["tifffile"] = placeholder
    try:
        spec.loader.exec_module(module)
    finally:
        if placeholder is not None:
            sys.modules.pop("tifffile", None)
    return module


def _official_functions(verified: EvaluatorSource):
    if verified.kind == "root":
        root = verified.path
        return _load_module("mvtec_ad_v1_generic", root / "generic_util.py").trapezoid, _load_module("mvtec_ad_v1_pro", root / "pro_curve_util.py").compute_pro
    with tarfile.open(verified.path, "r:xz") as archive, tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name in ("generic_util.py", "pro_curve_util.py"):
            (root / name).write_bytes(_archive_bytes(archive, {member.name: member for member in archive.getmembers() if member.isfile()}, name))
        return _load_module("mvtec_ad_v1_generic", root / "generic_util.py").trapezoid, _load_module("mvtec_ad_v1_pro", root / "pro_curve_util.py").compute_pro


def local_au_pro_0_05(anomaly_maps: Sequence[Any], ground_truth_maps: Sequence[Any | None], evaluator: Path) -> dict[str, Any]:
    """Run only the official local PRO path; no comparator or threshold metrics."""
    verified = verify_evaluator(evaluator)
    if len(anomaly_maps) != len(ground_truth_maps) or not anomaly_maps:
        raise ValueError("anomaly maps and ground truth maps must be non-empty and aligned")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required by the official evaluator") from exc
    predictions = [np.asarray(item) for item in anomaly_maps]
    if any(item.ndim != 2 for item in predictions):
        raise ValueError("official evaluator requires 2D anomaly maps")
    if any(item.shape != predictions[0].shape for item in predictions):
        raise ValueError("official evaluator requires identically shaped anomaly maps")
    restored = []
    ground_truth = []
    for index, (prediction, mask) in enumerate(zip(predictions, ground_truth_maps)):
        if mask is None:
            ground_truth.append(np.zeros(prediction.shape)); restored.append(index)
        else:
            array = np.asarray(mask)
            if array.shape != prediction.shape:
                raise ValueError("ground truth shape must match anomaly map")
            ground_truth.append(array)
    record = {
        "status": RESULT_LABEL, "evaluator_version": EVALUATOR_VERSION,
        "integration_limit": INTEGRATION_LIMIT, "comparator": None,
        "config": {
            "pro_integration_limit": INTEGRATION_LIMIT,
            "spatial_alignment": "CALLER_PREALIGNED_REQUIRED",
            "none_masks_restored_to_zeros_only": True,
        },
        "threshold_metrics": "BLOCKED_NO_VERIFIED_COMPARATOR",
        "source": {"kind": verified.kind, "archive_sha256": verified.archive_sha256, "source_hashes": verified.source_hashes},
        "ground_truth_restoration": {"none_masks_restored_to_zeros": restored},
    }
    if not any(np.any(mask > 0) for mask in ground_truth):
        return {**record, "empty_mask_behavior": EMPTY_MASK_STATUS, "output": None}
    trapezoid, compute_pro = _official_functions(verified)
    fprs, pros = compute_pro(predictions, ground_truth)
    au_pro = float(trapezoid(fprs, pros, x_max=INTEGRATION_LIMIT) / INTEGRATION_LIMIT)
    return {**record, "empty_mask_behavior": "OFFICIAL_UTILITY_EXECUTED", "output": {"au_pro_0_05": au_pro, "fpr": fprs.tolist(), "pro": pros.tolist()}}


evaluate_local_au_pro_0_05 = local_au_pro_0_05
