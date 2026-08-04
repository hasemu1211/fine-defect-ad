"""Pinned, resumable EfficientAD upstream asset acquisition; never acquires MVTec AD 2."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from .storage import Allocation, READY, STOPPED_INCOMPLETE, StorageBlocked, preflight, require_proof, roots_from_env

ASSETS = {
    "teacher": {
        "archive": "efficientad_pretrained_weights.zip",
        "url": "https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
        "bytes": 39_960_466,
        "sha256": "c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
        "format": "zip",
    },
    "imagenette": {
        "archive": "imagenette2.tgz",
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        "bytes": 1_557_161_267,
        "sha256": "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
        "format": "tar.gz",
    },
}


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256(); size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block); size += len(block)
    return digest.hexdigest(), size


def download_plan(run_id: str) -> dict:
    """Budget exactly one persistent final archive per pinned upstream asset."""
    return {
        "run_id": run_id,
        "allocations": [
            {"root": "data", "bytes": item["bytes"], "kind": "persistent", "component_id": f"r1-{name}-raw", "source": item["url"]}
            for name, item in ASSETS.items()
        ],
        "reserve_bytes": 0,
        "reserve_evidence": {
            "max_pending_atomic_write_bytes": 0,
            "measured_high_water_bytes": 0,
            "runtime_or_source_citation": "same-directory .partial to final atomic rename; each raw archive is budgeted once",
        },
        "assets": ASSETS,
    }


def _asset(name: str) -> dict:
    try:
        return ASSETS[name]
    except KeyError as exc:
        raise StorageBlocked(f"unknown R1 upstream asset {name}") from exc


def _raw_path(root: Path, asset: dict) -> Path:
    return root / "efficientad-upstream" / "raw" / asset["archive"]


def _validate_final(path: Path, asset: dict) -> bool:
    if not path.exists():
        return False
    digest, size = _hash(path)
    if size != asset["bytes"] or digest != asset["sha256"]:
        raise StorageBlocked("existing archive does not match pinned size and SHA256; refusing to replace it")
    return True


def _content_range(response, start: int, total: int) -> None:
    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{total - 1}/{total}":
        raise StorageBlocked("server did not honor validated HTTP Range resume")


def download(*, name: str, run_id: str, roots: dict[str, Path] | None = None) -> dict:
    """Download one pinned archive after a fresh full-capacity proof; failures persist."""
    asset = _asset(name); active = roots or roots_from_env()
    if "data" not in active or "artifact" not in active:
        raise StorageBlocked("data and artifact roots are required for acquisition")
    raw = _raw_path(active["data"], asset)
    if not _under(raw, active["data"]):
        raise StorageBlocked("raw archive must remain under data root")
    if _validate_final(raw, asset):
        return {"status": READY, "asset": name, "path": str(raw), "existing": True}
    plan = download_plan(run_id)
    proof = preflight(run_id=run_id, allocations=[Allocation(**item) for item in plan["allocations"]], reserve_bytes=0, reserve_evidence=plan["reserve_evidence"], roots=active)
    partial = raw.with_name("." + raw.name + ".partial")
    if partial.exists():
        digest, offset = _hash(partial)
    else:
        digest, offset = hashlib.sha256(), 0
    if offset > asset["bytes"]:
        raise StorageBlocked("partial archive exceeds pinned length")
    if offset == asset["bytes"]:
        if digest != asset["sha256"]:
            raise StorageBlocked("complete partial archive hash mismatch; partial preserved")
        require_proof(proof, run_id=run_id, roots=active)
        os.replace(partial, raw); _fsync_directory(raw.parent)
        return {"status": READY, "asset": name, "path": str(raw), "sha256": asset["sha256"], "resumed": True}
    request = urllib.request.Request(asset["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if offset:
                _content_range(response, offset, asset["bytes"])
                if length != str(asset["bytes"] - offset): raise StorageBlocked("resumed Content-Length mismatch")
            elif response.status != 200 or length != str(asset["bytes"]):
                raise StorageBlocked("Content-Length does not match pinned archive size")
            require_proof(proof, run_id=run_id, roots=active)
            raw.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("ab" if offset else "wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block); digest.update(block)
                output.flush(); os.fsync(output.fileno())
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            marker = active["artifact"] / f".invalidated-{run_id}.json"
            with marker.open("w", encoding="utf-8") as stream:
                json.dump({"run_id": run_id, "status": "INVALIDATED", "workflow_status": STOPPED_INCOMPLETE, "cause": "ENOSPC", "partial_path": str(partial)}, stream, sort_keys=True)
                stream.flush(); os.fsync(stream.fileno())
            _fsync_directory(marker.parent)
            return {"status": "INVALIDATED", "workflow_status": STOPPED_INCOMPLETE, "asset": name, "partial_path": str(partial)}
        raise
    if partial.stat().st_size != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
        raise StorageBlocked("downloaded archive hash or size mismatch; partial preserved")
    os.replace(partial, raw); _fsync_directory(raw.parent)
    return {"status": READY, "asset": name, "path": str(raw), "sha256": asset["sha256"]}


def _members(asset: dict, archive: Path) -> Iterable[tuple[str, int, object]]:
    if asset["format"] == "zip":
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist(): yield member.filename, member.file_size, member
    else:
        with tarfile.open(archive, "r:gz") as tarred:
            for member in tarred:
                if member.isfile(): yield member.name, member.size, member
                elif member.isdir(): yield member.name, 0, member
                else: raise StorageBlocked(f"unsafe non-file tar member {member.name}")


def inspect(name: str, root: Path) -> dict:
    asset = _asset(name); archive = _raw_path(root, asset)
    if not _validate_final(archive, asset): raise StorageBlocked("archive is not READY")
    members = []
    for member_name, size, _ in _members(asset, archive):
        relative = Path(member_name)
        if relative.is_absolute() or ".." in relative.parts or not member_name:
            raise StorageBlocked(f"unsafe archive member {member_name!r}")
        members.append({"path": member_name, "bytes": size})
    return {"asset": name, "archive": str(archive), "archive_sha256": asset["sha256"], "members": members,
            "uncompressed_bytes": sum(member["bytes"] for member in members), "maximum_file_bytes": max((member["bytes"] for member in members), default=0)}


def extraction_plan(run_id: str, root: Path) -> dict:
    inspected = [inspect(name, root) for name in ASSETS]
    allocations = [
        {"root": "data", "bytes": ASSETS[item["asset"]]["bytes"], "kind": "persistent", "component_id": f"r1-{item['asset']}-raw", "source": ASSETS[item["asset"]]["url"]}
        for item in inspected
    ] + [
        {"root": "data", "bytes": item["uncompressed_bytes"], "kind": "persistent", "component_id": f"r1-{item['asset']}-extracted", "source": f"local streamed inspection of {item['archive']}", "exact_uncompressed_bytes_source": f"local streamed archive member sizes; maximum file {item['maximum_file_bytes']}"}
        for item in inspected
    ] + [
        {"root": "data", "bytes": item["maximum_file_bytes"], "kind": "transient", "component_id": f"r1-{item['asset']}-extract-temp", "source": f"local streamed inspection of {item['archive']}"}
        for item in inspected
    ]
    return {"run_id": run_id, "allocations": allocations, "reserve_bytes": 0,
            "reserve_evidence": {"max_pending_atomic_write_bytes": 0, "measured_high_water_bytes": 0, "runtime_or_source_citation": "same-directory per-file .partial to final atomic renames; transient allocation is inspected maximum file"},
            "inspections": inspected}


def _copy_stream(source, destination: Path) -> str:
    temporary = destination.with_name("." + destination.name + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block); digest.update(block)
        output.flush(); os.fsync(output.fileno())
    os.replace(temporary, destination); _fsync_directory(destination.parent)
    return digest.hexdigest()


def extract(*, name: str, run_id: str, roots: dict[str, Path] | None = None) -> dict:
    """Extract one inspected archive to a dedicated data subdirectory after fresh proof."""
    active = roots or roots_from_env(); asset = _asset(name); root = active["data"]
    plan = extraction_plan(run_id, root)
    proof = preflight(run_id=run_id, allocations=[Allocation(**item) for item in plan["allocations"]], reserve_bytes=0, reserve_evidence=plan["reserve_evidence"], roots=active)
    require_proof(proof, run_id=run_id, roots=active)
    target = root / "efficientad-upstream" / "extracted" / name
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    archive = _raw_path(root, asset)
    if asset["format"] == "zip":
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                relative = Path(info.filename)
                if relative.is_absolute() or ".." in relative.parts or not info.filename: raise StorageBlocked(f"unsafe archive member {info.filename!r}")
                destination = target / relative
                if not _under(destination, target): raise StorageBlocked("archive member escapes extraction root")
                if info.is_dir(): destination.mkdir(parents=True, exist_ok=True); continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source: copied.append({"path": info.filename, "sha256": _copy_stream(source, destination)})
    else:
        with tarfile.open(archive, "r:gz") as tarred:
            for info in tarred:
                relative = Path(info.name)
                if relative.is_absolute() or ".." in relative.parts or not info.name or not (info.isfile() or info.isdir()): raise StorageBlocked(f"unsafe archive member {info.name!r}")
                destination = target / relative
                if not _under(destination, target): raise StorageBlocked("archive member escapes extraction root")
                if info.isdir(): destination.mkdir(parents=True, exist_ok=True); continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tarred.extractfile(info)
                if source is None: raise StorageBlocked(f"cannot read archive member {info.name}")
                with source: copied.append({"path": info.name, "sha256": _copy_stream(source, destination)})
    _fsync_directory(target)
    return {"status": READY, "asset": name, "target": str(target), "members": copied}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("plan-download", "download", "inspect", "plan-extract", "extract")); parser.add_argument("--asset", choices=tuple(ASSETS)); parser.add_argument("--run-id", required=True); parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        roots = roots_from_env()
        if args.action == "plan-download": result = download_plan(args.run_id)
        elif args.action == "download": result = download(name=args.asset, run_id=args.run_id, roots=roots)
        elif args.action == "inspect": result = inspect(args.asset, roots["data"])
        elif args.action == "plan-extract": result = extraction_plan(args.run_id, roots["data"])
        else: result = extract(name=args.asset, run_id=args.run_id, roots=roots)
        output = json.dumps(result, sort_keys=True)
        if args.output: args.output.write_text(output + "\n", encoding="utf-8")
        print(output); return 0
    except (OSError, StorageBlocked, urllib.error.URLError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "STORAGE_BLOCKED", "workflow_status": STOPPED_INCOMPLETE, "reason": str(exc)})); return 2

if __name__ == "__main__": raise SystemExit(main())
