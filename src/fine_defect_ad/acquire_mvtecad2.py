"""Fail-closed, resumable MVTec AD 2 archive acquisition (no implicit download)."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

from .storage import Allocation, READY, STOPPED_INCOMPLETE, StorageBlocked, invalidate_run, preflight, require_proof, roots_from_env

ARCHIVE_NAME = "mvtec_ad_2.tar.gz"
ARCHIVE_BYTES = 32_739_596_982
ARCHIVE_SHA256 = "c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4"
ARCHIVE_URL = "https://www.mydrive.ch/shares/150997/701c90d3aea6588f404936e32a674602/download/466712769-1743429042/mvtec_ad_2.tar.gz"
ANOMALIB_PROVENANCE = "https://github.com/open-edge-platform/anomalib/blob/3759687e76395c4d6d239552d3bf6d72e003da78/src/anomalib/data/datamodules/image/mvtecad2.py"
OFFICIAL_FORM_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2"
ACK_FIELDS = {"status", "official_form_url", "license", "noncommercial", "accepted_at"}
AUDIT_METHOD = "tarfile-r|gz-member-size-count-sha256-compressed-stream"


def _under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _checkout_root() -> Path:
    """Resolve the checkout from this module, never the caller's directory."""
    source_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=source_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode:
        raise StorageBlocked("unable to resolve source checkout for terms acknowledgement")
    return Path(result.stdout.strip()).resolve()


def _ack_is_private_or_ignored(path: Path) -> bool:
    """Outside this checkout is private; inside it must be Git-ignored."""
    root = _checkout_root()
    if not _under(path, root):
        return True
    return subprocess.run(["git", "check-ignore", "-q", "--", str(path)], cwd=root, check=False).returncode == 0


def load_terms_ack(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or not _ack_is_private_or_ignored(path):
        raise StorageBlocked("terms acknowledgement must be private or git-ignored")
    try:
        ack = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageBlocked("invalid terms acknowledgement") from exc
    if not isinstance(ack, dict) or set(ack) != ACK_FIELDS:
        raise StorageBlocked("terms acknowledgement has missing, extra, or PII fields")
    if (ack.get("status"), ack.get("official_form_url"), ack.get("license"), ack.get("noncommercial")) != (
        "ACCEPTED", OFFICIAL_FORM_URL, "CC-BY-NC-SA-4.0", True
    ):
        raise StorageBlocked("terms acknowledgement does not match MVTec AD 2 conditions")
    try:
        accepted_at = datetime.fromisoformat(ack["accepted_at"].replace("Z", "+00:00"))
        if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
            raise ValueError("timezone required")
    except (AttributeError, ValueError) as exc:
        raise StorageBlocked("terms acknowledgement accepted_at must be timezone-aware") from exc
    return ack


def archive_plan(run_id: str) -> dict:
    """Archive metadata is sizing only; extraction evidence authorizes download."""
    return {
        "status": "SIZING_ONLY_NOT_DOWNLOAD_AUTHORIZED", "workflow_status": STOPPED_INCOMPLETE,
        "run_id": run_id,
        "allocations": [{"root": "data", "bytes": ARCHIVE_BYTES, "kind": "persistent", "component_id": "mvtecad2-archive", "source": ANOMALIB_PROVENANCE}],
        "reserve_bytes": 0,
        "reserve_evidence": {"max_pending_atomic_write_bytes": 0, "measured_high_water_bytes": 0, "runtime_or_source_citation": "same-inode .partial to final atomic rename; archive only, extraction peak unknown"},
        "archive": {"url": ARCHIVE_URL, "content_length": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256, "provenance": ANOMALIB_PROVENANCE},
    }


class _CountingReader:
    def __init__(self, stream):
        self.stream = stream; self.bytes = 0; self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self.stream.read(size)
        self.bytes += len(block); self.digest.update(block)
        return block


def load_extraction_evidence(path: Path) -> dict:
    """Accept only metadata audit evidence bound to the pinned archive identity."""
    try:
        evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageBlocked("exact extraction sizing evidence is required") from exc
    required = {"archive_url", "archive_bytes", "archive_sha256", "audit_method", "exact_uncompressed_bytes", "max_member_bytes", "member_count"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise StorageBlocked("extraction evidence must be a complete pinned metadata audit")
    if (evidence["archive_url"], evidence["archive_bytes"], evidence["archive_sha256"], evidence["audit_method"]) != (ARCHIVE_URL, ARCHIVE_BYTES, ARCHIVE_SHA256, AUDIT_METHOD):
        raise StorageBlocked("extraction evidence is not bound to the pinned archive")
    total, member, count = evidence["exact_uncompressed_bytes"], evidence["max_member_bytes"], evidence["member_count"]
    if not all(isinstance(value, int) and value >= 0 for value in (total, member, count)) or member > total or not count:
        raise StorageBlocked("invalid exact extraction sizing evidence")
    return evidence

def acquisition_plan(run_id: str, extraction_evidence: dict) -> dict:
    archive = archive_plan(run_id)
    total, member = extraction_evidence["exact_uncompressed_bytes"], extraction_evidence["max_member_bytes"]
    provenance = json.dumps({key: extraction_evidence[key] for key in ("archive_url", "archive_bytes", "archive_sha256", "audit_method")}, sort_keys=True)
    archive.update({
        "status": "DOWNLOAD_AUTHORIZED_PLAN",
        "allocations": [
            *archive["allocations"],
            {"root": "data", "bytes": total, "kind": "persistent", "component_id": "mvtecad2-extraction", "source": provenance, "exact_uncompressed_bytes_source": provenance},
            {"root": "data", "bytes": member, "kind": "transient", "component_id": "mvtecad2-extraction-member", "source": provenance},
        ],
    })
    archive["reserve_evidence"]["runtime_or_source_citation"] = "archive plus exact extraction persistent bytes and maximum member transient bytes; " + provenance
    return archive


def audit_metadata(*, terms_ack: Path, url: str = ARCHIVE_URL, opener: Callable = urllib.request.urlopen) -> dict:
    """Stream tar headers only after terms acceptance; never writes or proves readiness."""
    load_terms_ack(terms_ack)
    if url != ARCHIVE_URL:
        raise StorageBlocked("archive URL must match pinned anomalib provenance")
    with opener(urllib.request.Request(url), timeout=60) as response:
        if response.status != 200 or response.headers.get("Content-Length") != str(ARCHIVE_BYTES):
            raise StorageBlocked("Content-Length does not match pinned archive size")
        counted = _CountingReader(response)
        total = maximum = count = 0
        with tarfile.open(fileobj=counted, mode="r|gz") as archive:
            for member in archive:
                total += member.size; maximum = max(maximum, member.size); count += 1
        if counted.bytes != ARCHIVE_BYTES or counted.digest.hexdigest() != ARCHIVE_SHA256:
            raise StorageBlocked("metadata audit compressed archive identity mismatch")
    return {"status": "SIZING_AUDIT_ONLY_NOT_STORAGE_READY", "archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256, "audit_method": AUDIT_METHOD, "exact_uncompressed_bytes": total,
            "max_member_bytes": maximum, "member_count": count}

def _hash_prefix(path: Path) -> tuple[hashlib._Hash, int]:
    digest = hashlib.sha256()
    size = 0
    if path.exists():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block); size += len(block)
    return digest, size


def _validate_existing(final: Path) -> bool:
    if not final.exists():
        return False
    if not final.is_file() or final.stat().st_size != ARCHIVE_BYTES:
        raise StorageBlocked("existing final archive does not match expected size")
    digest, _ = _hash_prefix(final)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise StorageBlocked("existing final archive hash mismatch; refusing to replace it")
    return True


def _content_range(response, start: int) -> None:
    value = response.headers.get("Content-Range", "")
    expected = f"bytes {start}-{ARCHIVE_BYTES - 1}/{ARCHIVE_BYTES}"
    if response.status != 206 or value != expected:
        raise StorageBlocked("server did not honor validated HTTP Range resume")


def _stop_enospc(proof, run_id: str, partial: Path) -> dict:
    return invalidate_run(proof, run_id=run_id, cause="ENOSPC", partial_path=partial)

def download(*, run_id: str, terms_ack: Path, destination: Path | None = None, url: str = ARCHIVE_URL, opener: Callable = urllib.request.urlopen) -> dict:
    """Download only after terms, exact extraction sizing, and a fresh storage proof."""
    load_terms_ack(terms_ack)
    if url != ARCHIVE_URL:
        raise StorageBlocked("archive URL must match pinned anomalib provenance")
    sizing = audit_metadata(terms_ack=terms_ack, url=url, opener=opener)
    roots = roots_from_env()
    final = (Path(destination) if destination else roots["data"] / ARCHIVE_NAME).expanduser().resolve()
    if not _under(final, roots["data"]):
        raise StorageBlocked("archive destination must remain under data root")
    if _validate_existing(final):
        return {"status": READY, "run_id": run_id, "path": str(final), "existing": True}
    plan = acquisition_plan(run_id, sizing)
    proof = preflight(run_id=run_id, allocations=[Allocation(**item) for item in plan["allocations"]], reserve_bytes=plan["reserve_bytes"], reserve_evidence=plan["reserve_evidence"])
    partial = final.with_name("." + final.name + ".partial")
    digest, offset = _hash_prefix(partial)
    if offset > ARCHIVE_BYTES:
        raise StorageBlocked("partial archive exceeds expected size")
    if offset == ARCHIVE_BYTES:
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise StorageBlocked("complete partial archive hash mismatch; partial preserved")
        require_proof(proof, run_id=run_id)
        try: os.replace(partial, final)
        except OSError as exc:
            if exc.errno == errno.ENOSPC: return _stop_enospc(proof, run_id, partial)
            raise
        return {"status": READY, "run_id": run_id, "path": str(final), "sha256": ARCHIVE_SHA256, "resumed": True}
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout=60) as response:
            length = response.headers.get("Content-Length")
            if offset:
                _content_range(response, offset)
                if length != str(ARCHIVE_BYTES - offset): raise StorageBlocked("resumed Content-Length mismatch")
            elif response.status != 200 or length != str(ARCHIVE_BYTES):
                raise StorageBlocked("Content-Length does not match pinned archive size")
            require_proof(proof, run_id=run_id)
            final.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("ab" if offset else "wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block); digest.update(block)
                output.flush(); os.fsync(output.fileno())
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            return _stop_enospc(proof, run_id, partial)
        raise
    if partial.stat().st_size != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
        raise StorageBlocked("downloaded archive hash or size mismatch; partial preserved")
    try: os.replace(partial, final)
    except OSError as exc:
        if exc.errno == errno.ENOSPC: return _stop_enospc(proof, run_id, partial)
        raise
    return {"status": READY, "run_id": run_id, "path": str(final), "sha256": ARCHIVE_SHA256}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True); parser.add_argument("--terms-ack", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--plan", action="store_true"); parser.add_argument("--metadata-audit", action="store_true"); parser.add_argument("--extract", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.extract:
            print(json.dumps(extract(run_id=args.run_id, terms_ack=args.terms_ack, destination=args.destination), sort_keys=True)); return 0
        if args.plan:
            plan = archive_plan(args.run_id)
            print(json.dumps(plan, sort_keys=True)); return 0
        if not args.terms_ack: raise StorageBlocked("--terms-ack is required")
        load_terms_ack(args.terms_ack)
        if args.metadata_audit:
            print(json.dumps(audit_metadata(terms_ack=args.terms_ack), sort_keys=True)); return 0
        print(json.dumps(download(run_id=args.run_id, terms_ack=args.terms_ack, destination=args.destination), sort_keys=True)); return 0
    except (OSError, StorageBlocked, urllib.error.URLError) as exc:
        print(json.dumps({"status": "STORAGE_BLOCKED", "workflow_status": STOPPED_INCOMPLETE, "reason": str(exc)})); return 2


# Extraction is deliberately separate from acquisition: it never contacts the source URL.
def _inspect_local_archive(archive_path: Path) -> tuple[list[tarfile.TarInfo], dict]:
    if not _validate_existing(archive_path):
        raise StorageBlocked("verified final archive is required")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
    total = maximum = 0; manifest = []
    for member in members:
        name = Path(member.name)
        if member.name.startswith("/") or ".." in name.parts or member.issym() or member.islnk() or member.isdev() or not (member.isfile() or member.isdir()):
            raise StorageBlocked("archive contains unsafe member")
        if member.isfile():
            total += member.size; maximum = max(maximum, member.size); manifest.append({"name": member.name, "size": member.size})
    if not manifest:
        raise StorageBlocked("archive has no regular members")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return members, {"exact_uncompressed_bytes": total, "max_member_bytes": maximum, "member_count": len(manifest), "member_manifest_sha256": digest}


def extract(*, run_id: str, terms_ack: Path | None = None, destination: Path | None = None) -> dict:
    """Safely extract the verified archive under data root; never overwrites or cleans."""
    if terms_ack is None:
        raise StorageBlocked("terms acknowledgement is required before extraction")
    load_terms_ack(terms_ack)
    roots = roots_from_env(); archive_path = (roots["data"] / ARCHIVE_NAME).resolve()
    target = (Path(destination) if destination else roots["data"] / "mvtec_ad_2").expanduser().resolve()
    if not _under(target, roots["data"]) or target == roots["data"].resolve():
        raise StorageBlocked("extraction destination must be a dedicated data-root descendant")
    members, sizing = _inspect_local_archive(archive_path)
    provenance = json.dumps({"archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES, "archive_sha256": ARCHIVE_SHA256, "audit_method": "local-verified-tar-inspection"}, sort_keys=True)
    plan = {"run_id": run_id, "allocations": [
        {"root":"data","bytes":ARCHIVE_BYTES,"kind":"persistent","component_id":"mvtecad2-archive","source":ANOMALIB_PROVENANCE},
        {"root":"data","bytes":sizing["exact_uncompressed_bytes"],"kind":"persistent","component_id":"mvtecad2-extraction","source":provenance,"exact_uncompressed_bytes_source":provenance},
        {"root":"data","bytes":sizing["max_member_bytes"],"kind":"transient","component_id":"mvtecad2-extraction-member","source":provenance},
    ], "reserve_bytes":0, "reserve_evidence":{"max_pending_atomic_write_bytes":0,"measured_high_water_bytes":0,"runtime_or_source_citation":provenance}}
    proof = preflight(run_id=run_id, allocations=[Allocation(**item) for item in plan["allocations"]], reserve_bytes=0, reserve_evidence=plan["reserve_evidence"])
    try:
        require_proof(proof, run_id=run_id)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in members:
                final = (target / member.name).resolve()
                if not _under(final, target):
                    raise StorageBlocked("extraction destination escaped")
                require_proof(proof, run_id=run_id)
                if member.isdir():
                    if final.exists() and (final.is_symlink() or not final.is_dir()):
                        raise StorageBlocked("extraction refuses unsafe existing directory")
                    final.mkdir(parents=True, exist_ok=True)
                    continue
                if final.exists():
                    raise StorageBlocked("extraction refuses existing destination")
                final.parent.mkdir(parents=True, exist_ok=True)
                partial = final.with_name("." + final.name + ".partial")
                with archive.extractfile(member) as source, partial.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                    output.flush(); os.fsync(output.fileno())
                os.replace(partial, final)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            return invalidate_run(proof, run_id=run_id, cause="ENOSPC", partial_path=target)
        raise
    return {"status": READY, "run_id": run_id, "archive_sha256": ARCHIVE_SHA256, "exact_uncompressed_bytes": sizing["exact_uncompressed_bytes"], "max_member_bytes": sizing["max_member_bytes"], "member_count": sizing["member_count"], "extracted_file_count": sizing["member_count"], "extracted_bytes": sizing["exact_uncompressed_bytes"], "member_manifest_sha256": sizing["member_manifest_sha256"]}


if __name__ == "__main__":
    raise SystemExit(main())
