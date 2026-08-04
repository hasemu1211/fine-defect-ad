"""Fail-closed, resumable MVTec AD 2 archive acquisition (no implicit download)."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

from .storage import Allocation, INVALIDATED, READY, STOPPED_INCOMPLETE, StorageBlocked, preflight, require_proof, roots_from_env

ARCHIVE_NAME = "mvtec_ad_2.tar.gz"
ARCHIVE_BYTES = 32_739_596_982
ARCHIVE_SHA256 = "c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4"
ARCHIVE_URL = "https://www.mydrive.ch/shares/150997/701c90d3aea6588f404936e32a674602/download/466712769-1743429042/mvtec_ad_2.tar.gz"
ANOMALIB_PROVENANCE = "https://github.com/open-edge-platform/anomalib/blob/3759687e76395c4d6d239552d3bf6d72e003da78/src/anomalib/data/datamodules/image/mvtecad2.py"
OFFICIAL_FORM_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2"
ACK_FIELDS = {"status", "official_form_url", "license", "noncommercial", "accepted_at"}


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
    """The atomic rename has one persistent archive allocation, not two copies."""
    return {
        "run_id": run_id,
        "allocations": [{"root": "data", "bytes": ARCHIVE_BYTES, "kind": "persistent", "component_id": "mvtecad2-archive", "source": ANOMALIB_PROVENANCE}],
        "reserve_bytes": 0,
        "reserve_evidence": {"max_pending_atomic_write_bytes": 0, "measured_high_water_bytes": 0, "runtime_or_source_citation": "same-inode .partial to final atomic rename; one persistent archive allocation"},
        "archive": {"url": ARCHIVE_URL, "content_length": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256, "provenance": ANOMALIB_PROVENANCE},
    }


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
    artifact = Path(proof.roots["artifact"]).resolve()
    marker = artifact / f".invalidated-{run_id}.json"
    marker.write_text(json.dumps({"run_id": run_id, "status": INVALIDATED, "workflow_status": STOPPED_INCOMPLETE, "cause": "ENOSPC", "partial_path": str(partial)}), encoding="utf-8")
    return {"status": INVALIDATED, "workflow_status": STOPPED_INCOMPLETE, "run_id": run_id, "cause": "ENOSPC"}


def download(*, run_id: str, terms_ack: Path, destination: Path | None = None, url: str = ARCHIVE_URL, opener: Callable = urllib.request.urlopen) -> dict:
    """Download only after acknowledgement and a fresh storage proof; preserves failures."""
    load_terms_ack(terms_ack)
    if url != ARCHIVE_URL:
        raise StorageBlocked("archive URL must match pinned anomalib provenance")
    roots = roots_from_env()
    final = (Path(destination) if destination else roots["data"] / ARCHIVE_NAME).expanduser().resolve()
    if not _under(final, roots["data"]):
        raise StorageBlocked("archive destination must remain under data root")
    if _validate_existing(final):
        return {"status": READY, "run_id": run_id, "path": str(final), "existing": True}
    plan = archive_plan(run_id)
    proof = preflight(run_id=run_id, allocations=[Allocation(**item) for item in plan["allocations"]], reserve_bytes=plan["reserve_bytes"], reserve_evidence=plan["reserve_evidence"])
    partial = final.with_name("." + final.name + ".partial")
    digest, offset = _hash_prefix(partial)
    if offset > ARCHIVE_BYTES:
        raise StorageBlocked("partial archive exceeds expected size")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout=60) as response:
            length = response.headers.get("Content-Length")
            if offset:
                _content_range(response, offset)
                if length != str(ARCHIVE_BYTES - offset): raise StorageBlocked("resumed Content-Length mismatch")
            elif response.status != 200 or length != str(ARCHIVE_BYTES):
                raise StorageBlocked("Content-Length does not match pinned archive size")
            require_proof(proof, run_id=run_id)  # immediately before material write
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
    os.replace(partial, final)
    return {"status": READY, "run_id": run_id, "path": str(final), "sha256": ARCHIVE_SHA256}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True); parser.add_argument("--terms-ack", type=Path)
    parser.add_argument("--destination", type=Path); parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.plan:
            print(json.dumps(archive_plan(args.run_id), sort_keys=True)); return 0
        if not args.terms_ack: raise StorageBlocked("--terms-ack is required before download")
        print(json.dumps(download(run_id=args.run_id, terms_ack=args.terms_ack, destination=args.destination), sort_keys=True)); return 0
    except (OSError, StorageBlocked, urllib.error.URLError) as exc:
        print(json.dumps({"status": "STORAGE_BLOCKED", "workflow_status": STOPPED_INCOMPLETE, "reason": str(exc)})); return 2


if __name__ == "__main__":
    raise SystemExit(main())
