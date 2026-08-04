"""Storage admission boundary for the R1 package overlay installer."""
from __future__ import annotations

import argparse
import errno
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

from .storage import (
    Allocation,
    PreflightProof,
    RunInvalidated,
    StorageBlocked,
    _mount,
    invalidate_run,
    preflight,
    require_proof,
    roots_from_env,
)

OVERLAY_LOCK = "requirements/r1-overlay.txt"
OVERLAY_LOCK_SHA256 = "700960caf7f1b5f55cde0f6d4fe53ef0efed1c96047ded9993b57e77d9115ccd"
OVERLAY_VERSION = "anomalib==2.6.0"
DOWNLOAD_BYTES = 136_329_100
OVERLAY_BYTES = 381_660_959
MAX_ARCHIVE_BYTES = 159_488_055
RESERVE_BYTES = DOWNLOAD_BYTES + MAX_ARCHIVE_BYTES
ARCHIVE_METHOD = "remote wheel central-directory; sdist full-stream"
_SOURCE = f"{OVERLAY_LOCK} sha256={OVERLAY_LOCK_SHA256}; {OVERLAY_VERSION}; {ARCHIVE_METHOD}"


def expected_overlay_plan(run_id: str) -> dict[str, Any]:
    """Return the sole source-backed R1 overlay allocation contract."""
    return {
        "run_id": run_id,
        "allocations": [
            {"root": "venv", "bytes": OVERLAY_BYTES, "kind": "persistent", "source": _SOURCE,
             "component_id": "r1-overlay", "exact_uncompressed_bytes_source": ARCHIVE_METHOD},
            {"root": "temp", "bytes": DOWNLOAD_BYTES, "kind": "transient", "source": _SOURCE,
             "component_id": "r1-overlay-download"},
        ],
        "reserve_bytes": RESERVE_BYTES,
        "reserve_evidence": {
            "max_pending_atomic_write_bytes": DOWNLOAD_BYTES,
            "measured_high_water_bytes": MAX_ARCHIVE_BYTES,
            "runtime_or_source_citation": _SOURCE,
            "archive_measurement_method": ARCHIVE_METHOD,
            "lock_sha256": OVERLAY_LOCK_SHA256,
            "lock_version": OVERLAY_VERSION,
        },
    }


def validate_overlay_plan(plan: dict[str, Any]) -> None:
    run_id = plan.get("run_id")
    if not isinstance(run_id, str) or not run_id or plan != expected_overlay_plan(run_id):
        raise StorageBlocked("R1 overlay plan must exactly match the pinned lock and measured D/U/reserve contract")
    lock_path = Path(__file__).parents[2] / OVERLAY_LOCK
    if not lock_path.is_file() or sha256(lock_path.read_bytes()).hexdigest() != OVERLAY_LOCK_SHA256:
        raise StorageBlocked("tracked R1 overlay lock hash does not match its pinned plan")


def admit_overlay_install(plan_path: Path, overlay: Path) -> PreflightProof:
    """Require a fresh exact-plan ext4 proof before any overlay write."""
    plan = json.loads(Path(plan_path).read_text())
    validate_overlay_plan(plan)
    roots = roots_from_env()
    overlay = Path(overlay).resolve()
    if overlay.parent != roots["venv"].resolve() or _mount(overlay.parent)[1] != "ext4":
        raise StorageBlocked("R1 overlay target must be directly under the configured ext4 venv root")
    allocations = [Allocation(**item) for item in plan["allocations"]]
    proof = preflight(run_id=plan["run_id"], allocations=allocations, reserve_bytes=plan["reserve_bytes"],
                      reserve_evidence=plan["reserve_evidence"])
    require_proof(proof, run_id=plan["run_id"])
    return proof


def _invalidated(proof: PreflightProof, overlay: Path) -> None:
    invalidate_run(proof, run_id=proof.run_id, cause="ENOSPC", partial_path=overlay)
    raise RunInvalidated("R1 overlay install invalidated by ENOSPC; use a new run ID")


def install_after_admission(plan_path: Path, overlay: Path, install: Callable[[], None]) -> PreflightProof:
    """Run the material installer only under one admitted run; ENOSPC stops it permanently."""
    proof = admit_overlay_install(plan_path, overlay)
    try:
        Path(overlay).mkdir(parents=True, exist_ok=True)
        install()
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            _invalidated(proof, Path(overlay))
        raise
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        if "no space left on device" in error.lower():
            _invalidated(proof, Path(overlay))
        raise
    return proof


def install_command(plan_path: Path, overlay: Path, command: Sequence[str]) -> PreflightProof:
    if not command:
        raise StorageBlocked("installer command is required")
    return install_after_admission(plan_path, overlay, lambda: subprocess.run(command, check=True, capture_output=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        proof = install_command(args.plan, args.overlay, command)
    except (OSError, StorageBlocked, RunInvalidated, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "STOPPED_INCOMPLETE", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "READY", "run_id": proof.run_id, "overlay": str(args.overlay)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
