"""Storage admission boundary for the R1 package overlay installer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .storage import Allocation, PreflightProof, StorageBlocked, _mount, preflight, require_proof, roots_from_env


def admit_overlay_install(plan_path: Path, overlay: Path) -> PreflightProof:
    """Require a fresh source-backed ext4 proof before any overlay write."""
    plan = json.loads(Path(plan_path).read_text())
    roots = roots_from_env()
    overlay = Path(overlay).resolve()
    if overlay.parent != roots["venv"].resolve() or _mount(overlay.parent)[1] != "ext4":
        raise StorageBlocked("R1 overlay target must be directly under the configured ext4 venv root")
    allocations = [Allocation(**item) for item in plan["allocations"]]
    if not any(item.root == "venv" and item.kind == "persistent" for item in allocations):
        raise StorageBlocked("source-backed plan must allocate the persistent venv overlay")
    proof = preflight(run_id=plan["run_id"], allocations=allocations, reserve_bytes=plan["reserve_bytes"],
                      reserve_evidence=plan["reserve_evidence"])
    require_proof(proof, run_id=plan["run_id"])
    return proof


def install_after_admission(plan_path: Path, overlay: Path, install: Callable[[], None]) -> PreflightProof:
    """Testable order boundary: no package callback executes without admission."""
    proof = admit_overlay_install(plan_path, overlay)
    install()
    return proof
