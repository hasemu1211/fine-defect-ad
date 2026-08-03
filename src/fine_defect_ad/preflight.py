"""CLI boundary: a source-backed plan is required; there is no implicit zero budget."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from .storage import Allocation, StorageBlocked, preflight

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', help='JSON source-backed allocation plan')
    args = parser.parse_args(argv)
    try:
        if not args.plan: raise StorageBlocked('source-backed --plan is required')
        plan = json.loads(Path(args.plan).read_text())
        allocations = [Allocation(**item) for item in plan['allocations']]
        proof = preflight(run_id=plan['run_id'], allocations=allocations, reserve_bytes=plan['reserve_bytes'], reserve_evidence=plan['reserve_evidence'])
        print(json.dumps(proof.__dict__, sort_keys=True, default=list)); return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, StorageBlocked) as exc:
        print(json.dumps({'status':'STORAGE_BLOCKED','workflow_status':'STOPPED_INCOMPLETE','reason':str(exc)})); return 2

if __name__ == '__main__': raise SystemExit(main())
