"""Domain-facing high-resolution validation evaluation CLI."""
from __future__ import annotations
from typing import Sequence
from .g002_e2_runtime import main as _main

def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv)

if __name__ == "__main__": raise SystemExit(main())
