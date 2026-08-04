#!/usr/bin/env bash
set -euo pipefail
: "${FINE_DEFECT_HOST_PYTHON:?set to the approved base Python}"
: "${FINE_DEFECT_VENV_ROOT:?set to an ext4 writable root}"
overlay="$FINE_DEFECT_VENV_ROOT/r1-overlay"
mkdir -p "$overlay"
"$FINE_DEFECT_HOST_PYTHON" -m pip install \
  --target "$overlay" --no-deps --require-hashes --no-cache-dir --no-compile \
  -r requirements/r1-overlay.txt
