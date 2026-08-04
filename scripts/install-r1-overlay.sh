#!/usr/bin/env bash
set -euo pipefail
: "${FINE_DEFECT_HOST_PYTHON:?set to the approved base Python}"
: "${FINE_DEFECT_VENV_ROOT:?set to an ext4 writable root}"
: "${FINE_DEFECT_STORAGE_PLAN:?set to a source-backed storage plan}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
overlay="$FINE_DEFECT_VENV_ROOT/r1-overlay"
PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$FINE_DEFECT_HOST_PYTHON" - <<PY
from fine_defect_ad.r1_overlay import admit_overlay_install
from pathlib import Path
admit_overlay_install(Path("$FINE_DEFECT_STORAGE_PLAN"), Path("$overlay"))
PY
mkdir -p "$overlay"
"$FINE_DEFECT_HOST_PYTHON" -m pip install \
  --target "$overlay" --no-deps --require-hashes --no-cache-dir --no-compile \
  -r "$repo/requirements/r1-overlay.txt"
PYTHONPATH="$overlay" "$FINE_DEFECT_HOST_PYTHON" - <<'PY'
import json
import anomalib, cv2, kornia, lightning, timm, torch, torchvision
from anomalib.models import EfficientAd
print(json.dumps({"status": "READY", "anomalib": anomalib.__version__, "efficient_ad": EfficientAd.__name__,
                  "lightning": lightning.__version__, "kornia": kornia.__version__, "opencv": cv2.__version__,
                  "timm": timm.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__,
                  "cuda_available": torch.cuda.is_available()}, sort_keys=True))
PY
