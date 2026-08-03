# FineDefect AD

R0 only: storage gating, dataset-manifest integrity, evidence schemas, no-sudo runtime evidence, and a native GPU-heavy-job lock. No model, dataset body, training, serving, or result claim exists.

The preflight command requires a source-backed JSON allocation plan; it never supplies a zero budget:

```bash
PYTHONPATH=src python3 -m fine_defect_ad.preflight --plan preflight-plan.json
```

`preflight-plan.json` must contain `run_id`, `allocations` (`source`, `component_id`, `root`, `kind`, `bytes`), `reserve_bytes`, and `reserve_evidence` (`max_pending_atomic_write_bytes`, `measured_high_water_bytes`, `runtime_or_source_citation`). It exits `2` with structured `STORAGE_BLOCKED` without a plan, required roots, Docker daemon evidence, or sufficient capacity. It never cleans, migrates, mounts, or repairs storage.
