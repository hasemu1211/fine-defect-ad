# Frozen paired raw-map analysis

The paired analysis replays only the same TESTpub114 E2-Split and posthoc SuperADD/DINOv3 raw maps with their ground-truth masks at 528×2112. It verifies map, source, and mask hashes before computing per-image tie-aware pixel AUROC, mask geometry descriptors, within-model score-rank deltas, descriptive quantile strata, and Spearman associations. It performs no inference, retraining, threshold selection, or test tuning.

Its findings are measured associations, not architecture-causality or model-selection claims. Raw score magnitudes are not compared across pipelines; image scores are rank-normalized within each pipeline. Visual panels contain anonymized raw-map-only representations—never source imagery or local paths.
