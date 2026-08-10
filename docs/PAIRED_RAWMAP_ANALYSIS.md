# Frozen paired raw-map analysis

The paired analysis replays only the same TESTpub114 E2-Split and posthoc SuperADD/DINOv3 raw maps with their ground-truth masks at 528×2112. It verifies map, source, and mask hashes before computing per-image tie-aware pixel AUROC, mask geometry descriptors, within-model score-rank deltas, descriptive quantile strata, and Spearman associations. It performs no inference, retraining, threshold selection, or test tuning.

Its findings are measured associations, not architecture-causality or model-selection claims. Raw score magnitudes are not compared across pipelines; image scores are rank-normalized within each pipeline. Visual panels contain anonymized raw-map-only representations—never source imagery or local paths.

Mask compactness uses the 4-connected digital perimeter: `4πA/P²`, where `P` counts exposed north/south/east/west pixel edges. Dataset-relative low/middle/high terciles include their exact cutpoints, counts, and mean paired pixel-AUROC deltas in the canonical evidence; they are descriptive only.
