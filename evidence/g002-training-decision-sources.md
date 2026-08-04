# G002 training decision sources

## Decision

The G002 1,000-step pilot is a systems pilot only: it measures runtime stability, resource use, ETA inputs, and checkpoint cadence. It does not establish convergence, `min_delta`, or patience. The fixed 70,000-step EfficientAD-S schedule remains the candidate protocol; there is no loss-plateau early stop or best-validation-loss checkpoint selection. Hard integrity failures (for example non-finite values, OOM, invalid provenance, or failed checkpoint persistence) stop the run without selecting a model.

Normal validation may be used only for normal-score/calibration/stability checks. `TESTpub`, `TESTpriv`, `TESTpriv,mix`, and any OOD identity, label, mask, score, or defect geometry are excluded from training, checkpoint, stop, geometry, threshold, and serving selection.

## Primary sources

- EfficientAD trains the student from anomaly-free normal images: <https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf>
- Anomalib EfficientAd uses a finite training schedule and decays LR at 95% of the computed training steps: <https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/efficient_ad.html>
- The pinned anomalib EfficientAd implementation logs training losses, while its validation path computes normal validation-map quantiles rather than a validation loss: <https://github.com/open-edge-platform/anomalib/blob/3759687e76395c4d6d239552d3bf6d72e003da78/src/anomalib/models/image/efficient_ad/lightning_model.py>
- Lightning early stopping requires an explicitly monitored metric; patience counts validation checks, not training epochs: <https://pytorch-lightning.readthedocs.io/en/2.2.4/pytorch/api/lightning.pytorch.callbacks.EarlyStopping.html>
- Hyperband and ASHA allocate training resources across candidate hyperparameter configurations; they are not evidence for selecting a plateau rule in one fixed run: <https://www.jmlr.org/papers/volume18/16-558/16-558.pdf>, <https://proceedings.mlsys.org/paper_files/paper/2020/hash/a06f20b349c6cf09a6b171c71b88bbfc-Abstract.html>

Any later convergence-based stop rule requires both literature support and a separately recorded longer curve using only official train data and normal validation data. It must state its monitored proxy, validation-check interval, `min_delta`, patience, seed/repeat policy, and fixed-schedule comparison before the run starts.
