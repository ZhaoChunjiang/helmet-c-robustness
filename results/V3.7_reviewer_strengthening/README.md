# V3.7 reviewer-strengthening results

These files supplement, but do **not** replace, the native V3.1 4-model × 3-seed result lock under `results/V3.1_final_4x3/`.

They were generated in response to pre-submission review requests for:

- elastic-transform sensitivity (`mPC14`);
- scale-stratified small-object evidence;
- a second-detector replication (YOLOv8n);
- final V3.1 detection endpoints for the C0/A3 mechanism screen;
- protocol reconciliation of the native Ultralytics validation call.

## Evidence boundary

- Helmet-C-Val uses the fixed 607-image validation split; it is not a blind test set.
- Native V3.1 headline metrics remain authoritative for aggregate ALL results.
- Scale-stratified values come from a separately calibrated post-hoc evaluator and are not substituted for native ALL values.
- The C0/A3 endpoint is a pre-specified 8-condition screen (Blur6 + Noise2), not a 75-condition mechanism benchmark.
- YOLOv8n is a seed-0 cross-architecture replication, not a multi-seed architecture benchmark.
- In `yolov8n_replication.csv`, the `delta_A1_minus_A0_pp` row is a derived A1−A0 difference from the two seed-0 runs; its `seed` field is intentionally blank to distinguish a derived row from an evaluated regime/seed record.

Protocol ID: `helmet-c-val-v3.1-deterministic-impulse-audited-glass-2026-09-29`

## C0/A3 mechanism definitions

The manuscript-facing C0/A3 mechanism screen corresponds to Sections 3.5 and 5.5 of the submitted manuscript.

- **C0 (matched paired control):** six-epoch fine-tuning on the same clean/degraded image pairs used by A3, with the same detection-loss budget but **without** a representation-consistency term.
- **A3 (ObjCons):** the matched paired fine-tuning condition with an exponential-moving-average clean teacher (`mu = 0.999`) and a degraded student. The student uses the same detection losses as C0 plus ROI-level cosine consistency with `beta = 0.5`.
- **Representation discrepancy:** the object-level clean/degraded cosine distance used by the manuscript consistency loss,
  `L_OC = (1/N) sum_i (1/L) sum_l [1 - cos(z^d_i,l, sg(z^c_i,l))]`, averaged over matched objects and feature levels. The reported **32.77% ± 0.44%** value is the across-seed relative reduction in this discrepancy for A3 versus C0; it is not an AP metric.
- **Detection endpoint:** the committed C0/A3 AP values are from the pre-specified **Screen8** only (six blur + two severe-noise conditions), not from the full 75-condition Helmet-C mechanism benchmark.

The corresponding machine-readable AP endpoints are in `mechanism_screen.csv`.

## Small-sample reporting note

YOLO11n multi-seed summaries use three training seeds (0, 42, 3407) and report the sample standard deviation across those three values. These SDs are descriptive across the tested seeds and are **not** confidence intervals; their displayed precision should not be interpreted as inferential certainty from a large sample.
