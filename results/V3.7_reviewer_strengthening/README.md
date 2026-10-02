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

Protocol ID: `helmet-c-val-v3.1-deterministic-impulse-audited-glass-2026-09-29`
