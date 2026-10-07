# Helmet-C-Val V3.1 deterministic protocol

Protocol ID:

```text
helmet-c-val-v3.1-deterministic-impulse-audited-glass-2026-09-29
```

## Why V3.1 supersedes the older evaluator

During the final audit, two implementation issues were identified in the earlier corruption-evaluation chain:

1. RGB arrays generated from PIL/imagecorruptions were passed directly to a NumPy inference path that Ultralytics interprets as BGR, causing a material color-order error in corrupted-image evaluation.
2. `impulse_noise` reproducibility was not guaranteed by `np.random.seed(...)` alone with the installed scikit-image behavior.

The frozen training checkpoints are unchanged. V3.1 is a post-hoc evaluation implementation correction.

## V3.1 rules

- Headline split: fixed SHWD Val, 607 images / 9,925 objects.
- 15 corruption families × severities 1–5 = 75 conditions.
- Corruption seed base: `3407`.
- Stable per-image seed: SHA256 of `base_seed|image_id|corruption|severity`, converted to uint32.
- Each corruption condition is materialized as lossless PNG.
- Detection metrics are obtained through native Ultralytics `model.val()` rather than a custom AP calculator.
- Clean AP must match the frozen checkpoint reference within the configured tolerance.
- `impulse_noise` uses an explicit RNG.
- `impulse_noise s1` and `glass_blur s1` must each pass a complete 607-image two-pass reproducibility audit before a formal V3.1 run is accepted.

## Final manuscript metrics

The final manuscript lock is under:

```text
results/V3.1_final_4x3/
```

## Legacy boundary

Pre-V3.1 corrupted-image metrics are retained for provenance only. In particular, old mPC/rPC, old scale-stratified corrupted metrics, old YOLOv8n corruption mPC/rPC, and old independent Helmet-C-Test corruption numbers are not used as current manuscript headline evidence.


## Public reproducibility scope

The public repository provides the V3.1 evaluation protocol, split manifests, corruption/evaluation utilities, and machine-readable result locks. It is not an end-to-end training reproduction package: SHWD images, trained checkpoints, and the full training implementation are not redistributed.

The 607-image Helmet-C-Val split is the same validation split used during training-time model selection for the frozen `best.pt` checkpoints. It is therefore not an untouched blind test set. The current manuscript reports this split explicitly as validation and does not use older pre-V3.1 Helmet-C-Test corruption metrics as headline evidence.

The preferred name for the 11-family non-weather summary is `nonweather11-mPC`. Existing result locks may retain the legacy key `seen11_mPC` for backward compatibility; the underlying set excludes snow, frost, fog, and elastic transform.

## rPC aggregation convention

For each training seed, relative corruption performance is computed **before** cross-seed aggregation:

- `rPC15_seed = mPC15_seed / clean_AP_seed`
- `weather_rPC_seed = weather_mPC_seed / clean_AP_seed`

The manuscript result lock then reports the **mean and sample standard deviation of these seed-level ratios** across seeds 0, 42, and 3407. It does not compute rPC by dividing the across-seed mean mPC by the across-seed mean clean AP.


## V3.7 protocol reconciliation

During reviewer-strengthening, the frozen A0 seed-0 clean reference was rerun under several native Ultralytics validation settings. The original formal V3.1 call, with `rect` and `half` omitted, reproduced clean AP50:95 = `0.6089874629020515` exactly. Explicitly forcing `rect=False` changed the metric. The public V3.1 evaluator therefore preserves the historical call by leaving those two arguments unspecified.

This reconciliation changes no frozen checkpoint and no committed V3.1 aggregate result; it corrects the portable wrapper so that its validation semantics match the historical numerical path.
