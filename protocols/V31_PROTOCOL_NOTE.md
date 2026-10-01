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

Pre-V3.1 corrupted-image metrics are retained for provenance only. In particular, old mPC/rPC, old scale-stratified corrupted metrics, old YOLOv8n corruption mPC/rPC, and old independent Helmet-C-Test corruption numbers are not used as V3.6 headline evidence.


## Public reproducibility scope

The public repository provides the V3.1 evaluation protocol, split manifests, corruption/evaluation utilities, and machine-readable result locks. It is not an end-to-end training reproduction package: SHWD images, trained checkpoints, and the full training implementation are not redistributed.

The 607-image Helmet-C-Val split is the same validation split used during training-time model selection for the frozen `best.pt` checkpoints. It is therefore not an untouched blind test set. V3.6 reports this split explicitly as validation and does not use older pre-V3.1 Helmet-C-Test corruption metrics as headline evidence.

The preferred name for the 11-family non-weather summary is `nonweather11-mPC`. Existing result locks may retain the legacy key `seen11_mPC` for backward compatibility; the underlying set excludes snow, frost, fog, and elastic transform.
