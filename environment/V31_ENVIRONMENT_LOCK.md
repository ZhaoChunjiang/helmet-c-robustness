# V3.1 Environment and Evaluation Lock

This file records the **current manuscript V3.6 / evaluator V3.1 public evaluation contract**.

Protocol ID:

```text
helmet-c-val-v3.1-deterministic-impulse-audited-glass-2026-09-29
```

## Scope

This repository exposes a portable evaluation/protocol package, not an end-to-end training reproduction package.

It provides:

- fixed SHWD split manifests;
- deterministic V3.1 corruption generation;
- native Ultralytics validation utilities;
- protocol notes;
- machine-readable V3.6 result locks.

It does **not** redistribute SHWD source images or trained checkpoints, and it does not contain the full A0/A1/A1-WH/A1-R10 training implementation.

## Captured core environment

The original formal records capture the following core stack:

```text
Operating system      Ubuntu 22.04
Python                3.10.8
NumPy                 1.26.4
PyTorch               2.1.2+cu118
PyTorch CUDA runtime  11.8
Ultralytics           8.4.140
imagecorruptions      1.1.2
GPU                   NVIDIA GeForce RTX 4090
NVIDIA driver         570.124.04
GPU memory            24564 MiB
```

The older `environment/FORMAL_ENVIRONMENT.md` documents the superseded pre-V3.1 chain and is retained only for provenance.

## Dependency-capture limitation

The original V3.1 run artifacts did **not** preserve exact versions of every transitive image-stack dependency, in particular scikit-image, Pillow, and torchvision. Those versions are therefore not reconstructed or guessed here.

`requirements.txt` is a convenience installation manifest for the public V3.1 utilities, not a claim of byte-for-byte recreation of the original environment.

The V3.1 corruption utility includes the narrow compatibility shim required by imagecorruptions 1.1.2 and uses an explicit RNG for impulse noise. A reproducibility audit should be run after constructing any replacement environment.

## Public V3.1 evaluation settings

The portable V3.1 evaluator explicitly uses:

```text
split      = val
images     = 607
objects    = 9,925
imgsz      = 640
batch      = 16
workers    = 8
rect       = False
device     = 0 (manuscript run; public script exposes --device, default 0)
half       = False
conf       = 0.001
NMS IoU    = 0.7
max_det    = 300
```

Each corruption condition is materialized as lossless PNG and evaluated through native Ultralytics `model.val()`.

The corruption base seed is `3407`. Per-image seeds are derived from:

```text
SHA256(base_seed | image_id | corruption | severity) -> uint32
```

The public V3.1 identity is the filename stem. The committed 607-image validation manifest contains 607 unique stems. The evaluator now rejects duplicate stems instead of silently allowing seed/output collisions.

## Random-state behavior

`deterministic_corruptions_v31.py` seeds NumPy and Python random state for each corruption call and restores the previous global states on exit. This prevents corruption generation from perturbing unrelated random operations in the same process while preserving the V3.1 image outputs.

## Determinism audit

The manuscript-era V3.1 acceptance rule was:

- fresh-process probes over all 15 corruption families;
- full 607-image two-pass reproduction for `impulse_noise s1`;
- full 607-image two-pass reproduction for `glass_blur s1`.

The public utility additionally supports:

```bash
python scripts/v31/run_v31_determinism_audit.py \
  --data /path/to/data.yaml \
  --workers 12 \
  --full-all \
  --output /path/to/V31_DETERMINISM_AUDIT_FULL75.json
```

This optional mode performs two full passes over all 75 corruption/severity conditions. It is a post-lock hardening facility; it does not imply that the original manuscript run performed 75 complete double replays.

## Validation-set status

The V3.6 headline robustness numbers are reported on the fixed 607-image SHWD validation split.

The training pipeline uses Ultralytics training and the frozen `best.pt` checkpoints. The validation split is therefore part of the training-time model-selection process and is **not an untouched blind test set**. V3.6 deliberately labels the benchmark as Helmet-C-Val and does not present the V3.1 corruption numbers as independent-test results.

The 1,517-image Test split and older pre-V3.1 Helmet-C-Test outputs are not used as V3.6 headline corruption evidence.

## Result-lock boundary

`results/V3.1_final_4x3/` is a manuscript result lock. Its table-verification CI checks the committed aggregate values for drift; it does not load images, rerun checkpoints, or recompute AP.

Two A1-R10 checkpoint hashes (training seeds 42 and 3407) were not present in the synchronized provenance artifacts and remain blank rather than being reconstructed or guessed.

## Metric naming

The 11-family non-weather average excludes:

```text
snow
frost
fog
elastic_transform
```

The preferred manuscript-facing name is **nonweather11-mPC**. Existing machine-readable locks may retain the legacy key `seen11_mPC` for backward compatibility. The two names refer to the same numerical 11-family set; the legacy label must not be interpreted as a universal training-seen set across all models.


## Remaining provenance limitations

Two A1-R10 checkpoint SHA256 values (training seeds 42 and 3407) were not preserved in the synchronized export artifacts and remain unavailable in the public result lock. They are not guessed or reconstructed.

The original run did not capture exact versions of all transitive imaging dependencies (notably scikit-image, Pillow, and torchvision). Consequently, a newly constructed environment can be audited for internal determinism, but the public files alone cannot establish byte-for-byte image identity with every original corruption realization.
