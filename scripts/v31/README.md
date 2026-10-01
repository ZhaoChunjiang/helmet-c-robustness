# Public V3.1 reproduction utilities

These scripts implement the final manuscript V3.6 Helmet-C-Val **evaluation** boundary.

They are not a full training reproduction package: SHWD source images and trained checkpoints are not bundled here, and the full training code for A0/A1/A1-WH/A1-R10 is not part of this public repository.

## 1. Determinism audit

```bash
python scripts/v31/run_v31_determinism_audit.py \
  --data /path/to/data.yaml \
  --workers 12 \
  --output /path/to/V31_DETERMINISM_AUDIT.json
```

The default audit checks all 15 corruption families on a fresh-process probe and performs complete 607-image two-pass reproduction for `impulse_noise s1` and `glass_blur s1`, matching the original V3.1 acceptance rule.

For a stronger environment check, the public utility can additionally replay **all 75 corruption/severity conditions** twice over all 607 images:

```bash
python scripts/v31/run_v31_determinism_audit.py \
  --data /path/to/data.yaml \
  --workers 12 \
  --full-all \
  --output /path/to/V31_DETERMINISM_AUDIT_FULL75.json
```

The optional full-75 audit is a post-lock hardening utility; it does not retroactively claim that the original manuscript run performed 75 full-set double replays.

## 2. Fresh V3.1 evaluation

```bash
python scripts/v31/evaluate_helmet_c_val_v31.py \
  --data /path/to/data.yaml \
  --audit-marker /path/to/V31_DETERMINISM_AUDIT.json \
  --model A0_seed0 /path/to/best.pt 0.6089874629020515 \
  --output /path/to/v31_eval
```

Repeat `--model LABEL BEST_PT EXPECTED_CLEAN_AP5095` for multiple frozen checkpoints. Each corruption condition is materialized once as lossless PNG and all requested models are evaluated with native Ultralytics `model.val()`.

The public V3.1 rerun path explicitly fixes `imgsz=640`, `batch=16`, `workers=8`, `rect=False`, `device=0`, `conf=0.001`, `iou=0.7`, and `max_det=300`. The fixed SHWD Val manifest contains 607 unique filename stems; the evaluator now rejects duplicate stems because V3.1 uses the stem as its image identity.

## 3. Fresh multi-seed aggregation

After rerunning frozen checkpoints for multiple training seeds, aggregate the resulting `summary.json` files with:

```bash
python scripts/v31/aggregate_v31_three_seed.py \
  --record A0 0 A0_seed0 /runs/a0_s0/summary.json \
  --record A0 42 A0_seed42 /runs/a0_s42/summary.json \
  --record A0 3407 A0_seed3407 /runs/a0_s3407/summary.json \
  --output /runs/A0_aggregate
```

The utility computes mean and **sample** standard deviation from the fresh evaluator outputs and emits rPC both as a ratio and as a percentage display value.

## 4. Repository final-lock verification

```bash
python scripts/v31/verify_v31_final_table.py
```

This verifies the exact V3.1/V3.6 machine-readable aggregate values committed under `results/V3.1_final_4x3/`. It is an **integrity check of committed result files**, not a model rerun and not evidence of end-to-end reproduction.

## Notes

- Training seed and corruption seed are different concepts. The training seeds are `0`, `42`, and `3407`; the corruption base seed is fixed at `3407`.
- V3.1 is an evaluation implementation correction. It does not change or retrain the frozen checkpoints.
- The public evaluator is a clean, path-portable implementation of the final protocol. The original AutoDL run scripts contained machine-local paths and are retained in the authors' run archive rather than treated as the portable entry point.

## Metric naming

The 11-family non-weather average excludes `snow`, `frost`, `fog`, and `elastic_transform`. New evaluator output exposes this as `nonweather11_mPC`. The legacy JSON alias `seen11_mPC` is retained so older result consumers do not break; it should not be interpreted as a universal training-seen set for every model.

## Environment capture limitation

The original V3.1 run artifacts recorded the core stack (Python, NumPy, PyTorch/CUDA, Ultralytics, imagecorruptions, GPU/driver) but did not preserve exact versions of every transitive imaging dependency such as scikit-image, Pillow, and torchvision. Those versions are therefore not guessed in this repository. See `../../environment/V31_ENVIRONMENT_LOCK.md`.
