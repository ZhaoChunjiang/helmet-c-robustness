# Public V3.1 reproduction utilities

These scripts implement the final manuscript V3.6 Helmet-C-Val evaluation boundary.

## 1. Determinism audit

```bash
python scripts/v31/run_v31_determinism_audit.py \
  --data /path/to/data.yaml \
  --workers 12 \
  --output /path/to/V31_DETERMINISM_AUDIT.json
```

The audit checks all 15 corruption families across fresh worker processes and performs complete 607-image two-pass reproduction for `impulse_noise s1` and `glass_blur s1`.

## 2. Fresh V3.1 evaluation

```bash
python scripts/v31/evaluate_helmet_c_val_v31.py \
  --data /path/to/data.yaml \
  --audit-marker /path/to/V31_DETERMINISM_AUDIT.json \
  --model A0_seed0 /path/to/best.pt 0.6089874629020515 \
  --output /path/to/v31_eval
```

Repeat `--model LABEL BEST_PT EXPECTED_CLEAN_AP5095` for multiple frozen checkpoints. Each corruption condition is materialized once as lossless PNG and all requested models are evaluated with native Ultralytics `model.val()`.

## 3. Repository final-lock verification

```bash
python scripts/v31/verify_v31_final_table.py
```

This verifies the exact V3.1/V3.6 machine-readable aggregate values committed under `results/V3.1_final_4x3/`.

## Notes

- Training seed and corruption seed are different concepts. The training seeds are `0`, `42`, and `3407`; the corruption base seed is fixed at `3407`.
- V3.1 is an evaluation implementation correction. It does not change or retrain the frozen checkpoints.
- The public evaluator is a clean, path-portable implementation of the final protocol. The original AutoDL run scripts contained machine-local paths and are retained in the authors' run archive rather than treated as the portable entry point.
