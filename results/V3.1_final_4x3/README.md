# Final V3.1 / manuscript V3.7 robustness lock

This directory is the current machine-readable manuscript-level robustness result.

Use these files for the V3.7 paper:

- `model_stats.csv` — 4 models × three-seed mean and sample standard deviation.
- `paired_deltas.csv` — paired A1-R10 − A1 differences used in the pairing-diversity control.
- `protocol.json` — evaluator/split/protocol lock.
- `checkpoint_identities.csv` — frozen checkpoint identities where they were available in the synchronized provenance artifacts.

The headline evaluation is Helmet-C-Val on the fixed 607-image SHWD validation split under deterministic evaluator V3.1.

Older robustness outputs under `results/three_seed/`, `results/formal/`, and `results/final_validation/` are **pre-V3.1 provenance artifacts and are superseded for headline robustness metrics**.

The A1-R10 seed-42 and seed-3407 checkpoint hashes were absent from the earlier synchronized export, but were recovered from the original AutoDL workspace during the V3.7 reviewer-strengthening audit and are now recorded in `checkpoint_identities.csv`. The aggregate robustness values were not changed.


## Interpretation of this directory

This directory is a **manuscript result lock**, not a substitute for rerunning the models. The table-verification script and CI check the committed aggregate values for drift; they do not load images, checkpoints, or recompute AP.

The public repository does not claim end-to-end training reproducibility. SHWD source images, trained checkpoints, and the full training implementation are not bundled.

The metric historically stored as `seen11_mPC` is the 11-family non-weather average obtained by excluding snow, frost, fog, and elastic transform. Manuscript-facing text should call it **nonweather11-mPC** to avoid implying a universal training-seen set across all four models.

Helmet-C-Val is the fixed 607-image validation benchmark. Because the frozen `best.pt` checkpoints are selected with the validation split during training, these V3.1 robustness numbers are not presented as blind independent-test estimates.
