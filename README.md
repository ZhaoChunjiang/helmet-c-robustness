# Helmet-C Robustness

[![V3.1 Final Table Verification](https://github.com/ZhaoChunjiang/helmet-c-robustness/actions/workflows/v31-final-verification.yml/badge.svg)](https://github.com/ZhaoChunjiang/helmet-c-robustness/actions/workflows/v31-final-verification.yml)

Public evaluation-protocol and result-lock materials for the manuscript:

**Robustness evaluation and mechanism analysis of small-object safety-helmet detection under synthetic common corruptions**

## Scope of this repository

This repository is **not an end-to-end training reproduction package**. It does not redistribute the SHWD source images or trained checkpoints, and it does not contain the full training code used to produce A0, A1, A1-WH, and A1-R10.

What it does provide is the final **V3.1 evaluation protocol**, fixed split manifests, deterministic corruption/evaluation utilities, protocol notes, and machine-readable V3.6 manuscript result locks. For the V3.6 manuscript, the **only current public evaluation entry point is `scripts/v31/`**. Older generators, runners, and result folders are retained only as provenance and must not be used to reproduce current headline robustness numbers.

A static lock-table check verifies committed result files only; it does **not** rerun models or constitute end-to-end experimental reproduction.

## Final manuscript protocol (V3.6 / evaluator V3.1)

The current headline results use the fixed **SHWD validation split (607 images, 9,925 objects)** and the deterministic Helmet-C-Val protocol.

Helmet-C contains 15 synthetic corruption types at five severity levels (75 corruption/severity conditions): Gaussian, shot, and impulse noise; defocus, glass, motion, and zoom blur; snow, frost, and fog; brightness and contrast; elastic transform; pixelate; and JPEG compression.

For corruption type `c` and severity `s`, the primary score is AP50:95. The main summary metrics are:

```text
mPC15 = mean AP50:95 over all 15 × 5 = 75 conditions
rPC15 = mPC15 / clean AP50:95
```

The V3.1 evaluator materializes each corrupted condition as **lossless PNG** and evaluates it through native Ultralytics `model.val()`. Corruption generation uses a stable per-image seed derived from:

```text
base corruption seed = 3407
image identity + corruption type + severity
```

with a SHA256-based mapping. `impulse_noise` uses an explicit RNG. Before a formal run is accepted, `impulse_noise s1` and `glass_blur s1` must pass a full **607-image × 2-pass** reproducibility audit.

## Final 4-model × 3-seed results

Training seeds are `0`, `42`, and `3407`. Values below are mean ± sample standard deviation across the three training seeds.

| Metric | A0 | A1 (Corr-Aug) | A1-WH | A1-R10 |
|---|---:|---:|---:|---:|
| clean AP50:95 | 0.609093 ± 0.000260 | 0.609589 ± 0.000865 | 0.610760 ± 0.001022 | 0.607964 ± 0.001196 |
| mPC15 | 0.385045 ± 0.001119 | 0.520876 ± 0.000561 | 0.518365 ± 0.000720 | **0.531700 ± 0.000762** |
| rPC15 | 63.22% ± 0.16% | 85.45% ± 0.07% | 84.87% ± 0.26% | **87.46% ± 0.09%** |
| weather-mPC | 0.452176 ± 0.006166 | 0.558918 ± 0.001035 | 0.532987 ± 0.001188 | **0.566016 ± 0.000378** |
| weather-rPC | 74.24% ± 0.98% | 91.69% ± 0.07% | 87.27% ± 0.34% | **93.10% ± 0.22%** |
| nonweather11-mPC | 0.352347 ± 0.000517 | 0.508199 ± 0.000465 | 0.511669 ± 0.000761 | **0.520941 ± 0.001126** |

Machine-readable copies are under `results/V3.1_final_4x3/`.

## Controls added for manuscript V3.6

### A1-WH: held-out synthetic weather family

A1-WH keeps the A1 architecture, optimizer, seeds, 1:1 clean/degraded ratio, epoch count, and total image-presentation budget, but excludes `snow`, `frost`, and `fog` from the degraded training pool.

Its weather-mPC is:

```text
A0      0.452176 ± 0.006166
A1-WH   0.532987 ± 0.001188
```

This is evidence of **partial transfer to this deliberately held-out synthetic weather family**. It is not evidence of arbitrary unseen-domain or real-world OOD robustness.

### A1-R10: frozen multi-view pairing-diversity control

A1 uses one fixed degraded counterpart per clean source image. A1-R10 uses ten frozen pairing views while preserving the same total presentation budget and the same corruption/severity composition.

Paired A1-R10 − A1 differences across the three seeds are:

| Metric | Paired difference |
|---|---:|
| clean AP50:95 | −0.162 ± 0.157 pp |
| mPC15 | +1.082 ± 0.102 pp |
| weather-mPC | +0.710 ± 0.129 pp |
| nonweather11-mPC | +1.274 ± 0.134 pp |

A1-R10 is a **finite frozen 10-view control**, not fully online i.i.d. augmentation.

## Fixed SHWD split

| Split | Images | hat | person* | Total objects |
|---|---:|---:|---:|---:|
| Train | 5457 | 6419 | 79778 | 86197 |
| Val | 607 | 747 | 9178 | 9925 |
| Test | 1517 | 1878 | 22558 | 24436 |
| Total | 7581 | 9044 | 111514 | 120558 |

`person` is the original SHWD class name for the non-helmeted-head negative class, not a full-body pedestrian class.

## Repository structure

```text
assignments/                 frozen training assignments retained for provenance
configs/                     Helmet-C configuration
splits/                      fixed SHWD split manifests
scripts/v31/                 final deterministic V3.1 public reproduction utilities
protocols/                   V3.1 protocol notes and supersession record
results/V3.1_final_4x3/      final manuscript-level machine-readable results
results/three_seed/          LEGACY pre-V3.1 outputs retained for provenance
results/formal/              LEGACY pre-V3.1 independent-test artifacts
results/final_validation/    LEGACY pre-V3.1 manuscript summary
```

## V3.1 public evaluation entry points

The current public V3.1 utilities are:

```text
scripts/v31/deterministic_corruptions_v31.py
scripts/v31/run_v31_determinism_audit.py
scripts/v31/evaluate_helmet_c_val_v31.py
scripts/v31/aggregate_v31_three_seed.py
scripts/v31/verify_v31_final_table.py
```

Run the static/final-table integrity verification with:

```bash
python scripts/v31/verify_v31_final_table.py
```

For a fresh Helmet-C-Val rerun, first run the determinism audit and then evaluate user-supplied frozen checkpoint(s). The original SHWD images and trained checkpoints are not bundled here. See `scripts/v31/README.md`.

`verify_v31_final_table.py` checks that the committed machine-readable result lock has not drifted; it does not load images, checkpoints, or recompute AP.

## V3.1 environment record

The current V3.1 environment/protocol record is documented in `environment/V31_ENVIRONMENT_LOCK.md`. The older `environment/FORMAL_ENVIRONMENT.md` is a **pre-V3.1 provenance record** and is not the V3.6 evaluation entry point.

The captured core environment was:

```text
Ubuntu              22.04
Python              3.10
PyTorch             2.1.2+cu118
Ultralytics         8.4.140
NumPy               1.26.4
imagecorruptions    1.1.2
input resolution    640 × 640
GPU                 NVIDIA RTX 4090
```

The original SHWD images and trained checkpoints are not redistributed here. Exact versions of some transitive image-stack dependencies (notably scikit-image, Pillow, and torchvision) were not captured in the original V3.1 run artifact; the repository records this limitation rather than inventing versions.

The public V3.1 evaluator explicitly freezes the manuscript-critical validation settings used by the portable rerun path. See `environment/V31_ENVIRONMENT_LOCK.md` for the distinction between recorded original-run facts and the current public rerun contract.

## Metric naming note

The 11-family non-weather summary is the mean over the 15 Helmet-C families after excluding `snow`, `frost`, `fog`, and `elastic_transform`. In the manuscript-facing terminology this is **nonweather11-mPC**. Some already-committed machine-readable lock files retain the legacy key `seen11_mPC` for backward compatibility; `seen11_mPC` is a compatibility alias for `nonweather11_mPC` and must not be interpreted as a universal "training-seen" set for every model.

## Legacy results and scripts: important warning

The folders `results/three_seed/`, `results/formal/`, and the older `results/final_validation/` were produced before the final V3.1 evaluator correction. They are retained **only for provenance**.

> **SUPERSEDED FOR HEADLINE ROBUSTNESS METRICS.**
> Do not use the old mPC/rPC, scale-stratified corrupted-AP, old independent-test corruption metrics, or the old pre-V3.1 Table 2 as the current manuscript result.

The legacy `scripts/generate_helmet_c.py` seed convention and the pre-V3.1 v1.0.4 runner belong to the superseded development chain. They are retained for provenance only. They are **not** interchangeable with `scripts/v31/` and must not be used to regenerate V3.6 headline conditions.

The clean AP values remain useful provenance where explicitly referenced, but the manuscript V3.6 robustness claims are locked to `results/V3.1_final_4x3/`.

## Interpretation boundary

Helmet-C is a synthetic single-corruption benchmark. A1-WH holds out one synthetic weather family. A1-R10 tests finite pairing diversity. These experiments do **not** establish robustness to arbitrary unseen corruptions, compound real-world degradations, or unconstrained distribution shift.

## Data and code availability

SHWD is publicly available from its original repository. This repository does not redistribute SHWD images, trained checkpoints, or large temporary corruption images. It provides the fixed split manifests, final V3.1 evaluation protocol and utilities, protocol notes, and machine-readable manuscript result locks.

Two A1-R10 checkpoint SHA256 fields (training seeds 42 and 3407) are intentionally blank in `results/V3.1_final_4x3/checkpoint_identities.csv` because those hashes were not present in the synchronized provenance artifacts. They are not reconstructed or guessed. Exact scikit-image, Pillow, and torchvision versions were also not captured in the original V3.1 run artifact, so a newly built environment can be audited for determinism but cannot be claimed to reproduce every original corruption image byte-for-byte from the public files alone.

Complete frozen assignment manifests for A1-WH and all ten A1-R10 views are not included. Their intervention rules are documented in the manuscript and `assignments/README.md`; the repository therefore exposes the evaluation protocol and result lock, not a complete training-side reconstruction of these controls.

Because the training-time `best.pt` checkpoints are selected using the validation split, the 607-image Helmet-C-Val benchmark is **not an untouched blind test set**. The final paper therefore reports it explicitly as the fixed validation benchmark and does not present V3.1 robustness numbers as independent-test results.

## Citation

Machine-readable citation metadata is provided in `CITATION.cff`.

Until publication metadata is available, cite the manuscript as:

```text
Chunjiang Zhao, Tailong Xu, and Jizhou Wang.
"Robustness evaluation and mechanism analysis of small-object
safety-helmet detection under synthetic common corruptions."
2026.
```

## License

Original code and repository materials are released under the MIT License. SHWD and third-party packages retain their own licenses.
