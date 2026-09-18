# Helmet-C Robustness

[![Reproducibility Check](https://github.com/ZhaoChunjiang/helmet-c-robustness/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/ZhaoChunjiang/helmet-c-robustness/actions/workflows/reproducibility.yml)

Reproducible evaluation code for studying corruption robustness in safety-helmet object detection.

This repository contains the formal evaluation protocol, reproducibility utilities, and supporting scripts for the Helmet-C benchmark used in our experiments.

## Overview

Helmet-C evaluates object detectors under common image corruptions.

The formal benchmark contains:

- 15 corruption types
- 5 severity levels
- 75 corruption/severity conditions
- 1517 images in the independent SHWD test split

The primary robustness metrics are:

- AP50:95
- mPC15: mean AP50:95 over all 75 corruption conditions
- rPC15: mPC15 / clean AP50:95

Scale-stratified evaluation is also performed for:

- ES: area < 16²
- S: 16² ≤ area < 32²
- M: 32² ≤ area < 96²
- L: area ≥ 96²

All object areas are defined after resizing to the formal 640 × 640 evaluation resolution.

## Repository Structure

```text
helmet-c-robustness/
├── .github/
│   └── workflows/
│       └── reproducibility.yml
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── assignments/
│   ├── README.md
│   └── A1_CORRUPTION_ASSIGNMENT_FROZEN.csv
├── configs/
│   └── helmet_c.yaml
├── environment/
│   └── FORMAL_ENVIRONMENT.md
├── scripts/
│   ├── export_split_manifest.py
│   ├── generate_helmet_c.py
│   ├── run_v10_A0C_HELMET_C_BASELINE_FORMAL_v104_ATOMIC_REPRO.py
│   ├── summarize_robustness.py
│   ├── summarize_three_seed.py
│   ├── validate_helmet_c.py
│   ├── validate_split_manifest.py
│   └── verify_table2.py
├── seeds/
│   └── seeds.txt
├── splits/
│   ├── split_summary.txt
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── results/
    ├── formal/
    │   └── seed_0/
    ├── three_seed/
    │   ├── seed_0/
    │   │   ├── A0_eval/
    │   │   └── A1_eval/
    │   ├── seed_42/
    │   │   ├── A0_eval/
    │   │   └── A1_eval/
    │   └── seed_3407/
    │       ├── A0_eval/
    │       └── A1_eval/
    └── final_validation/
        ├── README.md
        └── final_report/
```

## Formal Evaluation Runner

The canonical formal evaluation program is:

```text
scripts/run_v10_A0C_HELMET_C_BASELINE_FORMAL_v104_ATOMIC_REPRO.py
```

Script version:

```text
1.0.4-A0C-PARALLEL-ATOMIC-REPRO
```

Earlier experimental versions are not the canonical reproduction entry point.

## Formal Environment

The formal experiments were performed with:

- Ubuntu 22.04
- Python 3.10
- NVIDIA RTX 4090
- PyTorch 2.1.2+cu118
- Ultralytics 8.4.140
- NumPy 1.26.4
- imagecorruptions 1.1.2
- input resolution: 640 × 640

See:

```text
environment/FORMAL_ENVIRONMENT.md
```

for the complete environment record.

## Installation

Create a clean Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```
`requirements.txt` is a convenience installation manifest rather than a
complete historical lockfile. The exact frozen core environment used for
the formal experiments is documented in `environment/FORMAL_ENVIRONMENT.md`.

For strict reproduction of the formal CUDA environment, install the PyTorch 2.1.2 CUDA 11.8 build appropriate for your platform before running the formal experiment.

## Data Preparation

The original SHWD images are not redistributed in this repository.

Prepare the dataset in YOLO format and provide a dataset YAML file containing the fixed train, validation, and test split.

The formal split contains:

| Split | Images | hat | person | Total objects |
|---|---:|---:|---:|---:|
| Train | 5457 | 6419 | 79778 | 86197 |
| Val | 607 | 747 | 9178 | 9925 |
| Test | 1517 | 1878 | 22558 | 24436 |
| Total | 7581 | 9044 | 111514 | 120558 |

The test split must remain closed during development and should only be used for the final independent evaluation.

## Fixed Split Manifests

To export manifests from an already-existing fixed split:

```bash
python scripts/export_split_manifest.py \
    --train-dir /path/to/images/train \
    --val-dir /path/to/images/val \
    --test-dir /path/to/images/test \
    --output-dir splits
```

This script does not create a new random split. It records the split already used in the experiment.

Validate the exported manifests with:

```bash
python scripts/validate_split_manifest.py \
    --split-dir splits \
    --train-root /path/to/images/train \
    --val-root /path/to/images/val \
    --test-root /path/to/images/test
```

Expected image counts are:

```text
train = 5457
val   = 607
test  = 1517
```

## Formal Checkpoint

The formal A0 YOLO11n checkpoint is identified by the following SHA256 value:

```text
9173932805b7589a337ed7838e1abf52ec949e893f11ac1a9d1bc22122b8b5d2
```

The formal runner verifies the checkpoint hash before evaluation.

The checkpoint itself is not included in this repository.

## Self-Test

Before running the formal evaluation, test the custom evaluator with:

```bash
python scripts/run_v10_A0C_HELMET_C_BASELINE_FORMAL_v104_ATOMIC_REPRO.py \
    --self-test
```

The self-test should finish successfully before the formal benchmark is executed.

## Formal Helmet-C Evaluation

Run the final Helmet-C evaluation using explicit paths:

```bash
python scripts/run_v10_A0C_HELMET_C_BASELINE_FORMAL_v104_ATOMIC_REPRO.py \
    --data /path/to/data.yaml \
    --best /path/to/best.pt \
    --a0s-metrics /path/to/A0S_metrics_by_size_class.csv \
    --a0s-protocol /path/to/A0S_PROTOCOL_LOCK.json \
    --results-root /path/to/results/A0C_helmet_c/seed_0 \
    --zip-path /path/to/results/A0C_HELMET_C_RESULTS_TO_UPLOAD.zip
```

The frozen formal inference settings are:

```text
imgsz              = 640
batch              = 8
corruption workers = 12
prefetch images    = 96
corruption seed    = 3407
confidence         = 0.001
NMS IoU            = 0.7
max detections     = 300
```

## Deterministic Corruptions

The final formal runner generates corruptions deterministically.

The random seed for each corrupted image is derived from:

```text
base seed + image identity + corruption type + severity
```

using a SHA256-based deterministic seed scheme.

The frozen base corruption seed is:

```text
3407
```

This prevents corruption results from depending on filesystem traversal order or multiprocessing scheduling.

## Corruption Types

Helmet-C uses the following 15 corruption types:

### Noise

- gaussian_noise
- shot_noise
- impulse_noise

### Blur

- defocus_blur
- glass_blur
- motion_blur
- zoom_blur

### Weather / Imaging

- snow
- frost
- fog
- brightness

### Digital

- contrast
- elastic_transform
- pixelate
- jpeg_compression

Each corruption is evaluated at severity levels 1–5.

## Metrics

For corruption type \(c\) and severity \(s\), let the detection performance be:

```text
AP(c, s)
```

Then:

```text
mPC15 = mean AP50:95 over all 15 × 5 = 75 conditions
```

and:

```text
rPC15 = mPC15 / clean AP50:95
```

The formal runner additionally reports:

- AP50
- AP50:95
- maximum recall at IoU = 0.50
- recall at confidence = 0.25
- per-corruption summaries
- severity profiles
- ES / S / M / L performance
- class-specific hat / person results
- mPC14 excluding elastic_transform as a geometry-label sensitivity analysis

## Optional Materialized Helmet-C Generation

The script:

```text
scripts/generate_helmet_c.py
```

can be used to explicitly generate and save corrupted images.

This is provided as a convenience utility.

The canonical formal evaluation does **not** require saving all corrupted images to disk. The final formal runner generates corrupted images deterministically during evaluation and caches prediction results instead.

## Benchmark Integrity Check

If Helmet-C has been materialized to disk, validate it using:

```bash
python scripts/validate_helmet_c.py \
    --helmet-c-root /path/to/Helmet-C \
    --expected-images 1517
```

A complete materialized benchmark contains:

```text
15 corruptions × 5 severities = 75 conditions
1517 images per condition
113775 corrupted images in total
```

## Result Summarization

For externally generated condition-level AP results, robustness summaries can be calculated with:

```bash
python scripts/summarize_robustness.py \
    --input-csv results/helmet_c_ap.csv \
    --clean-ap CLEAN_AP_VALUE \
    --output-dir results/summary
```

The input CSV must contain exactly one row for every corruption/severity pair.

## Reproducibility Principles

This repository follows the following rules:

1. Fixed dataset splits are used for all experiments.
2. The independent test set is not used for method development.
3. The formal model checkpoint is identified using SHA256.
4. Corruption generation is deterministic.
5. Evaluation parameters are frozen.
6. The formal runtime environment is documented.
7. Partial or debugging runs are not reported as formal 75-condition results.

## Data and Model Availability

The original SHWD images are not redistributed in this repository.

Model checkpoints and large prediction caches are also excluded from the
GitHub repository. Their identities and experimental roles are recorded
through protocol files, SHA256 hashes, and reproducibility metadata.

The repository contains evaluation outputs, protocol records, summary
statistics, and scripts required to verify the reported robustness results.

Some frozen protocol files preserve machine-local paths from the original
experimental environment, for example:

```text
/root/autodl-tmp/...
```

These paths are retained only as provenance records of the original run.
They are not required directory locations for reproduction. Users should
provide their own paths through the command-line arguments documented in
this README.

## Paper

This repository accompanies the manuscript:

**Robustness evaluation and mechanism analysis of small-object safety-helmet detection under common imaging corruptions**

The repository contains the frozen Helmet-C evaluation protocol,
seed-level experimental outputs, reproducibility utilities, and formal
independent-test artifacts associated with the manuscript.

Publication metadata, journal information, and DOI will be added after
formal publication.

## Three-Seed Reproducibility

The primary YOLO11n three-seed comparison uses the training seeds:

```text
0
42
3407
```

The six frozen A0/A1 Helmet-C-Val result sets are stored under:

```text
results/three_seed/
```

where:

```text
A0 = clean-training baseline
A1 = corruption-aware training (Corr-Aug)
```

The script:

```text
scripts/summarize_three_seed.py
```

recomputes the seed-level values, three-seed means, sample standard
deviations, and paired A1 - A0 differences directly from the frozen
JSON result files.

The script:

```text
scripts/verify_table2.py
```

verifies that the six frozen result files reproduce the values reported
in Table 2 of the manuscript.

To run the verification manually from the repository root:

```bash
python scripts/summarize_three_seed.py
python scripts/verify_table2.py
```

A successful verification ends with:

```text
TABLE 2 VERIFICATION PASSED
```

## Automated Reproducibility Check

GitHub Actions automatically performs the three-seed reproducibility
verification whenever the relevant scripts, result files, or workflow
configuration are changed.

The workflow is defined in:

```text
.github/workflows/reproducibility.yml
```

A successful workflow run executes:

```text
Recompute three-seed summary
Verify manuscript Table 2
Upload reproduced summary
```

and generates the artifact:

```text
three-seed-reproduced-summary
```

containing:

```text
three_seed_raw.csv
three_seed_summary.csv
```

The current workflow status is shown by the badge at the top of this README.

## Formal Independent-Test Results

The frozen seed-0 A0 independent Helmet-C-Test outputs generated with the
final v1.0.4 formal runner are available under:

```text
results/formal/seed_0/
```

These files include the protocol lock, environment record, condition-level
metrics, robustness summaries, and formal figures.

The independent SHWD Test and Helmet-C-Test were evaluated only after the
method and evaluation protocol had been frozen.

## Validation Summary

Manuscript-level aggregate validation results are summarized in:

```text
results/final_validation/README.md
```

This summary includes the three-seed YOLO11n results and additional
validation evidence such as the YOLOv8n replication and independent-test
analysis.

The summary document does not replace the underlying seed-level
experimental outputs.

## Citation

If you use the code, Helmet-C protocol, evaluation procedure, or
experimental results in this repository, please cite the associated work.

Machine-readable citation metadata is provided in:

```text
CITATION.cff
```

Until formal publication metadata is available, the manuscript may be
cited as:

```text
Chunjiang Zhao, Tailong Xu, and Jizhou Wang.
"Robustness evaluation and mechanism analysis of small-object
safety-helmet detection under common imaging corruptions."
2026.
```

The journal name, volume, issue, pages, and DOI will be added after
formal publication.

## License

This repository is released under the MIT License.

See:

```text
LICENSE
```

for the complete license text.

The MIT License applies to the original code and materials released in
this repository. The original SHWD dataset and third-party software retain
their respective licenses and are not relicensed by this repository.

## Reproducibility Notes

For strict reproduction, users should pay particular attention to the
following frozen settings:

```text
Python             = 3.10
PyTorch            = 2.1.2+cu118
Ultralytics        = 8.4.140
NumPy              = 1.26.4
imagecorruptions   = 1.1.2
input resolution   = 640 × 640
corruption seed    = 3407
confidence         = 0.001
NMS IoU            = 0.7
max detections     = 300
```

The formal A0 checkpoint is identified by:

```text
SHA256:
9173932805b7589a337ed7838e1abf52ec949e893f11ac1a9d1bc22122b8b5d2
```

The original checkpoint file is not redistributed in this repository.

All reported formal Helmet-C results should be interpreted together with
the frozen protocol files, environment records, and deterministic
corruption-generation procedure included in this repository.
