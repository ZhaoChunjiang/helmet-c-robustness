# Final Validation Summary

This directory summarizes the frozen validation evidence reported in the paper.

The three-seed YOLO11n experiments use seeds:

- 0
- 42
- 3407

A0 denotes the standard baseline and Corr-Aug denotes corruption-augmented
training under the same image-presentation budget.

## Three-Seed Helmet-C-Val Results

Values are mean ± standard deviation across seeds 0, 42, and 3407.

| Metric | A0 | Corr-Aug | Change |
|---|---:|---:|---:|
| Clean AP50:95 | 0.6091 ± 0.0003 | 0.6096 ± 0.0009 | +0.05 ± 0.06 pp |
| mPC15 | 0.2532 ± 0.0020 | 0.3859 ± 0.0028 | +13.27 ± 0.47 pp |
| rPC15 | 41.58 ± 0.31% | 63.31 ± 0.54% | +21.73 ± 0.84 pp |
| ES mPC | 0.0554 ± 0.0045 | 0.0958 ± 0.0058 | +4.04 pp |
| S mPC | 0.1650 ± 0.0076 | 0.2667 ± 0.0059 | +10.17 pp |
| M mPC | 0.3726 ± 0.0017 | 0.5085 ± 0.0029 | +13.59 pp |
| L mPC | 0.4958 ± 0.0023 | 0.6730 ± 0.0066 | +17.72 pp |
| ES-person mPC | 0.0765 ± 0.0031 | 0.1400 ± 0.0018 | +6.35 ± 0.40 pp |

## Additional Validation

| Validation | Main result |
|---|---|
| Object-consistency probe, 3 seeds | Consistency distance −32.77 ± 0.44%; Blur6-ALL +0.077 ± 0.147 pp |
| YOLOv8n replication | Clean +0.02 pp; mPC15 +11.90 pp; rPC15 +19.60 pp |
| Independent SHWD Test | Clean −0.07 pp; mPC15 24.21% → 37.68%; rPC15 40.55% → 63.18% |
| Test sensitivity | mPC14 23.29% → 37.54%; ES-person mPC15 +7.34 pp |

Machine-readable final validation summaries and exported paper tables are
available under `results/final_validation/final_report/`.

## Protocol Note

Development used Train/Val only.

The independent SHWD Test and Helmet-C-Test were evaluated after the
method and evaluation protocol had been frozen.

The formal seed-0 A0 Helmet-C-Test artifacts are available under
`results/formal/seed_0/`.

The underlying three-seed A0/A1 Helmet-C-Val outputs are available under
`results/three_seed/`.

This summary reports manuscript-level aggregate values and does not
replace the underlying seed-level experimental outputs.
