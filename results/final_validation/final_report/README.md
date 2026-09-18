# Final Validation Report

This directory contains frozen manuscript-level validation summaries and
machine-readable exports generated during the final validation stage.

## Important Note on File Names

The `TABLE_1` to `TABLE_5` prefixes in this directory are internal export
indices from the frozen final-validation pipeline. They do not correspond
directly to the final manuscript table numbering.

In the manuscript:

- Table 1 reports the fixed SHWD train/validation/test split.
- Table 2 reports the three-seed YOLO11n A0 versus Corr-Aug robustness results.
- Table 3 reports mechanism stability and additional final validation.

The original exported filenames are preserved unchanged for provenance.

## Files

`FINAL_VALIDATION_REPORT.txt`
provides the frozen human-readable completion report.

`FINAL_VALIDATION_SUMMARY.json`
contains the machine-readable aggregate final-validation summary.

`TABLE_1_YOLO11N_MULTI_SEED.csv`
contains the seed-level YOLO11n A0/A1 results underlying manuscript Table 2.

`TABLE_2_A3_MULTI_SEED.csv`
contains the three-seed A3 mechanism-probe results contributing to
manuscript Table 3.

`TABLE_3_YOLOV8N_REPLICATION.csv`
contains the cross-architecture YOLOv8n replication results contributing
to manuscript Table 3.

`TABLE_4_FINAL_TEST.csv`
contains the independently opened SHWD Test results contributing to
manuscript Table 3.

`TABLE_5_SHEL5K_EXTERNAL.csv`
contains the SHEL5K external validation results retained as additional
supporting evidence.

## Provenance

These files are frozen outputs from the final validation stage and are
retained without modification.
