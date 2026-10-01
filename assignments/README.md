# Frozen Training Assignments

This directory contains frozen deterministic training assignments used in
the formal corruption-aware training protocol.

## A1 Corr-Aug Assignment

`A1_CORRUPTION_ASSIGNMENT_FROZEN.csv` contains exactly one deterministic
corruption assignment for each of the 5,457 SHWD training images.

The frozen assignment uses:

- 14 corruption types
- severity levels 1–3
- no `elastic_transform`
- one assignment per training image
- one deterministic corruption seed per assignment

The CSV contains the following fields:

`image_id`, `source_image`, `source_suffix`, `corruption`, `severity`,
`corruption_seed`, and `corrupted_filename`.

Validation confirmed:

- 5,457 rows
- 5,457 unique image IDs
- no duplicate assignments
- no missing values
- complete coverage of the frozen SHWD training split
- no extra image IDs
- no `elastic_transform` assignments

The `source_image` field preserves machine-local paths from the original
experimental environment as provenance. These paths are not required for
reproduction; `image_id` identifies the corresponding image in the fixed
training split.

The frozen CSV is retained without modification.


## A1-WH and A1-R10 public boundary

The repository does not claim to contain complete frozen assignment manifests for A1-WH or all ten A1-R10 views. The manuscript-facing construction rules are therefore the authoritative public description:

- **A1-WH:** start from the frozen A1 assignment; remove `snow`, `frost`, and `fog`; deterministically remap the affected rows to the remaining 11 non-weather families while preserving severity and the overall training budget.
- **A1-R10:** view 0 reproduces the frozen A1 assignment; views 1-9 deterministically permute the same complete joint `(corruption, severity)` multiset across the 5,457 training image IDs; the frozen 75-epoch schedule uses each view seven or eight times.

These controls are described so their intervention is auditable, but this repository should not be interpreted as a complete training-side reconstruction package for those two controls.
