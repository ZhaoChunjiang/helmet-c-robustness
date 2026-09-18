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
