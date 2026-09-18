# Formal Experiment Environment

This document records the software, hardware, and frozen evaluation
settings used for the formal Helmet-C experiments.

## Hardware and System

- GPU: NVIDIA GeForce RTX 4090
- GPU memory: 24564 MiB
- NVIDIA driver: 570.124.04
- Operating system: Ubuntu 22.04
- Python: 3.10
- Formal recorded Python runtime: 3.10.8
- Input resolution: 640 × 640

## Core Software

- PyTorch: 2.1.2+cu118
- CUDA runtime used by PyTorch: 11.8
- Ultralytics: 8.4.140
- NumPy: 1.26.4
- imagecorruptions: 1.1.2

The raw A0 environment record reports the `imagecorruptions` version as
`unknown` because the installed package did not expose a version attribute
through the runtime query used by the experiment script. The formal
dependency configuration and A1 protocol record identify the package
version as 1.1.2.

## Formal Helmet-C Runner

The final reproducible A0 independent-test evaluation program is:

```text
scripts/run_v10_A0C_HELMET_C_BASELINE_FORMAL_v104_ATOMIC_REPRO.py
```

Formal script version:

```text
1.0.4-A0C-PARALLEL-ATOMIC-REPRO
```

Earlier development versions are not the canonical formal reproduction
entry point.

## Formal A0 Checkpoint

The formal YOLO11n A0 checkpoint is identified by:

```text
SHA256:
9173932805b7589a337ed7838e1abf52ec949e893f11ac1a9d1bc22122b8b5d2
```

The checkpoint file itself is not redistributed in this repository.

The formal runner verifies this SHA256 value before executing the
independent-test evaluation.

## Formal Dataset Record

The formal SHWD split contains:

| Split | Images | hat objects | person objects | Total objects |
|---|---:|---:|---:|---:|
| Train | 5457 | 6419 | 79778 | 86197 |
| Val | 607 | 747 | 9178 | 9925 |
| Test | 1517 | 1878 | 22558 | 24436 |

The independent Test split remained closed during method development.

The formal dataset configuration used in the experiment is identified by:

```text
SHA256:
08ec5fa03dbc81775183fea4fe9dde94e5f8bc42feded9598c8e65831466c46f
```

## Frozen Inference Settings

The final formal Helmet-C evaluation uses:

```text
imgsz              = 640
requested batch    = 8
corruption workers = 12
prefetch images    = 96
corruption seed    = 3407
confidence         = 0.001
NMS IoU            = 0.7
max detections     = 300
device             = 0
```

CUDA out-of-memory batch backoff is enabled in the formal runner.

## Helmet-C Configuration

Helmet-C contains:

```text
15 corruption types
5 severity levels
75 corruption/severity conditions
```

The corruption types are:

```text
gaussian_noise
shot_noise
impulse_noise
defocus_blur
glass_blur
motion_blur
zoom_blur
snow
frost
fog
brightness
contrast
elastic_transform
pixelate
jpeg_compression
```

Severity levels are:

```text
1, 2, 3, 4, 5
```

## Deterministic Corruption Generation

The frozen base corruption seed is:

```text
3407
```

Each corrupted image receives a deterministic seed derived using:

```text
sha256(base_seed | image_id | corruption | severity) -> uint32
```

This makes corruption generation independent of filesystem traversal
order and multiprocessing scheduling.

The formal CPU corruption backend uses:

```text
ProcessPoolExecutor(spawn)
```

## Object-Size Groups

Object areas are calculated after resizing to the formal 640 × 640
evaluation resolution.

The frozen scale definitions are:

```text
ES = [0, 16^2)
S  = [16^2, 32^2)
M  = [32^2, 96^2)
L  = [96^2, +inf)
```

## Primary Robustness Metrics

The formal benchmark reports:

```text
mPC15 = mean AP50:95 over 15 corruptions × 5 severity levels
rPC15 = mPC15 / clean AP50:95
```

It additionally reports:

```text
mPC14_no_elastic
```

which excludes `elastic_transform` as a geometry-label sensitivity
analysis.

## Three-Seed Training Seeds

The primary YOLO11n robustness comparison uses the training seeds:

```text
0
42
3407
```

These training seeds should not be confused with the Helmet-C corruption
base seed. The corruption base seed remains fixed at 3407 for benchmark
generation.

## Environment Records

The frozen machine-readable environment record for the formal seed-0 A0
independent-test experiment is available at:

```text
results/formal/seed_0/A0C_environment.json
```

The corresponding frozen protocol record is:

```text
results/formal/seed_0/A0C_PROTOCOL_LOCK.json
```

## Machine-Local Paths

Some frozen protocol files preserve paths from the original experimental
machine, for example:

```text
/root/autodl-tmp/...
```

These paths are retained as provenance information and do not prescribe
the directory layout required by other users.

For reproduction, provide the appropriate local paths through the formal
runner command-line arguments.

## Reproduction Principle

Formal results should be interpreted together with:

```text
the frozen model identity
the frozen dataset identity
the deterministic corruption procedure
the fixed inference parameters
the documented software environment
the stored protocol lock files
```

Changing any of these elements may produce results that are not directly
comparable with the reported formal Helmet-C experiments.
