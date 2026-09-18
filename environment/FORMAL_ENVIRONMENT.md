# Formal Experiment Environment

This document records the environment used for the formal Helmet-C experiments.

## Hardware and System

- GPU: NVIDIA RTX 4090
- Operating system: Ubuntu 22.04
- Python: 3.10
- Input resolution: 640 × 640

## Core Software

- PyTorch: 2.1.2+cu118
- Ultralytics: 8.4.140
- NumPy: 1.26.4
- imagecorruptions: 1.1.2

## Formal Helmet-C Runner

The final reproducible evaluation program is:

```text
scripts/run_v10_A0C_HELMET_C_BASELINE_FORMAL_v104_ATOMIC_REPRO.py
