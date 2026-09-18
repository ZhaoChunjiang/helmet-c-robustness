#!/usr/bin/env python3
"""
Reproduce the three-seed A0 vs A1 (Corr-Aug) summary reported in the paper.

The script reads the six frozen Helmet-C-Val summary JSON files already
included in this repository and computes:

1. Per-seed metrics.
2. Mean and sample standard deviation across seeds 0, 42, and 3407.
3. Paired A1 - A0 differences.
4. CSV files suitable for independent verification.

Run from the repository root:

    python scripts/summarize_three_seed.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]

RESULT_FILES = {
    (0, "A0"): ROOT
    / "results/three_seed/seed_0/A0_eval/A0C_summary.json",

    (0, "A1"): ROOT
    / "results/three_seed/seed_0/A1_eval/A1_EVAL_SUMMARY.json",

    (42, "A0"): ROOT
    / "results/three_seed/seed_42/A0_eval/SUMMARY.json",

    (42, "A1"): ROOT
    / "results/three_seed/seed_42/A1_eval/SUMMARY.json",

    (3407, "A0"): ROOT
    / "results/three_seed/seed_3407/A0_eval/SUMMARY.json",

    (3407, "A1"): ROOT
    / "results/three_seed/seed_3407/A1_eval/SUMMARY.json",
}

SEEDS = [0, 42, 3407]

METRICS = {
    "clean_AP50_95": ("ALL/all_macro", "clean_AP50_95"),
    "mPC15": ("ALL/all_macro", "mPC15_AP50_95"),
    "rPC15": ("ALL/all_macro", "rPC15"),
    "ES_mPC": ("ES/all_macro", "mPC15_AP50_95"),
    "S_mPC": ("S/all_macro", "mPC15_AP50_95"),
    "M_mPC": ("M/all_macro", "mPC15_AP50_95"),
    "L_mPC": ("L/all_macro", "mPC15_AP50_95"),
    "ES_person_mPC": ("ES/person", "mPC15_AP50_95"),
}


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    views = data["global"]["views"]

    values = {}

    for metric, (view_name, field_name) in METRICS.items():
        values[metric] = float(
            views[view_name][field_name]
        )

    return values


def sample_stats(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values)


def main() -> None:
    rows = []

    for seed in SEEDS:
        for model in ["A0", "A1"]:
            metrics = load_metrics(
                RESULT_FILES[(seed, model)]
            )

            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    **metrics,
                }
            )

    output_dir = ROOT / "results/three_seed"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "three_seed_raw.csv"

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "model",
                *METRICS.keys(),
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []

    for metric in METRICS:
        a0 = [
            row[metric]
            for row in rows
            if row["model"] == "A0"
        ]

        a1 = [
            row[metric]
            for row in rows
            if row["model"] == "A1"
        ]

        a0_mean, a0_std = sample_stats(a0)
        a1_mean, a1_std = sample_stats(a1)

        paired_delta = [
            a1[i] - a0[i]
            for i in range(len(SEEDS))
        ]

        delta_mean, delta_std = sample_stats(
            paired_delta
        )

        summary_rows.append(
            {
                "metric": metric,
                "A0_mean": a0_mean,
                "A0_std": a0_std,
                "A1_mean": a1_mean,
                "A1_std": a1_std,
                "paired_delta_mean": delta_mean,
                "paired_delta_std": delta_std,
            }
        )

    summary_path = (
        output_dir / "three_seed_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 78)
    print("Three-seed A0 vs Corr-Aug reproduction")
    print("=" * 78)

    for row in summary_rows:
        metric = row["metric"]

        if metric == "rPC15":
            scale = 100.0
            suffix = "%"
        else:
            scale = 1.0
            suffix = ""

        print(
            f"{metric:<18} "
            f"A0={row['A0_mean'] * scale:.4f}"
            f" ± {row['A0_std'] * scale:.4f}{suffix}   "
            f"A1={row['A1_mean'] * scale:.4f}"
            f" ± {row['A1_std'] * scale:.4f}{suffix}"
        )

    print()
    print("Paired A1 - A0 changes:")

    for row in summary_rows:
        metric = row["metric"]

        # Express changes in percentage points.
        delta_mean_pp = (
            row["paired_delta_mean"] * 100.0
        )
        delta_std_pp = (
            row["paired_delta_std"] * 100.0
        )

        print(
            f"{metric:<18} "
            f"{delta_mean_pp:+.2f} "
            f"± {delta_std_pp:.2f} pp"
        )

    print()
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
