#!/usr/bin/env python3
"""
Summarize Helmet-C robustness evaluation results.

This script does NOT recompute object-detection AP from raw predictions.
AP50:95 values should be produced by the same official evaluator used
for the clean-set experiments.

Required input CSV columns
--------------------------
corruption,severity,AP50_95

Example
-------
corruption,severity,AP50_95
gaussian_noise,1,0.4123
gaussian_noise,2,0.3871
...
jpeg_compression,5,0.3012

Usage
-----
python scripts/summarize_robustness.py \
    --input-csv results/helmet_c_ap.csv \
    --clean-ap 0.4521 \
    --output-dir results/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

SEVERITIES = [1, 2, 3, 4, 5]

METRIC_COLUMN = "AP50_95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Helmet-C robustness metrics."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV containing AP50:95 for all 75 Helmet-C conditions.",
    )

    parser.add_argument(
        "--clean-ap",
        type=float,
        required=True,
        help="AP50:95 on the clean test set.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/summary"),
        help="Directory used to save summary files.",
    )

    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Result CSV not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "corruption",
        "severity",
        METRIC_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required CSV columns: "
            + ", ".join(sorted(missing))
        )

    df = df[
        ["corruption", "severity", METRIC_COLUMN]
    ].copy()

    df["corruption"] = (
        df["corruption"]
        .astype(str)
        .str.strip()
    )

    df["severity"] = pd.to_numeric(
        df["severity"],
        errors="raise",
    ).astype(int)

    df[METRIC_COLUMN] = pd.to_numeric(
        df[METRIC_COLUMN],
        errors="raise",
    )

    return df


def validate_results(df: pd.DataFrame) -> None:
    expected_pairs = {
        (corruption, severity)
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    }

    actual_pairs = set(
        zip(
            df["corruption"],
            df["severity"],
        )
    )

    missing_pairs = expected_pairs - actual_pairs
    unexpected_pairs = actual_pairs - expected_pairs

    duplicated = df.duplicated(
        subset=["corruption", "severity"],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = df.loc[
            duplicated,
            ["corruption", "severity"],
        ]

        raise RuntimeError(
            "Duplicate corruption/severity results detected:\n"
            + duplicate_rows.to_string(index=False)
        )

    if missing_pairs:
        preview = sorted(missing_pairs)[:10]

        raise RuntimeError(
            f"Missing {len(missing_pairs)} Helmet-C conditions. "
            f"Examples: {preview}"
        )

    if unexpected_pairs:
        preview = sorted(unexpected_pairs)[:10]

        raise RuntimeError(
            f"Unexpected Helmet-C conditions detected: "
            f"{preview}"
        )

    if len(df) != 75:
        raise RuntimeError(
            f"Expected exactly 75 rows, found {len(df)}."
        )

    if df[METRIC_COLUMN].isna().any():
        raise RuntimeError(
            f"{METRIC_COLUMN} contains missing values."
        )


def compute_corruption_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        df.groupby(
            "corruption",
            as_index=False,
        )[METRIC_COLUMN]
        .mean()
        .rename(
            columns={
                METRIC_COLUMN: "mean_AP50_95"
            }
        )
    )

    order = {
        name: index
        for index, name in enumerate(CORRUPTIONS)
    }

    summary["_order"] = (
        summary["corruption"]
        .map(order)
    )

    summary = (
        summary
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return summary


def compute_severity_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby(
            "severity",
            as_index=False,
        )[METRIC_COLUMN]
        .mean()
        .rename(
            columns={
                METRIC_COLUMN: "mean_AP50_95"
            }
        )
        .sort_values("severity")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    if args.clean_ap <= 0:
        raise ValueError(
            "--clean-ap must be greater than zero."
        )

    df = load_results(args.input_csv)

    validate_results(df)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    corruption_summary = (
        compute_corruption_summary(df)
    )

    severity_summary = (
        compute_severity_summary(df)
    )

    # Mean Performance under Corruption:
    # arithmetic mean over all 15 x 5 = 75 conditions.
    mPC = float(
        df[METRIC_COLUMN].mean()
    )

    # Relative Performance under Corruption.
    # Expressed as a percentage of clean-set AP.
    rPC = float(
        100.0 * mPC / args.clean_ap
    )

    overall_summary = {
        "metric": METRIC_COLUMN,
        "clean_AP50_95": float(args.clean_ap),
        "num_corruptions": len(CORRUPTIONS),
        "num_severities": len(SEVERITIES),
        "num_conditions": len(df),
        "mPC": mPC,
        "rPC_percent": rPC,
    }

    raw_output = (
        args.output_dir
        / "helmet_c_conditions.csv"
    )

    corruption_output = (
        args.output_dir
        / "corruption_summary.csv"
    )

    severity_output = (
        args.output_dir
        / "severity_summary.csv"
    )

    overall_output = (
        args.output_dir
        / "overall_summary.json"
    )

    df.sort_values(
        ["corruption", "severity"]
    ).to_csv(
        raw_output,
        index=False,
    )

    corruption_summary.to_csv(
        corruption_output,
        index=False,
    )

    severity_summary.to_csv(
        severity_output,
        index=False,
    )

    with overall_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("Helmet-C robustness summary")
    print("=" * 72)

    print(
        f"Clean AP50:95 : "
        f"{args.clean_ap:.6f}"
    )

    print(
        f"Helmet-C mPC  : "
        f"{mPC:.6f}"
    )

    print(
        f"Helmet-C rPC  : "
        f"{rPC:.2f}%"
    )

    print(
        f"Conditions    : "
        f"{len(df)} / 75"
    )

    print("-" * 72)
    print("Mean AP50:95 by corruption")
    print("-" * 72)

    for _, row in corruption_summary.iterrows():
        print(
            f"{row['corruption']:<20} "
            f"{row['mean_AP50_95']:.6f}"
        )

    print("-" * 72)
    print("Mean AP50:95 by severity")
    print("-" * 72)

    for _, row in severity_summary.iterrows():
        print(
            f"severity_{int(row['severity'])}: "
            f"{row['mean_AP50_95']:.6f}"
        )

    print("=" * 72)
    print("SUMMARY COMPLETED")
    print("=" * 72)

    print()
    print(f"Saved: {raw_output}")
    print(f"Saved: {corruption_output}")
    print(f"Saved: {severity_output}")
    print(f"Saved: {overall_output}")


if __name__ == "__main__":
    main()
