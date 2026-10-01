#!/usr/bin/env python3
"""Aggregate fresh V3.1 per-seed summary.json files.

This utility recomputes mean and sample standard deviation from fresh evaluator
outputs. It does not read the committed manuscript lock and does not require
hard-coded manuscript numbers.

Example:
  python scripts/v31/aggregate_v31_three_seed.py \
    --record A0 0 A0_seed0 /runs/a0_s0/summary.json \
    --record A0 42 A0_seed42 /runs/a0_s42/summary.json \
    --record A0 3407 A0_seed3407 /runs/a0_s3407/summary.json \
    --output /runs/A0_aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "clean_AP50_95",
    "mPC15",
    "rPC15",
    "weather_mPC",
    "weather_rPC",
    "nonweather11_mPC",
)


def load_record(model: str, seed: int, label: str, path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("status") != "PASS":
        raise RuntimeError(f"Evaluator summary is not PASS: {path}")
    models = obj.get("models", {})
    if label not in models:
        raise RuntimeError(f"Model label {label!r} not found in {path}")
    row = dict(models[label])
    if "nonweather11_mPC" not in row and "seen11_mPC" in row:
        row["nonweather11_mPC"] = row["seen11_mPC"]
    missing = [m for m in METRICS if m not in row]
    if missing:
        raise RuntimeError(f"Missing metrics {missing} in {path}")
    return {
        "model": model,
        "seed": int(seed),
        "label": label,
        "source": str(path),
        **{m: float(row[m]) for m in METRICS},
    }


def mean_sd(values):
    values = list(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--record",
        nargs=4,
        action="append",
        metavar=("MODEL", "SEED", "LABEL", "SUMMARY_JSON"),
        required=True,
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    records = [
        load_record(model, int(seed), label, Path(path))
        for model, seed, label, path in args.record
    ]
    grouped = defaultdict(list)
    for row in records:
        grouped[row["model"]].append(row)

    out = {"models": {}, "records": records}
    csv_rows = []
    for model, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda x: x["seed"])
        metrics = {}
        for metric in METRICS:
            mean, sd = mean_sd(r[metric] for r in rows)
            metrics[metric] = {"mean": mean, "sample_sd": sd}
            csv_rows.append(
                {
                    "model": model,
                    "n_seeds": len(rows),
                    "metric": metric,
                    "mean": mean,
                    "sample_sd": sd,
                    "display_scale": "ratio",
                }
            )
            if metric in {"rPC15", "weather_rPC"}:
                csv_rows.append(
                    {
                        "model": model,
                        "n_seeds": len(rows),
                        "metric": metric + "_percent",
                        "mean": 100.0 * mean,
                        "sample_sd": 100.0 * sd,
                        "display_scale": "percent",
                    }
                )
        out["models"][model] = {
            "seeds": [r["seed"] for r in rows],
            "metrics": metrics,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "n_seeds", "metric", "mean", "sample_sd", "display_scale"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print("V3.1 SEED AGGREGATION: PASS")


if __name__ == "__main__":
    main()
