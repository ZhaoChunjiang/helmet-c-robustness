#!/usr/bin/env python3
"""Verify the machine-readable final V3.1 manuscript result lock."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "V3.1_final_4x3"

EXPECTED = {
    "A0": [0.609093, 0.000260, 0.385045, 0.001119, 63.22, 0.16, 0.452176, 0.006166, 74.24, 0.98, 0.352347, 0.000517],
    "A1": [0.609589, 0.000865, 0.520876, 0.000561, 85.45, 0.07, 0.558918, 0.001035, 91.69, 0.07, 0.508199, 0.000465],
    "A1-WH": [0.610760, 0.001022, 0.518365, 0.000720, 84.87, 0.26, 0.532987, 0.001188, 87.27, 0.34, 0.511669, 0.000761],
    "A1-R10": [0.607964, 0.001196, 0.531700, 0.000762, 87.46, 0.09, 0.566016, 0.000378, 93.10, 0.22, 0.520941, 0.001126],
}
FIELDS = [
    "clean_AP50_95_mean", "clean_AP50_95_sample_sd", "mPC15_mean", "mPC15_sample_sd",
    "rPC15_mean_percent", "rPC15_sample_sd_percent", "weather_mPC_mean", "weather_mPC_sample_sd",
    "weather_rPC_mean_percent", "weather_rPC_sample_sd_percent", "seen11_mPC_mean", "seen11_mPC_sample_sd",
]
EXPECTED_DELTA = {
    "clean_AP50_95": (-0.162, 0.157),
    "mPC15": (1.082, 0.102),
    "weather_mPC": (0.710, 0.129),
    "seen11_mPC": (1.274, 0.134),
}


def close(a: float, b: float, tol: float = 5e-7):
    return abs(a - b) <= tol


def main():
    stats = {}
    with (RESULTS / "model_stats.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats[row["model"]] = row
    if set(stats) != set(EXPECTED):
        raise SystemExit(f"Unexpected models: {sorted(stats)}")
    for model, expected in EXPECTED.items():
        for field, target in zip(FIELDS, expected):
            got = float(stats[model][field])
            if not close(got, target):
                raise SystemExit(f"{model} {field}: {got} != {target}")

    deltas = {}
    with (RESULTS / "paired_deltas.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            deltas[row["metric"]] = (float(row["mean_delta_pp"]), float(row["sample_sd_pp"]))
    if set(deltas) != set(EXPECTED_DELTA):
        raise SystemExit(f"Unexpected paired-delta metrics: {sorted(deltas)}")
    for metric, target in EXPECTED_DELTA.items():
        got = deltas[metric]
        if not all(close(a, b, 5e-7) for a, b in zip(got, target)):
            raise SystemExit(f"{metric}: {got} != {target}")

    protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "helmet-c-val-v3.1-deterministic-impulse-audited-glass-2026-09-29":
        raise SystemExit("Protocol ID mismatch")
    if protocol.get("images") != 607 or protocol.get("conditions") != 75:
        raise SystemExit("Protocol image/condition count mismatch")

    print("V3.1 FINAL TABLE VERIFICATION: PASS")


if __name__ == "__main__":
    main()
