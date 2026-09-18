#!/usr/bin/env python3
"""
Verify that the six frozen three-seed Helmet-C-Val results reproduce
the values reported in Table 2 of the paper.

Run from the repository root:

    python scripts/verify_table2.py

The script reads the original JSON outputs. It does not use manually
entered per-seed results.
"""

from __future__ import annotations

from statistics import mean, stdev

from summarize_three_seed import (
    RESULT_FILES,
    SEEDS,
    METRICS,
    load_metrics,
)


# Values printed in the manuscript.
# A0/A1 AP and mPC values use four decimal places.
# rPC uses percentages with two decimal places.
# Delta values use percentage points.
EXPECTED = {
    "clean_AP50_95": {
        "A0_mean": "0.6091",
        "A0_std": "0.0003",
        "A1_mean": "0.6096",
        "A1_std": "0.0009",
        "delta_mean_pp": "+0.05",
        "delta_std_pp": "0.06",
    },
    "mPC15": {
        "A0_mean": "0.2532",
        "A0_std": "0.0020",
        "A1_mean": "0.3859",
        "A1_std": "0.0028",
        "delta_mean_pp": "+13.27",
        "delta_std_pp": "0.47",
    },
    "rPC15": {
        "A0_mean_percent": "41.58",
        "A0_std_percent": "0.31",
        "A1_mean_percent": "63.31",
        "A1_std_percent": "0.54",
        "delta_mean_pp": "+21.73",
        "delta_std_pp": "0.84",
    },
    "ES_mPC": {
        "A0_mean": "0.0554",
        "A0_std": "0.0045",
        "A1_mean": "0.0958",
        "A1_std": "0.0058",
        "delta_mean_pp": "+4.04",
    },
    "S_mPC": {
        "A0_mean": "0.1650",
        "A0_std": "0.0076",
        "A1_mean": "0.2667",
        "A1_std": "0.0059",
        "delta_mean_pp": "+10.17",
    },
    "M_mPC": {
        "A0_mean": "0.3726",
        "A0_std": "0.0017",
        "A1_mean": "0.5085",
        "A1_std": "0.0029",
        "delta_mean_pp": "+13.59",
    },
    "L_mPC": {
        "A0_mean": "0.4958",
        "A0_std": "0.0023",
        "A1_mean": "0.6730",
        "A1_std": "0.0066",
        "delta_mean_pp": "+17.72",
    },
    "ES_person_mPC": {
        "A0_mean": "0.0765",
        "A0_std": "0.0031",
        "A1_mean": "0.1400",
        "A1_std": "0.0018",
        "delta_mean_pp": "+6.35",
        "delta_std_pp": "0.40",
    },
}


def calculate() -> dict:
    per_seed = {}

    for seed in SEEDS:
        per_seed[seed] = {}

        for model in ("A0", "A1"):
            per_seed[seed][model] = load_metrics(
                RESULT_FILES[(seed, model)]
            )

    output = {}

    for metric in METRICS:
        a0 = [
            per_seed[seed]["A0"][metric]
            for seed in SEEDS
        ]

        a1 = [
            per_seed[seed]["A1"][metric]
            for seed in SEEDS
        ]

        delta = [
            per_seed[seed]["A1"][metric]
            - per_seed[seed]["A0"][metric]
            for seed in SEEDS
        ]

        output[metric] = {
            "A0_mean": mean(a0),
            "A0_std": stdev(a0),
            "A1_mean": mean(a1),
            "A1_std": stdev(a1),
            "delta_mean": mean(delta),
            "delta_std": stdev(delta),
        }

    return output


def compare(
    label: str,
    actual: str,
    expected: str,
    failures: list[str],
) -> None:
    passed = actual == expected

    status = "PASS" if passed else "FAIL"

    print(
        f"{status:<4}  {label:<35} "
        f"actual={actual:<10} expected={expected}"
    )

    if not passed:
        failures.append(
            f"{label}: actual={actual}, expected={expected}"
        )


def main() -> None:
    calculated = calculate()

    failures: list[str] = []

    print("=" * 90)
    print("TABLE 2 REPRODUCIBILITY CHECK")
    print("=" * 90)

    for metric, expected in EXPECTED.items():

        values = calculated[metric]

        print()
        print(f"[{metric}]")

        if metric == "rPC15":

            actual_fields = {
                "A0_mean_percent":
                    f"{100 * values['A0_mean']:.2f}",
                "A0_std_percent":
                    f"{100 * values['A0_std']:.2f}",
                "A1_mean_percent":
                    f"{100 * values['A1_mean']:.2f}",
                "A1_std_percent":
                    f"{100 * values['A1_std']:.2f}",
                "delta_mean_pp":
                    f"{100 * values['delta_mean']:+.2f}",
                "delta_std_pp":
                    f"{100 * values['delta_std']:.2f}",
            }

        else:

            actual_fields = {
                "A0_mean":
                    f"{values['A0_mean']:.4f}",
                "A0_std":
                    f"{values['A0_std']:.4f}",
                "A1_mean":
                    f"{values['A1_mean']:.4f}",
                "A1_std":
                    f"{values['A1_std']:.4f}",
                "delta_mean_pp":
                    f"{100 * values['delta_mean']:+.2f}",
                "delta_std_pp":
                    f"{100 * values['delta_std']:.2f}",
            }

        for field, expected_value in expected.items():

            compare(
                f"{metric}/{field}",
                actual_fields[field],
                expected_value,
                failures,
            )

    print()
    print("=" * 90)

    if failures:

        print("TABLE 2 VERIFICATION FAILED")
        print("=" * 90)

        for failure in failures:
            print(" -", failure)

        raise SystemExit(1)

    print("TABLE 2 VERIFICATION PASSED")
    print(
        "All manuscript values are reproduced from "
        "the six frozen seed-level JSON files."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
