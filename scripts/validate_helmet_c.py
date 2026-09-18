#!/usr/bin/env python3
"""
Validate a generated Helmet-C benchmark.

Checks:
1. All 15 corruption types exist.
2. All five severity levels exist for every corruption.
3. Every corruption/severity condition contains the expected number of images.
4. No unexpected corruption/severity directories are present.
5. manifest.csv is checked when available.

Example
-------
python scripts/validate_helmet_c.py \
    --helmet-c-root data/Helmet-C \
    --expected-images 1517
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the generated Helmet-C benchmark."
    )

    parser.add_argument(
        "--helmet-c-root",
        type=Path,
        required=True,
        help="Root directory of the generated Helmet-C benchmark.",
    )

    parser.add_argument(
        "--expected-images",
        type=int,
        default=1517,
        help="Expected number of images for each corruption/severity condition.",
    )

    return parser.parse_args()


def count_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_directories(
    root: Path,
    expected_images: int,
) -> None:

    expected_corruptions = set(CORRUPTIONS)

    actual_corruptions = {
        path.name
        for path in root.iterdir()
        if path.is_dir()
    }

    missing = expected_corruptions - actual_corruptions
    unexpected = actual_corruptions - expected_corruptions

    if missing:
        raise RuntimeError(
            "Missing corruption directories: "
            + ", ".join(sorted(missing))
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected corruption directories: "
            + ", ".join(sorted(unexpected))
        )

    print("=" * 72)
    print("Helmet-C directory validation")
    print("=" * 72)

    grand_total = 0

    for corruption in CORRUPTIONS:

        corruption_root = root / corruption

        for severity in SEVERITIES:

            condition_root = (
                corruption_root / f"severity_{severity}"
            )

            if not condition_root.exists():
                raise RuntimeError(
                    f"Missing directory: {condition_root}"
                )

            count = count_images(condition_root)

            if count != expected_images:
                raise RuntimeError(
                    f"{corruption} / severity_{severity}: "
                    f"expected {expected_images}, found {count}"
                )

            grand_total += count

            print(
                f"{corruption:<20} "
                f"severity_{severity}: "
                f"{count:>5}  [PASS]"
            )

    expected_total = (
        len(CORRUPTIONS)
        * len(SEVERITIES)
        * expected_images
    )

    if grand_total != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total images, "
            f"but found {grand_total}."
        )

    print("-" * 72)
    print(
        f"Conditions : "
        f"{len(CORRUPTIONS)} x {len(SEVERITIES)} "
        f"= {len(CORRUPTIONS) * len(SEVERITIES)}"
    )

    print(f"Images per condition : {expected_images}")
    print(f"Total corrupted images: {grand_total}")


def validate_manifest(
    root: Path,
    expected_images: int,
) -> None:

    manifest_path = root / "manifest.csv"

    if not manifest_path.exists():
        print()
        print(
            "manifest.csv not found: "
            "directory validation completed only."
        )
        return

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    expected_rows = (
        len(CORRUPTIONS)
        * len(SEVERITIES)
        * expected_images
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"manifest.csv: expected {expected_rows} rows, "
            f"found {len(rows)}."
        )

    required_fields = {
        "source_image",
        "output_image",
        "corruption",
        "severity",
        "seed",
    }

    if not rows:
        raise RuntimeError("manifest.csv is empty.")

    fields = set(rows[0].keys())

    missing_fields = required_fields - fields

    if missing_fields:
        raise RuntimeError(
            "manifest.csv missing fields: "
            + ", ".join(sorted(missing_fields))
        )

    output_paths = [
        row["output_image"]
        for row in rows
    ]

    if len(output_paths) != len(set(output_paths)):
        raise RuntimeError(
            "Duplicate output_image entries detected "
            "in manifest.csv."
        )

    condition_counts = {}

    for row in rows:

        corruption = row["corruption"]

        try:
            severity = int(row["severity"])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid severity value: {row['severity']}"
            ) from error

        if corruption not in CORRUPTIONS:
            raise RuntimeError(
                f"Unexpected corruption in manifest: "
                f"{corruption}"
            )

        if severity not in SEVERITIES:
            raise RuntimeError(
                f"Unexpected severity in manifest: "
                f"{severity}"
            )

        key = (corruption, severity)

        condition_counts[key] = (
            condition_counts.get(key, 0) + 1
        )

    for corruption in CORRUPTIONS:

        for severity in SEVERITIES:

            key = (corruption, severity)

            count = condition_counts.get(key, 0)

            if count != expected_images:
                raise RuntimeError(
                    f"Manifest condition "
                    f"{corruption}/severity_{severity}: "
                    f"expected {expected_images}, "
                    f"found {count}."
                )

    print("manifest.csv           : PASS")
    print(f"Manifest rows          : {len(rows)}")


def main() -> None:
    args = parse_args()

    if not args.helmet_c_root.exists():
        raise FileNotFoundError(
            f"Helmet-C root not found: "
            f"{args.helmet_c_root}"
        )

    validate_directories(
        args.helmet_c_root,
        args.expected_images,
    )

    validate_manifest(
        args.helmet_c_root,
        args.expected_images,
    )

    print("=" * 72)
    print("HELMET-C VALIDATION PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
