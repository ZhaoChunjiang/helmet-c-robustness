#!/usr/bin/env python3
"""
Export fixed train/val/test image manifests for SHWD.

The script DOES NOT create a new random split.
It records an already-existing experimental split so that the
exact image membership can be reproduced and audited.

Example
-------
python scripts/export_split_manifest.py \
    --train-dir /path/to/SHWD/images/train \
    --val-dir /path/to/SHWD/images/val \
    --test-dir /path/to/SHWD/images/test \
    --output-dir splits
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
        description="Export fixed SHWD train/val/test manifests."
    )

    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing training images.",
    )

    parser.add_argument(
        "--val-dir",
        type=Path,
        required=True,
        help="Directory containing validation images.",
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test images.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits"),
        help="Directory used to save split manifests.",
    )

    return parser.parse_args()


def collect_images(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    files = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path.relative_to(root).as_posix())

    return sorted(files)


def save_manifest(items: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(item + "\n")


def check_overlap(
    train: list[str],
    val: list[str],
    test: list[str],
) -> None:
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    if train_val or train_test or val_test:
        raise RuntimeError(
            "Split overlap detected:\n"
            f"train-val: {len(train_val)}\n"
            f"train-test: {len(train_test)}\n"
            f"val-test: {len(val_test)}"
        )


def main() -> None:
    args = parse_args()

    train = collect_images(args.train_dir)
    val = collect_images(args.val_dir)
    test = collect_images(args.test_dir)

    check_overlap(train, val, test)

    save_manifest(
        train,
        args.output_dir / "train.txt",
    )

    save_manifest(
        val,
        args.output_dir / "val.txt",
    )

    save_manifest(
        test,
        args.output_dir / "test.txt",
    )

    summary_path = args.output_dir / "split_summary.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        file.write(f"train_images: {len(train)}\n")
        file.write(f"val_images: {len(val)}\n")
        file.write(f"test_images: {len(test)}\n")
        file.write(f"total_images: {len(train) + len(val) + len(test)}\n")

    print("=" * 60)
    print("SHWD fixed split manifest exported")
    print("=" * 60)
    print(f"Train : {len(train)}")
    print(f"Val   : {len(val)}")
    print(f"Test  : {len(test)}")
    print(f"Total : {len(train) + len(val) + len(test)}")
    print()
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
