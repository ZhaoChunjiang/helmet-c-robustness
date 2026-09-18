#!/usr/bin/env python3
"""
Validate fixed SHWD train/val/test split manifests.

Checks:
1. Expected image counts.
2. Duplicate entries inside each split.
3. Overlap between train/val/test.
4. Optional file-existence verification.

Example
-------
python scripts/validate_split_manifest.py \
    --split-dir splits \
    --train-root /path/to/SHWD/images/train \
    --val-root /path/to/SHWD/images/val \
    --test-root /path/to/SHWD/images/test
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_COUNTS = {
    "train": 5457,
    "val": 607,
    "test": 1517,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fixed SHWD split manifests."
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("splits"),
        help="Directory containing train.txt, val.txt and test.txt.",
    )

    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help="Optional root directory of training images.",
    )

    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Optional root directory of validation images.",
    )

    parser.add_argument(
        "--test-root",
        type=Path,
        default=None,
        help="Optional root directory of test images.",
    )

    return parser.parse_args()


def load_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        entries = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return entries


def check_duplicates(name: str, entries: list[str]) -> None:
    unique = set(entries)

    if len(unique) != len(entries):
        duplicates = len(entries) - len(unique)

        raise RuntimeError(
            f"{name}: {duplicates} duplicate manifest entries detected."
        )


def check_expected_count(name: str, entries: list[str]) -> None:
    expected = EXPECTED_COUNTS[name]
    actual = len(entries)

    if actual != expected:
        raise RuntimeError(
            f"{name}: expected {expected} images, "
            f"but found {actual}."
        )


def check_overlap(
    train: list[str],
    val: list[str],
    test: list[str],
) -> None:
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    overlaps = {
        "train-val": train_set & val_set,
        "train-test": train_set & test_set,
        "val-test": val_set & test_set,
    }

    problems = {
        name: values
        for name, values in overlaps.items()
        if values
    }

    if problems:
        message = ["Cross-split overlap detected:"]

        for name, values in problems.items():
            message.append(
                f"  {name}: {len(values)} overlapping entries"
            )

        raise RuntimeError("\n".join(message))


def check_files_exist(
    name: str,
    entries: list[str],
    root: Path | None,
) -> None:
    if root is None:
        return

    if not root.exists():
        raise FileNotFoundError(
            f"{name} image root does not exist: {root}"
        )

    missing = []

    for relative_path in entries:
        image_path = root / relative_path

        if not image_path.exists():
            missing.append(relative_path)

    if missing:
        preview = "\n".join(
            f"  {item}"
            for item in missing[:10]
        )

        raise RuntimeError(
            f"{name}: {len(missing)} files listed in the manifest "
            f"were not found.\n"
            f"First missing entries:\n{preview}"
        )


def main() -> None:
    args = parse_args()

    manifests = {
        "train": load_manifest(
            args.split_dir / "train.txt"
        ),
        "val": load_manifest(
            args.split_dir / "val.txt"
        ),
        "test": load_manifest(
            args.split_dir / "test.txt"
        ),
    }

    roots = {
        "train": args.train_root,
        "val": args.val_root,
        "test": args.test_root,
    }

    print("=" * 64)
    print("SHWD split validation")
    print("=" * 64)

    for name, entries in manifests.items():
        check_duplicates(name, entries)
        check_expected_count(name, entries)
        check_files_exist(
            name,
            entries,
            roots[name],
        )

        print(
            f"{name:<5}: "
            f"{len(entries):>5} images  [PASS]"
        )

    check_overlap(
        manifests["train"],
        manifests["val"],
        manifests["test"],
    )

    total = sum(
        len(entries)
        for entries in manifests.values()
    )

    print("-" * 64)
    print(f"Total : {total} images")
    print("Duplicate entries : PASS")
    print("Cross-split overlap: PASS")
    print("=" * 64)
    print("VALIDATION PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
