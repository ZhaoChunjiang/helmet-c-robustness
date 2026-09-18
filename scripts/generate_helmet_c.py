#!/usr/bin/env python3
"""
Generate the Helmet-C corruption benchmark.

Helmet-C contains 15 corruption types, each with five severity levels.
Corruption generation is deterministic for each
(image, corruption, severity) combination.

Example
-------
python scripts/generate_helmet_c.py \
    --input-root data/SHWD/test/images \
    --output-root data/Helmet-C \
    --seed 3407
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from imagecorruptions import corrupt
from tqdm import tqdm


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
        description="Generate deterministic Helmet-C corrupted images."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing the clean test images.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory used to store Helmet-C.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Base seed used for deterministic corruption generation.",
    )

    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=CORRUPTIONS,
        choices=CORRUPTIONS,
        help="Corruption types to generate.",
    )

    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        default=SEVERITIES,
        choices=SEVERITIES,
        help="Severity levels to generate.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing corrupted images.",
    )

    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    """Collect all supported images recursively."""
    images = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)


def make_deterministic_seed(
    base_seed: int,
    relative_path: Path,
    corruption_name: str,
    severity: int,
) -> int:
    """
    Generate a deterministic 32-bit seed.

    The result does not depend on filesystem traversal order.
    """
    key = (
        f"{base_seed}|"
        f"{relative_path.as_posix()}|"
        f"{corruption_name}|"
        f"{severity}"
    )

    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)

    return array


def apply_corruption(
    image: np.ndarray,
    corruption_name: str,
    severity: int,
    seed: int,
) -> np.ndarray:
    """
    Apply one ImageNet-C style corruption deterministically.
    """
    previous_state = np.random.get_state()

    try:
        np.random.seed(seed)

        corrupted = corrupt(
            image,
            corruption_name=corruption_name,
            severity=severity,
        )

    finally:
        np.random.set_state(previous_state)

    return np.asarray(corrupted, dtype=np.uint8)


def save_png(image: np.ndarray, path: Path) -> None:
    """Save corrupted image losslessly as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(
        path,
        format="PNG",
        compress_level=6,
    )


def main() -> None:
    args = parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {args.input_root}"
        )

    images = collect_images(args.input_root)

    if not images:
        raise RuntimeError(
            f"No supported images found in: {args.input_root}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Helmet-C benchmark generation")
    print("=" * 72)
    print(f"Input root      : {args.input_root}")
    print(f"Output root     : {args.output_root}")
    print(f"Images          : {len(images)}")
    print(f"Corruptions     : {len(args.corruptions)}")
    print(f"Severity levels : {args.severities}")
    print(f"Base seed       : {args.seed}")
    print("=" * 72)

    manifest_rows = []

    total = (
        len(images)
        * len(args.corruptions)
        * len(args.severities)
    )

    progress = tqdm(
        total=total,
        desc="Generating Helmet-C",
        unit="image",
    )

    for corruption_name in args.corruptions:

        for severity in args.severities:

            for source_path in images:

                relative_path = source_path.relative_to(
                    args.input_root
                )

                # Helmet-C images are stored losslessly as PNG.
                output_relative_path = (
                    Path(corruption_name)
                    / f"severity_{severity}"
                    / relative_path.with_suffix(".png")
                )

                output_path = (
                    args.output_root / output_relative_path
                )

                image_seed = make_deterministic_seed(
                    args.seed,
                    relative_path,
                    corruption_name,
                    severity,
                )

                if args.overwrite or not output_path.exists():

                    clean_image = load_rgb(source_path)

                    corrupted_image = apply_corruption(
                        clean_image,
                        corruption_name,
                        severity,
                        image_seed,
                    )

                    save_png(
                        corrupted_image,
                        output_path,
                    )

                manifest_rows.append(
                    {
                        "source_image": relative_path.as_posix(),
                        "output_image": output_relative_path.as_posix(),
                        "corruption": corruption_name,
                        "severity": severity,
                        "seed": image_seed,
                    }
                )

                progress.update(1)

    progress.close()

    manifest_path = args.output_root / "manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_image",
                "output_image",
                "corruption",
                "severity",
                "seed",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("Helmet-C generation completed.")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Generated conditions: "
          f"{len(args.corruptions)} corruptions "
          f"x {len(args.severities)} severities")


if __name__ == "__main__":
    main()
