#!/usr/bin/env python3
"""Cross-process and full-condition reproducibility audit for V3.1."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from deterministic_corruptions_v31 import (
    PROTOCOL_ID,
    array_digest,
    corrupt_file,
)

CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _task(args):
    path, corr, sev = args
    image_id, arr = corrupt_file(Path(path), corr, sev)
    return image_id, array_digest(arr)


def val_images(data_yaml: Path) -> list[Path]:
    root = data_yaml.resolve().parent
    imdir = root / "images" / "val"
    ims = sorted(p for p in imdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
    if len(ims) != 607:
        raise RuntimeError(f"Expected 607 validation images, got {len(ims)}")
    return ims


def fresh_one(path: Path, corr: str, sev: int):
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        return ex.submit(_task, (str(path), corr, sev)).result()


def quick_audit(ims: list[Path]):
    probe = ims[0]
    rows = []
    for corr, sev in [(c, 1) for c in CORRUPTIONS] + [("impulse_noise", 5), ("glass_blur", 5)]:
        _, a = fresh_one(probe, corr, sev)
        _, b = fresh_one(probe, corr, sev)
        ok = a == b
        print(f"{corr:18s} s{sev}: {'PASS' if ok else 'FAIL'}")
        rows.append({"corruption": corr, "severity": sev, "digest_A": a, "digest_B": b, "match": ok})
        if not ok:
            raise RuntimeError(f"Cross-process reproducibility failed: {corr} s{sev}")
    return rows


def full_condition(ims: list[Path], corr: str, sev: int, workers: int):
    tasks = [(str(p), corr, sev) for p in ims]

    def one_pass(tag: str):
        ctx = mp.get_context("spawn")
        digests = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for i, (_, digest) in enumerate(ex.map(_task, tasks, chunksize=1), start=1):
                digests.append(digest)
                if i % 150 == 0 or i == len(tasks):
                    print(f"{corr} s{sev} {tag}: {i}/{len(tasks)}", flush=True)
        return hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()

    a = one_pass("REPRO_A")
    b = one_pass("REPRO_B")
    print(f"{corr} s{sev} REPRO_A {a}")
    print(f"{corr} s{sev} REPRO_B {b}")
    if a != b:
        raise RuntimeError(f"Full {corr} s{sev} reproducibility failed")
    return {"corruption": corr, "severity": sev, "digest_A": a, "digest_B": b, "match": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="SHWD data.yaml with images/val and labels/val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--output", type=Path, default=Path("V31_DETERMINISM_AUDIT.json"))
    args = ap.parse_args()

    ims = val_images(args.data)
    quick = quick_audit(ims)
    impulse = full_condition(ims, "impulse_noise", 1, args.workers)
    glass = full_condition(ims, "glass_blur", 1, args.workers)

    obj = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "quick_cross_process": quick,
        "full_impulse_s1": impulse,
        "full_glass_s1": glass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print("V3.1 DETERMINISM AUDIT: PASS")
    print("audit marker =", args.output)


if __name__ == "__main__":
    main()
