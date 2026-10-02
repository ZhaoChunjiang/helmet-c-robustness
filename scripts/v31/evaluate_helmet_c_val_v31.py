#!/usr/bin/env python3
"""Portable public V3.1 Helmet-C-Val evaluator.

The script materializes each corruption condition as lossless PNG and evaluates
one or more frozen YOLO checkpoints with native Ultralytics model.val(). It is a
clean public implementation of the final V3.1 protocol; the original formal
AutoDL scripts are preserved separately in the authors' run archive.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from deterministic_corruptions_v31 import PROTOCOL_ID, corrupt_file

CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
WEATHER = {"snow", "frost", "fog"}
NONWEATHER11 = tuple(c for c in CORRUPTIONS if c not in WEATHER and c != "elastic_transform")
# Backward-compatible alias for older machine-readable locks.
SEEN11 = NONWEATHER11
SEVERITIES = (1, 2, 3, 4, 5)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def canonical_paths(data_yaml: Path):
    root = data_yaml.resolve().parent
    imdir = root / "images" / "val"
    lbdir = root / "labels" / "val"
    ims = sorted(p for p in imdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
    if len(ims) != 607:
        raise RuntimeError(f"Expected 607 validation images, got {len(ims)}")
    stems = [p.stem for p in ims]
    if len(set(stems)) != len(stems):
        dup = sorted({s for s in stems if stems.count(s) > 1})
        raise RuntimeError(
            "V3.1 image identity uses filename stems; duplicate stems are not allowed: "
            + ", ".join(dup[:10])
        )
    counts = {0: 0, 1: 0}
    for p in ims:
        lab = lbdir / f"{p.stem}.txt"
        if not lab.is_file():
            raise RuntimeError(f"Missing label: {lab}")
        for line in lab.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                cls = int(float(line.split()[0]))
                counts[cls] = counts.get(cls, 0) + 1
    if counts != {0: 747, 1: 9178}:
        raise RuntimeError(f"Unexpected validation GT counts: {counts}")
    return ims, lbdir


def write_condition(ims, lbdir: Path, out_root: Path, corr: str, sev: int):
    images = out_root / "images" / "val"
    labels = out_root / "labels" / "val"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for p in ims:
        image_id, arr = corrupt_file(p, corr, sev)
        Image.fromarray(arr, mode="RGB").save(images / f"{image_id}.png", format="PNG")
        shutil.copy2(lbdir / f"{image_id}.txt", labels / f"{image_id}.txt")

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(out_root.resolve()) + "\n"
        "train: images/val\n"
        "val: images/val\n"
        "test: images/val\n"
        "names:\n  0: hat\n  1: person\n",
        encoding="utf-8",
    )
    return data_yaml


def metric_dict(metrics):
    p, r, ap50, ap = metrics.box.mean_results()
    return {"precision": float(p), "recall": float(r), "AP50": float(ap50), "AP50_95": float(ap)}


def official_val(model: YOLO, data: Path, project: Path, name: str, device: str):
    # Historical V3.1 numerical contract: rect/half were intentionally omitted
    # in the formal AutoDL runner. A V3.7 protocol-reconciliation check showed
    # that this exact call reproduces the frozen A0 seed-0 clean AP50:95
    # (0.6089874629020515), whereas forcing rect=False changes the result.
    m = model.val(
        data=str(data), split="val", imgsz=640, batch=16, workers=8,
        device=device, conf=0.001, iou=0.7, max_det=300,
        plots=False, save=False,
        project=str(project), name=name, exist_ok=True, verbose=False,
    )
    return metric_dict(m)


def summary(clean: float, values: dict[tuple[str, int], float]):
    all_vals = [values[(c, s)] for c in CORRUPTIONS for s in SEVERITIES]
    weather = [values[(c, s)] for c in WEATHER for s in SEVERITIES]
    nonweather = [values[(c, s)] for c in NONWEATHER11 for s in SEVERITIES]
    mean = lambda xs: sum(xs) / len(xs)
    mpc = mean(all_vals)
    w = mean(weather)
    return {
        "clean_AP50_95": clean,
        "mPC15": mpc,
        "rPC15": mpc / clean,
        "weather_mPC": w,
        "weather_rPC": w / clean,
        "nonweather11_mPC": mean(nonweather),
        "seen11_mPC": mean(nonweather),  # legacy alias retained for compatibility
        "per_corruption": {c: mean([values[(c, s)] for s in SEVERITIES]) for c in CORRUPTIONS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", nargs=3, action="append", metavar=("LABEL", "BEST_PT", "EXPECTED_CLEAN_AP5095"), required=True)
    ap.add_argument("--audit-marker", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=str, default="0", help="Ultralytics validation device; manuscript runs used CUDA device 0")
    ap.add_argument("--clean-tol", type=float, default=1e-4)
    args = ap.parse_args()

    audit = json.loads(args.audit_marker.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Compatible PASS determinism audit required")
    if not audit.get("full_impulse_s1", {}).get("match") or not audit.get("full_glass_s1", {}).get("match"):
        raise RuntimeError("Full impulse/glass reproducibility audit must pass")

    ims, lbdir = canonical_paths(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    models = []
    for label, best, expected in args.model:
        model = YOLO(best)
        clean = official_val(model, args.data, args.output / "native_val", f"{label}_clean", args.device)
        diff = abs(clean["AP50_95"] - float(expected))
        if diff > args.clean_tol:
            raise RuntimeError(f"{label} clean AP mismatch {diff} > {args.clean_tol}")
        models.append((label, model, clean["AP50_95"]))

    values = {label: {} for label, _, _ in models}
    rows = []
    with tempfile.TemporaryDirectory(prefix="helmet_c_v31_") as td:
        td = Path(td)
        for corr in CORRUPTIONS:
            for sev in SEVERITIES:
                cond = td / f"{corr}_s{sev}"
                data = write_condition(ims, lbdir, cond, corr, sev)
                for label, model, _ in models:
                    m = official_val(model, data, args.output / "native_val", f"{label}_{corr}_s{sev}", args.device)
                    values[label][(corr, sev)] = m["AP50_95"]
                    rows.append({"model": label, "corruption": corr, "severity": sev, **m})
                shutil.rmtree(cond)

    with (args.output / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "corruption", "severity", "precision", "recall", "AP50", "AP50_95"])
        w.writeheader(); w.writerows(rows)

    result = {"status": "PASS", "protocol_id": PROTOCOL_ID, "models": {}}
    for label, _, clean in models:
        result["models"][label] = summary(clean, values[label])
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("V3.1 HELMET-C-VAL EVALUATION: PASS")


if __name__ == "__main__":
    main()
