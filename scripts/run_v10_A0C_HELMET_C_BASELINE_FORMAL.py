#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v10_A0C_HELMET_C_BASELINE_FORMAL.py

新论文 V1.0 — A0-C Helmet-C 15×5 退化鲁棒性基线正式程序
========================================================

用途
----
使用已经锁定的 A0 YOLO11n seed=0 best.pt，在 SHWD 正式 test=1517 图像上，
按照 15 类 Common Corruptions × 5 个 severity 构造 Helmet-C 评价协议，
得到 clean -> corruption 的鲁棒性基线。

本程序不重新训练模型，不保存 113,775 张退化图像；退化图像在内存中即时生成，
只缓存每个 corruption/severity 的预测结果，显著节省磁盘空间。

冻结的 15 类退化
----------------
Noise:
    gaussian_noise
    shot_noise
    impulse_noise
Blur:
    defocus_blur
    glass_blur
    motion_blur
    zoom_blur
Weather / imaging:
    snow
    frost
    fog
    brightness
Digital:
    contrast
    elastic_transform
    pixelate
    jpeg_compression

Severity:
    1, 2, 3, 4, 5

正式核心指标
------------
对 ALL / ES / S / M / L，且分别对 all-macro / hat / person 输出：
- AP50
- AP50:95
- R50_max
- R50@conf=0.25

鲁棒性汇总：
- mPC15        : 15×5 条件上的平均 AP50:95
- rPC15        : mPC15 / clean_AP50:95
- absolute drop: clean_AP50:95 - mPC15
- severity profile
- per-corruption mean across severity
- ES / S / hat / person 的对应鲁棒性指标

尺度定义（与 A0-S 完全一致）
----------------------------
统一到 imgsz=640 后：
ES : area < 16^2
S  : 16^2 <= area < 32^2
M  : 32^2 <= area < 96^2
L  : area >= 96^2

正式锁定
--------
A0 checkpoint:
    /root/autodl-tmp/checkpoints/A0_baseline/seed_0/best.pt
A0 best.pt SHA256:
    9173932805b7589a337ed7838e1abf52ec949e893f11ac1a9d1bc22122b8b5d2
Formal data:
    /root/autodl-tmp/helmet_yolo_v10/data.yaml
Formal test:
    1517 images
    hat=1878, person=22558, total=24436
Clean A0-S reference:
    /root/autodl-tmp/results/A0S_small_object/seed_0/A0S_metrics_by_size_class.csv

依赖
----
- numpy==1.26.4（当前 A0 冻结环境）
- pillow
- matplotlib
- torch
- ultralytics
- imagecorruptions

安装 Common Corruptions 依赖：
    pip install imagecorruptions==1.1.2

重要说明
--------
1. 本程序用 imagecorruptions 的标准 corruption/severity 实现，不自行改 severity 参数。
2. elastic_transform 会发生轻微几何形变；为保持与经典 corruption benchmark 可比，
   标签仍沿用原始框。程序会同时额外给出 mPC14_no_elastic，作为几何标签敏感性参考。
3. 推理采用显式小批次内存图像输入，并带 CUDA OOM 自动降 batch；
   默认 batch=8，若 OOM 自动 8 -> 4 -> 2 -> 1。
4. 每个 corruption/severity 独立缓存。即使中途关机，再运行会自动从未完成条件继续。
5. 不要升级/更改 A0 已冻结的 PyTorch / Ultralytics / NumPy 环境。

运行
----
cd /root/autodl-tmp
python run_v10_A0C_HELMET_C_BASELINE_FORMAL.py

如需查看进度：
    tail -f /root/autodl-tmp/results/A0C_helmet_c/seed_0/A0C_PROGRESS.log

只测试 evaluator：
    python run_v10_A0C_HELMET_C_BASELINE_FORMAL.py --self-test

强制重算某些已缓存条件：
    python run_v10_A0C_HELMET_C_BASELINE_FORMAL.py --force-corruption motion_blur
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_VERSION = "1.0.0-A0C"

DEFAULT_DATA = Path("/root/autodl-tmp/helmet_yolo_v10/data.yaml")
DEFAULT_BEST = Path("/root/autodl-tmp/checkpoints/A0_baseline/seed_0/best.pt")
DEFAULT_A0S_METRICS = Path(
    "/root/autodl-tmp/results/A0S_small_object/seed_0/A0S_metrics_by_size_class.csv"
)
DEFAULT_A0S_PROTOCOL = Path(
    "/root/autodl-tmp/results/A0S_small_object/seed_0/A0S_PROTOCOL_LOCK.json"
)
DEFAULT_RESULTS_ROOT = Path("/root/autodl-tmp/results/A0C_helmet_c/seed_0")
DEFAULT_ZIP = Path(
    "/root/autodl-tmp/results/A0C_helmet_c/"
    "A0C_HELMET_C_RESULTS_TO_UPLOAD.zip"
)

EXPECTED_BEST_SHA256 = (
    "9173932805b7589a337ed7838e1abf52"
    "ec949e893f11ac1a9d1bc22122b8b5d2"
)
EXPECTED_TEST_IMAGES = 1517
EXPECTED_TEST_CLASS_OBJECTS = {0: 1878, 1: 22558}
EXPECTED_NAMES = {0: "hat", 1: "person"}

DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_CONF = 0.001
DEFAULT_IOU_NMS = 0.7
DEFAULT_MAX_DET = 300
DEFAULT_DEVICE = "0"

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

SIZE_BINS = {
    "ALL": (0.0, None),
    "ES": (0.0, 16.0 ** 2),
    "S": (16.0 ** 2, 32.0 ** 2),
    "M": (32.0 ** 2, 96.0 ** 2),
    "L": (96.0 ** 2, None),
}

CORRUPTIONS: Tuple[str, ...] = (
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
)

SEVERITIES: Tuple[int, ...] = (1, 2, 3, 4, 5)

# Metrics we most need for method design and paper tables.
KEY_VIEWS = (
    ("ALL", "all_macro"),
    ("ALL", "hat"),
    ("ALL", "person"),
    ("ES", "all_macro"),
    ("ES", "hat"),
    ("ES", "person"),
    ("S", "all_macro"),
    ("S", "hat"),
    ("S", "person"),
    ("M", "all_macro"),
    ("L", "all_macro"),
)


class FormalA0CError(RuntimeError):
    pass


def die(msg: str) -> None:
    raise FormalA0CError(msg)


@dataclass
class GT:
    image_id: str
    cls_id: int
    xyxy: np.ndarray
    area640: float


@dataclass
class Pred:
    image_id: str
    cls_id: int
    conf: float
    xyxy: np.ndarray
    area640: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V1.0 A0-C Helmet-C 15x5 formal robustness baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--best", type=Path, default=DEFAULT_BEST)
    p.add_argument("--a0s-metrics", type=Path, default=DEFAULT_A0S_METRICS)
    p.add_argument("--a0s-protocol", type=Path, default=DEFAULT_A0S_PROTOCOL)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou-nms", type=float, default=DEFAULT_IOU_NMS)
    p.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument(
        "--force-corruption",
        action="append",
        default=[],
        help="Recompute all severities for this corruption; may be repeated",
    )
    p.add_argument(
        "--only-corruption",
        action="append",
        default=[],
        help="Run only selected corruption(s); useful for debugging",
    )
    p.add_argument(
        "--only-severity",
        type=int,
        action="append",
        default=[],
        help="Run only selected severity value(s); useful for debugging",
    )
    p.add_argument(
        "--include-cache-in-zip",
        action="store_true",
        help="Include large prediction caches in the upload ZIP",
    )
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Mapping):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return str(v)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: jsonable(row.get(k)) for k in fieldnames})


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {text}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: Sequence[str]) -> Dict[str, Any]:
    try:
        r = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {
            "command": " ".join(cmd),
            "returncode": r.returncode,
            "stdout": r.stdout.strip(),
            "stderr": r.stderr.strip(),
        }
    except Exception as e:
        return {
            "command": " ".join(cmd),
            "returncode": None,
            "stdout": "",
            "stderr": repr(e),
        }


def import_runtime():
    try:
        import torch
    except Exception as e:
        raise FormalA0CError("PyTorch import failed.") from e

    try:
        import ultralytics
        from ultralytics import YOLO
    except Exception as e:
        raise FormalA0CError(
            "Ultralytics import failed. Keep the frozen A0 environment."
        ) from e

    try:
        from PIL import Image
    except Exception as e:
        raise FormalA0CError(
            "Pillow import failed: pip install pillow"
        ) from e

    # NumPy 1.24+ removed old aliases that some older corruption dependencies
    # may still reference. Add harmless compatibility aliases before import.
    alias_pairs = {
        "int": int,
        "float": float,
        "bool": bool,
        "object": object,
    }
    for name, obj in alias_pairs.items():
        if name not in np.__dict__:
            setattr(np, name, obj)

    try:
        import imagecorruptions
        from imagecorruptions import corrupt
    except Exception as e:
        raise FormalA0CError(
            "imagecorruptions is required for the frozen Helmet-C protocol.\n"
            "Install it in the SAME A0 environment:\n"
            "    pip install imagecorruptions==1.1.2"
        ) from e

    return torch, ultralytics, YOLO, Image, imagecorruptions, corrupt


def in_size_bin(area640: float, bin_name: str) -> bool:
    lo, hi = SIZE_BINS[bin_name]
    return area640 >= lo and (hi is None or area640 < hi)


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    area_a = (
        max(0.0, box[2] - box[0])
        * max(0.0, box[3] - box[1])
    )
    area_b = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = area_a + area_b - inter
    return np.divide(inter, np.maximum(union, 1e-12))


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0 or precision.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0.0, 1.0, 101)
    y = np.interp(x, mrec, mpre)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def eval_class_bin_at_iou(
    gts: List[GT],
    preds: List[Pred],
    cls_id: int,
    bin_name: str,
    iou_thr: float,
    fixed_conf: Optional[float] = None,
) -> Dict[str, Any]:
    gt_by_image_eligible: Dict[str, List[GT]] = defaultdict(list)
    gt_by_image_ignore: Dict[str, List[GT]] = defaultdict(list)

    for gt in gts:
        if gt.cls_id != cls_id:
            continue
        if bin_name == "ALL" or in_size_bin(gt.area640, bin_name):
            gt_by_image_eligible[gt.image_id].append(gt)
        else:
            gt_by_image_ignore[gt.image_id].append(gt)

    n_gt = sum(len(v) for v in gt_by_image_eligible.values())
    if n_gt == 0:
        return {
            "n_gt": 0,
            "n_eval_preds": 0,
            "n_ignored_preds": 0,
            "tp_total": 0,
            "fp_total": 0,
            "recall_curve": np.array([], dtype=np.float64),
            "precision_curve": np.array([], dtype=np.float64),
            "ap": float("nan"),
            "recall_final": float("nan"),
        }

    cls_preds = [
        p
        for p in preds
        if p.cls_id == cls_id
        and (fixed_conf is None or p.conf >= fixed_conf)
    ]
    cls_preds.sort(key=lambda p: p.conf, reverse=True)

    matched: Dict[str, np.ndarray] = {
        image_id: np.zeros(len(arr), dtype=bool)
        for image_id, arr in gt_by_image_eligible.items()
    }

    tp_flags: List[float] = []
    fp_flags: List[float] = []
    ignored_count = 0

    for pred in cls_preds:
        eligible = gt_by_image_eligible.get(pred.image_id, [])
        ignored = gt_by_image_ignore.get(pred.image_id, [])

        did_match = False
        if eligible:
            eboxes = np.stack([x.xyxy for x in eligible], axis=0)
            ious = box_iou_one_to_many(pred.xyxy, eboxes)
            order = np.argsort(-ious)
            for j in order:
                if ious[j] < iou_thr:
                    break
                if not matched[pred.image_id][j]:
                    matched[pred.image_id][j] = True
                    did_match = True
                    break

        if did_match:
            tp_flags.append(1.0)
            fp_flags.append(0.0)
            continue

        if bin_name != "ALL" and ignored:
            iboxes = np.stack([x.xyxy for x in ignored], axis=0)
            if np.max(box_iou_one_to_many(pred.xyxy, iboxes)) >= iou_thr:
                ignored_count += 1
                continue

        if bin_name != "ALL" and not in_size_bin(pred.area640, bin_name):
            ignored_count += 1
            continue

        tp_flags.append(0.0)
        fp_flags.append(1.0)

    if not tp_flags:
        return {
            "n_gt": n_gt,
            "n_eval_preds": 0,
            "n_ignored_preds": ignored_count,
            "tp_total": 0,
            "fp_total": 0,
            "recall_curve": np.array([], dtype=np.float64),
            "precision_curve": np.array([], dtype=np.float64),
            "ap": 0.0,
            "recall_final": 0.0,
        }

    tp = np.asarray(tp_flags, dtype=np.float64)
    fp = np.asarray(fp_flags, dtype=np.float64)
    tpc = np.cumsum(tp)
    fpc = np.cumsum(fp)

    recall = tpc / max(float(n_gt), 1e-12)
    precision = tpc / np.maximum(tpc + fpc, 1e-12)

    return {
        "n_gt": n_gt,
        "n_eval_preds": len(tp_flags),
        "n_ignored_preds": ignored_count,
        "tp_total": int(tpc[-1]) if len(tpc) else 0,
        "fp_total": int(fpc[-1]) if len(fpc) else 0,
        "recall_curve": recall,
        "precision_curve": precision,
        "ap": compute_ap(recall, precision),
        "recall_final": float(recall[-1]) if len(recall) else 0.0,
    }


def eval_class_bin(
    gts: List[GT],
    preds: List[Pred],
    cls_id: int,
    bin_name: str,
) -> Dict[str, Any]:
    ap_by_iou: List[float] = []
    r50_max = None
    n_gt = None
    n_eval_preds = None
    n_ignored_preds = None

    for iou_thr in IOU_THRESHOLDS:
        r = eval_class_bin_at_iou(
            gts=gts,
            preds=preds,
            cls_id=cls_id,
            bin_name=bin_name,
            iou_thr=float(iou_thr),
            fixed_conf=None,
        )
        ap_by_iou.append(float(r["ap"]))
        if abs(float(iou_thr) - 0.50) < 1e-9:
            r50_max = float(r["recall_final"])
            n_gt = int(r["n_gt"])
            n_eval_preds = int(r["n_eval_preds"])
            n_ignored_preds = int(r["n_ignored_preds"])

    fixed = eval_class_bin_at_iou(
        gts=gts,
        preds=preds,
        cls_id=cls_id,
        bin_name=bin_name,
        iou_thr=0.50,
        fixed_conf=0.25,
    )

    valid_aps = [x for x in ap_by_iou if not math.isnan(x)]
    return {
        "size_bin": bin_name,
        "class_id": cls_id,
        "class_name": EXPECTED_NAMES.get(cls_id, str(cls_id)),
        "gt_count": n_gt,
        "AP50": ap_by_iou[0] if ap_by_iou else float("nan"),
        "AP50_95": (
            float(np.mean(valid_aps))
            if valid_aps
            else float("nan")
        ),
        "R50_max": r50_max,
        "R50_at_conf_025": float(fixed["recall_final"]),
        "eval_predictions_at_iou50": n_eval_preds,
        "ignored_predictions_at_iou50": n_ignored_preds,
    }


def aggregate_macro(rows: List[dict], bin_name: str) -> dict:
    target = [
        r
        for r in rows
        if r["size_bin"] == bin_name
        and r["class_id"] in (0, 1)
    ]

    def mean_field(name: str) -> float:
        vals = [
            float(r[name])
            for r in target
            if r[name] is not None
            and not math.isnan(float(r[name]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "size_bin": bin_name,
        "class_id": "all",
        "class_name": "all_macro",
        "gt_count": int(sum(int(r["gt_count"]) for r in target)),
        "AP50": mean_field("AP50"),
        "AP50_95": mean_field("AP50_95"),
        "R50_max": mean_field("R50_max"),
        "R50_at_conf_025": mean_field("R50_at_conf_025"),
        "eval_predictions_at_iou50": int(
            sum(int(r["eval_predictions_at_iou50"]) for r in target)
        ),
        "ignored_predictions_at_iou50": int(
            sum(int(r["ignored_predictions_at_iou50"]) for r in target)
        ),
    }


def parse_formal_test(
    data_yaml: Path,
    imgsz: int,
    Image,
) -> Tuple[List[GT], List[dict], List[dict]]:
    root = data_yaml.resolve().parent
    image_dir = root / "images" / "test"
    label_dir = root / "labels" / "test"

    if not image_dir.is_dir() or not label_dir.is_dir():
        die(f"Formal test split missing under: {root}")

    image_paths = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() or p.is_symlink()
    )

    if len(image_paths) != EXPECTED_TEST_IMAGES:
        die(
            f"Test image count mismatch: "
            f"{len(image_paths)} != {EXPECTED_TEST_IMAGES}"
        )

    gts: List[GT] = []
    image_manifest: List[dict] = []
    class_counts: Counter = Counter()
    size_counts: Counter = Counter()
    size_class_counts: Counter = Counter()

    for idx, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem
        label_path = label_dir / f"{image_id}.txt"
        if not label_path.is_file():
            die(f"Missing label: {label_path}")

        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception as e:
            raise FormalA0CError(
                f"Cannot read image size: {image_path}: {e}"
            ) from e

        scale = min(float(imgsz) / float(w), float(imgsz) / float(h))
        object_count = 0

        lines = [
            x.strip()
            for x in label_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if x.strip()
        ]

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                die(
                    f"Invalid YOLO label line: "
                    f"{label_path}:{line_no}: {line}"
                )

            cls_id = int(float(parts[0]))
            if cls_id not in (0, 1):
                die(
                    f"Unexpected class id {cls_id}: "
                    f"{label_path}:{line_no}"
                )

            xc, yc, bw, bh = map(float, parts[1:])
            box_w = bw * w
            box_h = bh * h
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h
            area640 = max(0.0, box_w * scale) * max(
                0.0, box_h * scale
            )

            gts.append(
                GT(
                    image_id=image_id,
                    cls_id=cls_id,
                    xyxy=np.asarray(
                        [x1, y1, x2, y2],
                        dtype=np.float64,
                    ),
                    area640=float(area640),
                )
            )
            class_counts[cls_id] += 1
            object_count += 1

            for bin_name in ("ES", "S", "M", "L"):
                if in_size_bin(area640, bin_name):
                    size_counts[bin_name] += 1
                    size_class_counts[(bin_name, cls_id)] += 1
                    break

        image_manifest.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "width": w,
                "height": h,
                "resize_scale_to_640": scale,
                "gt_objects": object_count,
            }
        )

        if idx % 300 == 0 or idx == len(image_paths):
            print(f"  GT parsing: {idx}/{len(image_paths)}")

    for cls_id, expected in EXPECTED_TEST_CLASS_OBJECTS.items():
        got = int(class_counts[cls_id])
        if got != expected:
            die(
                f"Test object count mismatch for "
                f"{EXPECTED_NAMES[cls_id]}: {got} != {expected}"
            )

    size_rows = []
    for bin_name in ("ES", "S", "M", "L"):
        for cls_id in (0, 1):
            size_rows.append(
                {
                    "size_bin": bin_name,
                    "class_id": cls_id,
                    "class_name": EXPECTED_NAMES[cls_id],
                    "gt_count": int(
                        size_class_counts[(bin_name, cls_id)]
                    ),
                }
            )
        size_rows.append(
            {
                "size_bin": bin_name,
                "class_id": "all",
                "class_name": "all",
                "gt_count": int(size_counts[bin_name]),
            }
        )

    return gts, image_manifest, size_rows


def load_clean_a0s_metrics(
    path: Path,
) -> Dict[Tuple[str, str], dict]:
    if not path.is_file():
        die(
            f"A0-S clean metrics not found: {path}\n"
            "Run A0-S first so rPC uses the identical frozen evaluator."
        )

    out: Dict[Tuple[str, str], dict] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["size_bin"], row["class_name"])
            parsed = dict(row)
            for field in (
                "gt_count",
                "AP50",
                "AP50_95",
                "R50_max",
                "R50_at_conf_025",
            ):
                if parsed.get(field, "") != "":
                    try:
                        parsed[field] = float(parsed[field])
                    except Exception:
                        pass
            out[key] = parsed

    for key in KEY_VIEWS:
        if key not in out:
            die(f"Required clean A0-S metric missing: {key}")
    return out


def validate_a0s_protocol(
    protocol_path: Path,
    best_sha: str,
) -> None:
    if not protocol_path.is_file():
        die(f"A0-S protocol lock missing: {protocol_path}")
    obj = json.loads(protocol_path.read_text(encoding="utf-8"))
    got = (
        obj.get("checkpoint", {})
        .get("sha256")
    )
    if got != best_sha:
        die(
            "A0-S protocol checkpoint does not match A0-C checkpoint.\n"
            f"A0-S: {got}\nA0-C: {best_sha}"
        )


def prediction_area640(
    xyxy: np.ndarray,
    w: int,
    h: int,
    imgsz: int,
) -> float:
    scale = min(float(imgsz) / float(w), float(imgsz) / float(h))
    bw = max(0.0, float(xyxy[2] - xyxy[0]))
    bh = max(0.0, float(xyxy[3] - xyxy[1]))
    return float((bw * scale) * (bh * scale))


def condition_cache_paths(
    results_root: Path,
    corruption_name: str,
    severity: int,
) -> Tuple[Path, Path, Path]:
    cache_dir = results_root / "prediction_cache"
    metrics_dir = results_root / "condition_metrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{corruption_name}_s{severity}"
    return (
        cache_dir / f"{stem}.jsonl.gz",
        cache_dir / f"{stem}.meta.json",
        metrics_dir / f"{stem}.metrics.json",
    )


def condition_protocol(
    best_sha: str,
    data_sha: str,
    corruption_name: str,
    severity: int,
    imgsz: int,
    conf: float,
    iou_nms: float,
    max_det: int,
    imagecorruptions_version: str,
) -> dict:
    return {
        "best_sha256": best_sha,
        "data_yaml_sha256": data_sha,
        "corruption": corruption_name,
        "severity": severity,
        "imgsz": imgsz,
        "conf": conf,
        "iou_nms": iou_nms,
        "max_det": max_det,
        "imagecorruptions_version": imagecorruptions_version,
    }


def valid_condition_cache(
    cache_path: Path,
    meta_path: Path,
    metrics_path: Path,
    expected_protocol: dict,
) -> bool:
    if not (
        cache_path.is_file()
        and meta_path.is_file()
        and metrics_path.is_file()
    ):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
    except Exception:
        return False

    return (
        meta.get("status") == "COMPLETE"
        and meta.get("protocol") == expected_protocol
        and metrics.get("status") == "PASS"
        and metrics.get("protocol") == expected_protocol
    )


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda oom" in text
        or "cudaoutofmemoryerror" in text
    )


def apply_corruption(
    image_np: np.ndarray,
    corruption_name: str,
    severity: int,
    corrupt_fn,
) -> np.ndarray:
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    out = corrupt_fn(
        image_np,
        corruption_name=corruption_name,
        severity=severity,
    )
    out = np.asarray(out)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    if out.shape[-1] == 4:
        out = out[..., :3]
    return np.ascontiguousarray(out)


def predict_chunk_with_backoff(
    model,
    arrays: List[np.ndarray],
    ids: List[str],
    rows: List[dict],
    imgsz: int,
    conf: float,
    iou_nms: float,
    max_det: int,
    device: str,
    torch_mod,
) -> Tuple[List[dict], int]:
    """
    Predict a chunk. If CUDA OOM happens, recursively split until batch=1.
    Returns (serialized_image_predictions, largest_successful_batch_here).
    """
    if len(arrays) == 0:
        return [], 0

    try:
        results = model.predict(
            source=arrays if len(arrays) > 1 else arrays[0],
            imgsz=imgsz,
            conf=conf,
            iou=iou_nms,
            max_det=max_det,
            device=device,
            batch=len(arrays),
            stream=False,
            save=False,
            verbose=False,
        )

        if not isinstance(results, list):
            results = list(results)

        if len(results) != len(arrays):
            die(
                f"Ultralytics returned {len(results)} results "
                f"for {len(arrays)} in-memory images"
            )

        serialized = []

        for result, image_id, row in zip(results, ids, rows):
            w = int(row["width"])
            h = int(row["height"])
            boxes_out = []

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy()

                for b, c, cls_raw in zip(xyxy, confs, classes):
                    cls_id = int(cls_raw)
                    if cls_id not in (0, 1):
                        continue
                    boxes_out.append(
                        {
                            "cls_id": cls_id,
                            "conf": float(c),
                            "xyxy": [
                                float(x)
                                for x in b.tolist()
                            ],
                            "area640": prediction_area640(
                                b, w, h, imgsz
                            ),
                        }
                    )

            serialized.append(
                {
                    "image_id": image_id,
                    "width": w,
                    "height": h,
                    "predictions": boxes_out,
                }
            )

        del results
        return serialized, len(arrays)

    except RuntimeError as e:
        if not is_cuda_oom(e):
            raise

        try:
            torch_mod.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        if len(arrays) == 1:
            raise FormalA0CError(
                "CUDA OOM even at batch=1. "
                "Check for another GPU process with nvidia-smi."
            ) from e

        mid = max(1, len(arrays) // 2)
        left, b1 = predict_chunk_with_backoff(
            model=model,
            arrays=arrays[:mid],
            ids=ids[:mid],
            rows=rows[:mid],
            imgsz=imgsz,
            conf=conf,
            iou_nms=iou_nms,
            max_det=max_det,
            device=device,
            torch_mod=torch_mod,
        )
        right, b2 = predict_chunk_with_backoff(
            model=model,
            arrays=arrays[mid:],
            ids=ids[mid:],
            rows=rows[mid:],
            imgsz=imgsz,
            conf=conf,
            iou_nms=iou_nms,
            max_det=max_det,
            device=device,
            torch_mod=torch_mod,
        )
        return left + right, max(b1, b2)


def generate_condition_predictions(
    model,
    image_manifest: List[dict],
    cache_path: Path,
    meta_path: Path,
    protocol: dict,
    corruption_name: str,
    severity: int,
    requested_batch: int,
    imgsz: int,
    conf: float,
    iou_nms: float,
    max_det: int,
    device: str,
    torch_mod,
    Image,
    corrupt_fn,
    progress_log: Path,
) -> dict:
    temp_jsonl = cache_path.with_suffix("")
    temp_jsonl.unlink(missing_ok=True)
    cache_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)

    start = time.time()
    total_preds = 0
    processed = 0
    max_success_batch = 0

    append_log(
        progress_log,
        f"START {corruption_name} severity={severity}",
    )

    with temp_jsonl.open("w", encoding="utf-8") as fout:
        for chunk_start in range(
            0, len(image_manifest), requested_batch
        ):
            rows = image_manifest[
                chunk_start: chunk_start + requested_batch
            ]

            arrays: List[np.ndarray] = []
            ids: List[str] = []

            for row in rows:
                image_path = Path(row["image_path"])
                try:
                    with Image.open(image_path) as im:
                        image_np = np.asarray(
                            im.convert("RGB"),
                            dtype=np.uint8,
                        )
                except Exception as e:
                    raise FormalA0CError(
                        f"Cannot read image: {image_path}: {e}"
                    ) from e

                corrupted = apply_corruption(
                    image_np=image_np,
                    corruption_name=corruption_name,
                    severity=severity,
                    corrupt_fn=corrupt_fn,
                )
                arrays.append(corrupted)
                ids.append(row["image_id"])

            serialized, success_batch = predict_chunk_with_backoff(
                model=model,
                arrays=arrays,
                ids=ids,
                rows=rows,
                imgsz=imgsz,
                conf=conf,
                iou_nms=iou_nms,
                max_det=max_det,
                device=device,
                torch_mod=torch_mod,
            )
            max_success_batch = max(
                max_success_batch, success_batch
            )

            for obj in serialized:
                fout.write(
                    json.dumps(
                        obj,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_preds += len(obj["predictions"])

            processed += len(rows)

            del arrays
            del serialized
            gc.collect()
            if processed % 256 == 0 or processed == len(
                image_manifest
            ):
                append_log(
                    progress_log,
                    f"{corruption_name} s{severity}: "
                    f"{processed}/{len(image_manifest)}",
                )

    with temp_jsonl.open("rb") as src, gzip.open(
        cache_path,
        "wb",
        compresslevel=6,
    ) as dst:
        shutil.copyfileobj(src, dst)
    temp_jsonl.unlink(missing_ok=True)

    elapsed = time.time() - start

    meta = {
        "status": "COMPLETE",
        "protocol": protocol,
        "images": processed,
        "predictions": total_preds,
        "requested_batch": requested_batch,
        "largest_successful_batch": max_success_batch,
        "wall_seconds": elapsed,
        "wall_minutes": elapsed / 60.0,
        "completed": now_iso(),
    }
    write_json(meta_path, meta)

    append_log(
        progress_log,
        f"DONE {corruption_name} severity={severity} "
        f"time={elapsed/60.0:.2f} min "
        f"preds={total_preds}",
    )
    return meta


def load_prediction_cache(
    cache_path: Path,
    expected_images: int,
) -> Tuple[List[Pred], dict]:
    preds: List[Pred] = []
    seen = set()
    class_counts = Counter()

    with gzip.open(
        cache_path,
        "rt",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            image_id = obj["image_id"]
            if image_id in seen:
                die(
                    f"Duplicate image in prediction cache: "
                    f"{image_id}"
                )
            seen.add(image_id)

            for p in obj.get("predictions", []):
                cls_id = int(p["cls_id"])
                preds.append(
                    Pred(
                        image_id=image_id,
                        cls_id=cls_id,
                        conf=float(p["conf"]),
                        xyxy=np.asarray(
                            p["xyxy"],
                            dtype=np.float64,
                        ),
                        area640=float(p["area640"]),
                    )
                )
                class_counts[cls_id] += 1

    if len(seen) != expected_images:
        die(
            f"Prediction cache image count mismatch: "
            f"{len(seen)} != {expected_images}"
        )

    return preds, {
        "images": len(seen),
        "predictions": len(preds),
        "class_predictions": dict(class_counts),
    }


def evaluate_condition(
    gts: List[GT],
    preds: List[Pred],
    corruption_name: str,
    severity: int,
    protocol: dict,
) -> dict:
    class_rows: List[dict] = []

    for bin_name in ("ALL", "ES", "S", "M", "L"):
        for cls_id in (0, 1):
            row = eval_class_bin(
                gts=gts,
                preds=preds,
                cls_id=cls_id,
                bin_name=bin_name,
            )
            row["corruption"] = corruption_name
            row["severity"] = severity
            class_rows.append(row)

    rows = list(class_rows)
    for bin_name in ("ALL", "ES", "S", "M", "L"):
        r = aggregate_macro(
            class_rows,
            bin_name,
        )
        r["corruption"] = corruption_name
        r["severity"] = severity
        rows.append(r)

    return {
        "status": "PASS",
        "protocol": protocol,
        "corruption": corruption_name,
        "severity": severity,
        "metrics": rows,
    }


def load_condition_metrics(
    metrics_path: Path,
) -> List[dict]:
    obj = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )
    if obj.get("status") != "PASS":
        die(f"Condition metrics not PASS: {metrics_path}")
    return obj["metrics"]


def find_metric(
    rows: List[dict],
    size_bin: str,
    class_name: str,
) -> dict:
    matches = [
        r
        for r in rows
        if r["size_bin"] == size_bin
        and r["class_name"] == class_name
    ]
    if len(matches) != 1:
        die(
            f"Expected exactly one metric row for "
            f"{size_bin}/{class_name}, got {len(matches)}"
        )
    return matches[0]


def aggregate_all_conditions(
    condition_metric_rows: List[dict],
    clean: Dict[Tuple[str, str], dict],
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """
    Returns:
      condition_key_rows
      per_corruption_rows
      severity_rows
      global_summary
    """
    condition_key_rows: List[dict] = []

    by_condition: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for row in condition_metric_rows:
        by_condition[
            (row["corruption"], int(row["severity"]))
        ].append(row)

    for (corr, sev), rows in sorted(by_condition.items()):
        for size_bin, class_name in KEY_VIEWS:
            r = find_metric(rows, size_bin, class_name)
            clean_r = clean[(size_bin, class_name)]
            clean_ap = float(clean_r["AP50_95"])
            ap = float(r["AP50_95"])
            condition_key_rows.append(
                {
                    "corruption": corr,
                    "severity": sev,
                    "size_bin": size_bin,
                    "class_name": class_name,
                    "AP50": float(r["AP50"]),
                    "AP50_95": ap,
                    "R50_max": float(r["R50_max"]),
                    "R50_at_conf_025": float(
                        r["R50_at_conf_025"]
                    ),
                    "clean_AP50_95": clean_ap,
                    "absolute_drop_AP50_95": clean_ap - ap,
                    "retention_ratio": (
                        ap / clean_ap
                        if clean_ap > 0
                        else float("nan")
                    ),
                }
            )

    # Per-corruption mean over five severities.
    per_corruption_rows: List[dict] = []
    for corr in CORRUPTIONS:
        corr_rows = [
            r
            for r in condition_key_rows
            if r["corruption"] == corr
        ]
        if not corr_rows:
            continue
        for size_bin, class_name in KEY_VIEWS:
            sub = [
                r
                for r in corr_rows
                if r["size_bin"] == size_bin
                and r["class_name"] == class_name
            ]
            if not sub:
                continue
            clean_ap = float(
                clean[(size_bin, class_name)]["AP50_95"]
            )
            mean_ap = float(
                np.mean([float(r["AP50_95"]) for r in sub])
            )
            per_corruption_rows.append(
                {
                    "corruption": corr,
                    "size_bin": size_bin,
                    "class_name": class_name,
                    "mean_AP50_95_over_severity": mean_ap,
                    "clean_AP50_95": clean_ap,
                    "absolute_drop": clean_ap - mean_ap,
                    "rPC_corruption": (
                        mean_ap / clean_ap
                        if clean_ap > 0
                        else float("nan")
                    ),
                    "severity_conditions": len(sub),
                }
            )

    # Mean across all 15 corruptions at each severity.
    severity_rows: List[dict] = []
    for sev in SEVERITIES:
        sev_rows = [
            r
            for r in condition_key_rows
            if int(r["severity"]) == sev
        ]
        for size_bin, class_name in KEY_VIEWS:
            sub = [
                r
                for r in sev_rows
                if r["size_bin"] == size_bin
                and r["class_name"] == class_name
            ]
            if not sub:
                continue
            clean_ap = float(
                clean[(size_bin, class_name)]["AP50_95"]
            )
            mean_ap = float(
                np.mean([float(r["AP50_95"]) for r in sub])
            )
            severity_rows.append(
                {
                    "severity": sev,
                    "size_bin": size_bin,
                    "class_name": class_name,
                    "mean_AP50_95_across_corruptions": mean_ap,
                    "clean_AP50_95": clean_ap,
                    "absolute_drop": clean_ap - mean_ap,
                    "retention_ratio": (
                        mean_ap / clean_ap
                        if clean_ap > 0
                        else float("nan")
                    ),
                    "corruption_conditions": len(sub),
                }
            )

    # Global mPC15 over all 75 conditions.
    global_views = {}
    for size_bin, class_name in KEY_VIEWS:
        sub = [
            r
            for r in condition_key_rows
            if r["size_bin"] == size_bin
            and r["class_name"] == class_name
        ]
        if not sub:
            continue
        clean_ap = float(
            clean[(size_bin, class_name)]["AP50_95"]
        )
        mpc = float(
            np.mean([float(r["AP50_95"]) for r in sub])
        )

        sub14 = [
            r
            for r in sub
            if r["corruption"] != "elastic_transform"
        ]
        mpc14 = float(
            np.mean([float(r["AP50_95"]) for r in sub14])
        ) if sub14 else float("nan")

        global_views[f"{size_bin}/{class_name}"] = {
            "clean_AP50_95": clean_ap,
            "mPC15_AP50_95": mpc,
            "rPC15": (
                mpc / clean_ap
                if clean_ap > 0
                else float("nan")
            ),
            "absolute_drop15": clean_ap - mpc,
            "conditions15": len(sub),
            "mPC14_no_elastic_AP50_95": mpc14,
            "rPC14_no_elastic": (
                mpc14 / clean_ap
                if clean_ap > 0
                else float("nan")
            ),
            "conditions14": len(sub14),
        }

    # Rank harmful corruptions for key research views.
    rankings = {}
    for size_bin, class_name in (
        ("ALL", "all_macro"),
        ("ES", "all_macro"),
        ("ES", "hat"),
        ("ES", "person"),
        ("S", "person"),
    ):
        sub = [
            r
            for r in per_corruption_rows
            if r["size_bin"] == size_bin
            and r["class_name"] == class_name
        ]
        sub.sort(
            key=lambda r: float(r["absolute_drop"]),
            reverse=True,
        )
        rankings[f"{size_bin}/{class_name}"] = [
            {
                "corruption": r["corruption"],
                "mean_AP50_95": r[
                    "mean_AP50_95_over_severity"
                ],
                "absolute_drop": r["absolute_drop"],
                "rPC_corruption": r["rPC_corruption"],
            }
            for r in sub
        ]

    global_summary = {
        "views": global_views,
        "harmfulness_rankings": rankings,
    }
    return (
        condition_key_rows,
        per_corruption_rows,
        severity_rows,
        global_summary,
    )


def make_figures(
    condition_key_rows: List[dict],
    per_corruption_rows: List[dict],
    severity_rows: List[dict],
    results_root: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"WARNING: matplotlib unavailable: {e}")
        return []

    created = []

    # 1) Overall AP50:95 heatmap: 15 corruptions x 5 severity
    mat = np.full(
        (len(CORRUPTIONS), len(SEVERITIES)),
        np.nan,
        dtype=float,
    )
    for i, corr in enumerate(CORRUPTIONS):
        for j, sev in enumerate(SEVERITIES):
            matches = [
                r
                for r in condition_key_rows
                if r["corruption"] == corr
                and int(r["severity"]) == sev
                and r["size_bin"] == "ALL"
                and r["class_name"] == "all_macro"
            ]
            if matches:
                mat[i, j] = float(matches[0]["AP50_95"])

    fig, ax = plt.subplots(figsize=(8.5, 8.2))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(SEVERITIES)))
    ax.set_xticklabels([str(s) for s in SEVERITIES])
    ax.set_yticks(np.arange(len(CORRUPTIONS)))
    ax.set_yticklabels(CORRUPTIONS)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Corruption")
    ax.set_title("Helmet-C: overall AP@0.5:0.95")
    fig.colorbar(im, ax=ax, label="AP@0.5:0.95")
    fig.tight_layout()
    p = results_root / "fig_A0C_heatmap_overall_AP5095.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    created.append(str(p))

    # 2) ES-person heatmap.
    mat = np.full(
        (len(CORRUPTIONS), len(SEVERITIES)),
        np.nan,
        dtype=float,
    )
    for i, corr in enumerate(CORRUPTIONS):
        for j, sev in enumerate(SEVERITIES):
            matches = [
                r
                for r in condition_key_rows
                if r["corruption"] == corr
                and int(r["severity"]) == sev
                and r["size_bin"] == "ES"
                and r["class_name"] == "person"
            ]
            if matches:
                mat[i, j] = float(matches[0]["AP50_95"])

    fig, ax = plt.subplots(figsize=(8.5, 8.2))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(SEVERITIES)))
    ax.set_xticklabels([str(s) for s in SEVERITIES])
    ax.set_yticks(np.arange(len(CORRUPTIONS)))
    ax.set_yticklabels(CORRUPTIONS)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Corruption")
    ax.set_title("Helmet-C: ES person AP@0.5:0.95")
    fig.colorbar(im, ax=ax, label="AP@0.5:0.95")
    fig.tight_layout()
    p = results_root / "fig_A0C_heatmap_ES_person_AP5095.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    created.append(str(p))

    # 3) Severity profile, one figure only.
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for size_bin, class_name, label in (
        ("ALL", "all_macro", "ALL"),
        ("ES", "all_macro", "ES"),
        ("ES", "hat", "ES-hat"),
        ("ES", "person", "ES-person"),
    ):
        sub = [
            r
            for r in severity_rows
            if r["size_bin"] == size_bin
            and r["class_name"] == class_name
        ]
        sub.sort(key=lambda r: int(r["severity"]))
        if sub:
            ax.plot(
                [int(r["severity"]) for r in sub],
                [
                    float(
                        r["mean_AP50_95_across_corruptions"]
                    )
                    for r in sub
                ],
                marker="o",
                label=label,
            )
    ax.set_xticks(list(SEVERITIES))
    ax.set_xlabel("Severity")
    ax.set_ylabel("Mean AP@0.5:0.95")
    ax.set_title("Helmet-C severity profile")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    p = results_root / "fig_A0C_severity_profile.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    created.append(str(p))

    return created


def make_upload_zip(
    zip_path: Path,
    results_root: Path,
    include_cache: bool,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.unlink(missing_ok=True)

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for p in sorted(results_root.rglob("*")):
            if not p.is_file():
                continue

            if (
                not include_cache
                and "prediction_cache" in p.parts
                and p.suffix == ".gz"
            ):
                continue

            arc = Path("A0C_helmet_c") / p.relative_to(
                results_root
            )
            zf.write(p, arcname=str(arc))


def self_test() -> int:
    gts = [
        GT(
            "im1",
            0,
            np.array([0, 0, 10, 10], dtype=float),
            100.0,
        ),
        GT(
            "im1",
            0,
            np.array([100, 100, 200, 200], dtype=float),
            10000.0,
        ),
    ]
    preds = [
        Pred(
            "im1",
            0,
            0.99,
            np.array([0, 0, 10, 10], dtype=float),
            100.0,
        ),
        Pred(
            "im1",
            0,
            0.95,
            np.array([100, 100, 200, 200], dtype=float),
            10000.0,
        ),
        Pred(
            "im1",
            0,
            0.50,
            np.array([30, 30, 40, 40], dtype=float),
            100.0,
        ),
    ]
    r = eval_class_bin_at_iou(
        gts,
        preds,
        cls_id=0,
        bin_name="ES",
        iou_thr=0.5,
    )
    assert r["n_gt"] == 1
    assert r["tp_total"] == 1
    assert r["fp_total"] == 1
    assert r["n_ignored_preds"] == 1
    assert abs(r["recall_final"] - 1.0) < 1e-12
    assert r["ap"] > 0.99
    print("A0-C evaluator self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()

    if args.self_test:
        return self_test()

    invalid_force = [
        c for c in args.force_corruption
        if c not in CORRUPTIONS
    ]
    invalid_only = [
        c for c in args.only_corruption
        if c not in CORRUPTIONS
    ]
    invalid_severity = [
        s for s in args.only_severity
        if s not in SEVERITIES
    ]
    if invalid_force or invalid_only or invalid_severity:
        die(
            f"Invalid selection. force={invalid_force}, "
            f"only={invalid_only}, severity={invalid_severity}"
        )

    selected_corruptions = (
        tuple(args.only_corruption)
        if args.only_corruption
        else CORRUPTIONS
    )
    selected_severities = (
        tuple(sorted(set(args.only_severity)))
        if args.only_severity
        else SEVERITIES
    )

    results_root = args.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    progress_log = results_root / "A0C_PROGRESS.log"

    (
        torch,
        ultralytics,
        YOLO,
        Image,
        imagecorruptions,
        corrupt_fn,
    ) = import_runtime()

    if not torch.cuda.is_available():
        die(
            "CUDA GPU is unavailable. Run A0-C on the "
            "same formal GPU instance."
        )

    if not args.best.is_file():
        die(f"A0 best.pt not found: {args.best}")
    if not args.data.is_file():
        die(f"Formal data.yaml not found: {args.data}")

    best_sha = sha256_file(args.best)
    data_sha = sha256_file(args.data)

    if best_sha != EXPECTED_BEST_SHA256:
        die(
            "A0 best.pt SHA256 mismatch.\n"
            f"Expected: {EXPECTED_BEST_SHA256}\n"
            f"Actual  : {best_sha}"
        )

    validate_a0s_protocol(
        args.a0s_protocol,
        best_sha,
    )
    clean = load_clean_a0s_metrics(
        args.a0s_metrics
    )

    ic_version = getattr(
        imagecorruptions,
        "__version__",
        "unknown",
    )

    print("=" * 92)
    print("NEW PAPER V1.0 — A0-C HELMET-C 15x5 ROBUSTNESS BASELINE")
    print("=" * 92)
    print(f"Timestamp            : {now_iso()}")
    print(f"best.pt              : {args.best.resolve()}")
    print(f"best SHA256          : {best_sha}")
    print(f"data.yaml            : {args.data.resolve()}")
    print(f"imgsz                : {args.imgsz}")
    print(f"requested batch      : {args.batch}")
    print(f"conf / NMS IoU       : {args.conf} / {args.iou_nms}")
    print(f"max_det              : {args.max_det}")
    print(f"Ultralytics          : {getattr(ultralytics, '__version__', 'unknown')}")
    print(f"PyTorch              : {getattr(torch, '__version__', 'unknown')}")
    print(f"NumPy                : {np.__version__}")
    print(f"imagecorruptions     : {ic_version}")
    print(
        f"Selected conditions   : "
        f"{len(selected_corruptions)} corruptions x "
        f"{len(selected_severities)} severity"
    )
    print("=" * 92)

    protocol = {
        "protocol_name": "V1.0_A0C_HELMET_C_BASELINE_FORMAL",
        "script_version": SCRIPT_VERSION,
        "created": now_iso(),
        "checkpoint": {
            "path": str(args.best.resolve()),
            "sha256": best_sha,
        },
        "data": {
            "path": str(args.data.resolve()),
            "sha256": data_sha,
            "test_images": EXPECTED_TEST_IMAGES,
            "test_class_objects": EXPECTED_TEST_CLASS_OBJECTS,
        },
        "clean_reference": {
            "a0s_metrics": str(args.a0s_metrics.resolve()),
            "a0s_protocol": str(args.a0s_protocol.resolve()),
            "note": (
                "rPC uses A0-S clean metrics from the same "
                "frozen custom size-aware evaluator."
            ),
        },
        "corruptions": list(CORRUPTIONS),
        "severities": list(SEVERITIES),
        "corruption_backend": {
            "package": "imagecorruptions",
            "version": ic_version,
        },
        "inference": {
            "imgsz": args.imgsz,
            "requested_batch": args.batch,
            "cuda_oom_auto_backoff": True,
            "conf": args.conf,
            "iou_nms": args.iou_nms,
            "max_det": args.max_det,
            "device": args.device,
        },
        "size_bins_area_at_imgsz": {
            "ES": "[0, 16^2)",
            "S": "[16^2, 32^2)",
            "M": "[32^2, 96^2)",
            "L": "[96^2, +inf)",
        },
        "metrics": {
            "mPC15": "mean AP50:95 over 15 corruptions x 5 severity",
            "rPC15": "mPC15 / clean AP50:95",
            "mPC14_no_elastic": (
                "same average excluding elastic_transform"
            ),
        },
    }
    write_json(
        results_root / "A0C_PROTOCOL_LOCK.json",
        protocol,
    )

    environment = {
        "timestamp": now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "ultralytics": getattr(
            ultralytics, "__version__", None
        ),
        "imagecorruptions": ic_version,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    write_json(
        results_root / "A0C_environment.json",
        environment,
    )

    append_log(progress_log, "A0-C START")

    print("\n[1/5] Parsing formal test GT ...")
    gts, image_manifest, size_rows = parse_formal_test(
        args.data,
        args.imgsz,
        Image,
    )
    write_csv(
        results_root / "A0C_test_image_manifest.csv",
        [
            "image_id",
            "image_path",
            "width",
            "height",
            "resize_scale_to_640",
            "gt_objects",
        ],
        image_manifest,
    )
    write_csv(
        results_root / "A0C_gt_size_distribution.csv",
        [
            "size_bin",
            "class_id",
            "class_name",
            "gt_count",
        ],
        size_rows,
    )

    print("\n[2/5] Loading A0 model ...")
    model = YOLO(str(args.best.resolve()))

    all_condition_metrics: List[dict] = []
    condition_status_rows: List[dict] = []

    total_selected = (
        len(selected_corruptions)
        * len(selected_severities)
    )
    condition_index = 0

    print("\n[3/5] Running Helmet-C conditions ...")
    for corr in selected_corruptions:
        for severity in selected_severities:
            condition_index += 1

            cache_path, meta_path, metrics_path = (
                condition_cache_paths(
                    results_root,
                    corr,
                    severity,
                )
            )

            cond_protocol = condition_protocol(
                best_sha=best_sha,
                data_sha=data_sha,
                corruption_name=corr,
                severity=severity,
                imgsz=args.imgsz,
                conf=args.conf,
                iou_nms=args.iou_nms,
                max_det=args.max_det,
                imagecorruptions_version=ic_version,
            )

            force = corr in set(args.force_corruption)
            valid_cache = valid_condition_cache(
                cache_path,
                meta_path,
                metrics_path,
                cond_protocol,
            )

            print(
                f"\n--- [{condition_index}/{total_selected}] "
                f"{corr} severity={severity} ---"
            )

            if valid_cache and not force:
                append_log(
                    progress_log,
                    f"REUSE {corr} severity={severity}",
                )
                metrics_rows = load_condition_metrics(
                    metrics_path
                )
                all_condition_metrics.extend(metrics_rows)
                condition_status_rows.append(
                    {
                        "corruption": corr,
                        "severity": severity,
                        "status": "REUSED",
                        "cache": str(cache_path),
                        "metrics": str(metrics_path),
                    }
                )
                continue

            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            metrics_path.unlink(missing_ok=True)

            meta = generate_condition_predictions(
                model=model,
                image_manifest=image_manifest,
                cache_path=cache_path,
                meta_path=meta_path,
                protocol=cond_protocol,
                corruption_name=corr,
                severity=severity,
                requested_batch=max(1, int(args.batch)),
                imgsz=args.imgsz,
                conf=args.conf,
                iou_nms=args.iou_nms,
                max_det=args.max_det,
                device=args.device,
                torch_mod=torch,
                Image=Image,
                corrupt_fn=corrupt_fn,
                progress_log=progress_log,
            )

            preds, pred_stats = load_prediction_cache(
                cache_path,
                EXPECTED_TEST_IMAGES,
            )

            append_log(
                progress_log,
                f"EVAL {corr} severity={severity}",
            )
            condition_result = evaluate_condition(
                gts=gts,
                preds=preds,
                corruption_name=corr,
                severity=severity,
                protocol=cond_protocol,
            )
            condition_result["prediction_stats"] = pred_stats
            condition_result["inference_meta"] = meta
            write_json(
                metrics_path,
                condition_result,
            )

            metrics_rows = condition_result["metrics"]
            all_condition_metrics.extend(metrics_rows)

            condition_status_rows.append(
                {
                    "corruption": corr,
                    "severity": severity,
                    "status": "COMPUTED",
                    "cache": str(cache_path),
                    "metrics": str(metrics_path),
                }
            )

            del preds
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    write_csv(
        results_root / "A0C_condition_status.csv",
        [
            "corruption",
            "severity",
            "status",
            "cache",
            "metrics",
        ],
        condition_status_rows,
    )

    # If running only a subset for debugging, finish without pretending
    # it is the formal 75-condition mPC.
    full_protocol_complete = (
        set(selected_corruptions) == set(CORRUPTIONS)
        and set(selected_severities) == set(SEVERITIES)
        and len({
            (r["corruption"], int(r["severity"]))
            for r in all_condition_metrics
        }) == 75
    )

    print("\n[4/5] Aggregating robustness metrics ...")
    if full_protocol_complete:
        (
            condition_key_rows,
            per_corruption_rows,
            severity_rows,
            global_summary,
        ) = aggregate_all_conditions(
            all_condition_metrics,
            clean,
        )

        write_csv(
            results_root / "A0C_condition_key_metrics.csv",
            [
                "corruption",
                "severity",
                "size_bin",
                "class_name",
                "AP50",
                "AP50_95",
                "R50_max",
                "R50_at_conf_025",
                "clean_AP50_95",
                "absolute_drop_AP50_95",
                "retention_ratio",
            ],
            condition_key_rows,
        )

        write_csv(
            results_root / "A0C_per_corruption_summary.csv",
            [
                "corruption",
                "size_bin",
                "class_name",
                "mean_AP50_95_over_severity",
                "clean_AP50_95",
                "absolute_drop",
                "rPC_corruption",
                "severity_conditions",
            ],
            per_corruption_rows,
        )

        write_csv(
            results_root / "A0C_severity_summary.csv",
            [
                "severity",
                "size_bin",
                "class_name",
                "mean_AP50_95_across_corruptions",
                "clean_AP50_95",
                "absolute_drop",
                "retention_ratio",
                "corruption_conditions",
            ],
            severity_rows,
        )

        # Full 15 rows per condition (ALL/ES/S/M/L × 3 class views).
        write_csv(
            results_root / "A0C_all_condition_metrics.csv",
            [
                "corruption",
                "severity",
                "size_bin",
                "class_id",
                "class_name",
                "gt_count",
                "AP50",
                "AP50_95",
                "R50_max",
                "R50_at_conf_025",
                "eval_predictions_at_iou50",
                "ignored_predictions_at_iou50",
            ],
            all_condition_metrics,
        )

        summary = {
            "experiment": "A0C_HELMET_C_BASELINE",
            "status": "PASS",
            "completed": now_iso(),
            "conditions": 75,
            "checkpoint_sha256": best_sha,
            "global": global_summary,
        }
        write_json(
            results_root / "A0C_summary.json",
            summary,
        )

        figure_paths = make_figures(
            condition_key_rows,
            per_corruption_rows,
            severity_rows,
            results_root,
        )

        all_view = global_summary["views"][
            "ALL/all_macro"
        ]
        es_all = global_summary["views"][
            "ES/all_macro"
        ]
        es_hat = global_summary["views"][
            "ES/hat"
        ]
        es_person = global_summary["views"][
            "ES/person"
        ]
        s_person = global_summary["views"][
            "S/person"
        ]

        top_es_person = global_summary[
            "harmfulness_rankings"
        ]["ES/person"][:5]
        top_es_hat = global_summary[
            "harmfulness_rankings"
        ]["ES/hat"][:5]

        report_lines = [
            "NEW PAPER V1.0 — A0-C HELMET-C ROBUSTNESS BASELINE",
            "=" * 82,
            "STATUS                     : PASS",
            f"completed                  : {summary['completed']}",
            f"conditions                 : 75",
            f"best.pt SHA256             : {best_sha}",
            "",
            "ALL / all-macro:",
            f"clean AP50:95              : {all_view['clean_AP50_95']}",
            f"mPC15 AP50:95              : {all_view['mPC15_AP50_95']}",
            f"rPC15                      : {all_view['rPC15']}",
            f"absolute drop              : {all_view['absolute_drop15']}",
            f"mPC14 no elastic           : {all_view['mPC14_no_elastic_AP50_95']}",
            "",
            "ES / all-macro:",
            f"clean AP50:95              : {es_all['clean_AP50_95']}",
            f"mPC15 AP50:95              : {es_all['mPC15_AP50_95']}",
            f"rPC15                      : {es_all['rPC15']}",
            f"absolute drop              : {es_all['absolute_drop15']}",
            "",
            "ES / hat:",
            f"clean AP50:95              : {es_hat['clean_AP50_95']}",
            f"mPC15 AP50:95              : {es_hat['mPC15_AP50_95']}",
            f"rPC15                      : {es_hat['rPC15']}",
            f"absolute drop              : {es_hat['absolute_drop15']}",
            "",
            "ES / person:",
            f"clean AP50:95              : {es_person['clean_AP50_95']}",
            f"mPC15 AP50:95              : {es_person['mPC15_AP50_95']}",
            f"rPC15                      : {es_person['rPC15']}",
            f"absolute drop              : {es_person['absolute_drop15']}",
            "",
            "S / person:",
            f"clean AP50:95              : {s_person['clean_AP50_95']}",
            f"mPC15 AP50:95              : {s_person['mPC15_AP50_95']}",
            f"rPC15                      : {s_person['rPC15']}",
            f"absolute drop              : {s_person['absolute_drop15']}",
            "",
            "Top-5 harmful corruptions for ES-person:",
        ]
        for rank, r in enumerate(top_es_person, start=1):
            report_lines.append(
                f"{rank}. {r['corruption']}: "
                f"meanAP={r['mean_AP50_95']:.6f}, "
                f"drop={r['absolute_drop']:.6f}, "
                f"rPC={r['rPC_corruption']:.6f}"
            )

        report_lines += [
            "",
            "Top-5 harmful corruptions for ES-hat:",
        ]
        for rank, r in enumerate(top_es_hat, start=1):
            report_lines.append(
                f"{rank}. {r['corruption']}: "
                f"meanAP={r['mean_AP50_95']:.6f}, "
                f"drop={r['absolute_drop']:.6f}, "
                f"rPC={r['rPC_corruption']:.6f}"
            )

        report_lines += [
            "",
            "Important:",
            "- rPC uses the A0-S clean result from the identical custom evaluator.",
            "- elastic_transform is included in formal mPC15 for standard benchmark compatibility.",
            "- mPC14_no_elastic is also reported as a geometry-label sensitivity check.",
            "- Prediction caches are preserved server-side for resume/re-audit.",
        ]

        (
            results_root / "A0C_REPORT.txt"
        ).write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

    else:
        summary = {
            "experiment": "A0C_HELMET_C_BASELINE",
            "status": "PARTIAL_DEBUG_RUN",
            "completed": now_iso(),
            "selected_corruptions": list(
                selected_corruptions
            ),
            "selected_severities": list(
                selected_severities
            ),
            "condition_metric_rows": len(
                all_condition_metrics
            ),
        }
        write_json(
            results_root / "A0C_partial_summary.json",
            summary,
        )
        append_log(
            progress_log,
            "PARTIAL DEBUG RUN COMPLETE; "
            "formal mPC15 not calculated",
        )

    print("\n[5/5] Creating upload package ...")
    if not args.no_zip:
        make_upload_zip(
            args.zip_path.resolve(),
            results_root,
            include_cache=args.include_cache_in_zip,
        )

    append_log(
        progress_log,
        "A0-C COMPLETE "
        + ("PASS" if full_protocol_complete else "PARTIAL"),
    )

    print("\n" + "=" * 92)
    print("A0-C HELMET-C BASELINE COMPLETE")
    if full_protocol_complete:
        print("Status                  : PASS")
        all_view = global_summary["views"][
            "ALL/all_macro"
        ]
        es_person = global_summary["views"][
            "ES/person"
        ]
        es_hat = global_summary["views"][
            "ES/hat"
        ]
        print(
            f"ALL mPC15 AP50:95       : "
            f"{all_view['mPC15_AP50_95']}"
        )
        print(
            f"ALL rPC15               : "
            f"{all_view['rPC15']}"
        )
        print(
            f"ES-person mPC15         : "
            f"{es_person['mPC15_AP50_95']}"
        )
        print(
            f"ES-person rPC15         : "
            f"{es_person['rPC15']}"
        )
        print(
            f"ES-hat mPC15            : "
            f"{es_hat['mPC15_AP50_95']}"
        )
        print(
            f"ES-hat rPC15            : "
            f"{es_hat['rPC15']}"
        )
    else:
        print("Status                  : PARTIAL_DEBUG_RUN")

    print(f"Results                 : {results_root}")
    if not args.no_zip:
        print(f"Upload package          : {args.zip_path.resolve()}")
    print("=" * 92)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FormalA0CError as e:
        print("\n" + "=" * 92, file=sys.stderr)
        print("A0-C FORMAL BASELINE FAILED", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("=" * 92, file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
