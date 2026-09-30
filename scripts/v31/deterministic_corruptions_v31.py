#!/usr/bin/env python3
"""Deterministic Helmet-C corruption worker used by public V3.1 reproduction.

This clean public version mirrors the final V3.1 rules:
- stable per-image uint32 seed from SHA256(base|image_id|corruption|severity)
- explicit RNG for impulse_noise
- imagecorruptions==1.1.2 for the remaining corruption families
- compatibility shim for modern scikit-image gaussian(channel_axis=...)
"""
from __future__ import annotations

import hashlib
import inspect
import random
from pathlib import Path

import numpy as np

PROTOCOL_ID = "helmet-c-val-v3.1-deterministic-impulse-audited-glass-2026-09-29"
BASE_SEED = 3407
IMPULSE_C = (0.03, 0.06, 0.09, 0.17, 0.27)
_CORRUPT_FN = None


def image_seed(image_id: str, corruption: str, severity: int, base: int = BASE_SEED) -> int:
    payload = f"{int(base)}|{image_id}|{corruption}|{int(severity)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def _install_imagecorruptions_compat():
    """Install the narrow imagecorruptions/skimage compatibility shim."""
    for name, obj in {"int": int, "float": float, "bool": bool, "object": object}.items():
        if name not in np.__dict__:
            setattr(np, name, obj)

    from imagecorruptions import corrupt
    import imagecorruptions.corruptions as ic_corruptions
    from skimage.filters import gaussian as skimage_gaussian

    params = set(inspect.signature(skimage_gaussian).parameters)

    def gaussian_compat(
        image,
        sigma=1,
        output=None,
        mode="nearest",
        cval=0,
        multichannel=None,
        preserve_range=False,
        truncate=4.0,
        channel_axis=None,
        **kwargs,
    ):
        if multichannel is not None and channel_axis is None:
            channel_axis = -1 if bool(multichannel) else None
        candidates = {
            "sigma": sigma,
            "mode": mode,
            "cval": cval,
            "preserve_range": preserve_range,
            "truncate": truncate,
        }
        call_kwargs = {k: v for k, v in candidates.items() if k in params}
        if "channel_axis" in params:
            call_kwargs["channel_axis"] = channel_axis
        elif "multichannel" in params and multichannel is not None:
            call_kwargs["multichannel"] = bool(multichannel)
        if output is not None and "output" in params:
            call_kwargs["output"] = output
        for key, value in kwargs.items():
            if key in params:
                call_kwargs[key] = value
        return skimage_gaussian(image, **call_kwargs)

    ic_corruptions.gaussian = gaussian_compat
    return corrupt


def _impulse_noise(rgb: np.ndarray, severity: int, seed: int) -> np.ndarray:
    from skimage.util import random_noise

    amount = IMPULSE_C[int(severity) - 1]
    x = np.asarray(rgb, dtype=np.uint8) / 255.0
    params = inspect.signature(random_noise).parameters
    kwargs = {"mode": "s&p", "amount": amount}

    if "rng" in params:
        kwargs["rng"] = np.random.default_rng(np.uint32(seed))
    elif "seed" in params:
        kwargs["seed"] = int(np.uint32(seed))
    else:
        raise RuntimeError(
            "skimage.util.random_noise exposes neither rng nor seed; "
            "deterministic impulse_noise cannot be guaranteed"
        )

    out = random_noise(x, **kwargs)
    return np.clip(out, 0.0, 1.0) * 255.0


def corrupt_rgb(rgb: np.ndarray, image_id: str, corruption: str, severity: int) -> np.ndarray:
    """Generate one deterministic RGB uint8 corruption realization."""
    global _CORRUPT_FN
    severity = int(severity)
    seed = image_seed(image_id, corruption, severity)

    np.random.seed(seed)
    random.seed(seed)

    if corruption == "impulse_noise":
        out = _impulse_noise(rgb, severity, seed)
    else:
        if _CORRUPT_FN is None:
            _CORRUPT_FN = _install_imagecorruptions_compat()
        out = _CORRUPT_FN(
            np.asarray(rgb, dtype=np.uint8),
            corruption_name=corruption,
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


def corrupt_file(path: Path, corruption: str, severity: int) -> tuple[str, np.ndarray]:
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return path.stem, corrupt_rgb(rgb, path.stem, corruption, severity)


def array_digest(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()
