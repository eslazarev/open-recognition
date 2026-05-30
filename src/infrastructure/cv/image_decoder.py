"""Decode AWS-style `Image.Bytes` into a numpy BGR array for OpenCV.

The Rekognition spec allows JPEG or PNG up to ~5 MB inline. We enforce
the same limit and surface format problems as domain errors so that
the wire layer translates them into AWS error envelopes.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from domain.errors import ImageTooLargeError, InvalidImageFormatError

MAX_BYTES = 5 * 1024 * 1024  # 5 MiB inline cap, matches AWS Rekognition
_ALLOWED_MODES = {"RGB", "L"}


def decode_image(blob: bytes) -> NDArray[np.uint8]:
    if len(blob) > MAX_BYTES:
        raise ImageTooLargeError(
            f"Image exceeds {MAX_BYTES} bytes (got {len(blob)})"
        )
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageFormatError("Image format not recognised") from exc

    if img.format not in {"JPEG", "PNG"}:
        raise InvalidImageFormatError(
            f"Only JPEG and PNG are supported, got {img.format}"
        )
    if img.mode not in _ALLOWED_MODES:
        img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale → 3-channel
        arr = np.stack([arr] * 3, axis=-1)
    # PIL gives RGB; OpenCV expects BGR.
    return arr[:, :, ::-1].copy()
