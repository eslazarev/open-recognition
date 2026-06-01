"""Brightness + sharpness of the face crop. Heuristic 0..100 scale — NOT
numerically equal to AWS Quality, just monotonic in the same direction.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from domain.face import BoundingBox
from domain.face_attributes import ImageQuality

_SHARP_FULL_SCALE = 500.0


def assess_quality(image: NDArray[np.uint8], bbox: BoundingBox) -> ImageQuality:
    h, w = image.shape[:2]
    x0 = max(0, int(bbox.left * w))
    y0 = max(0, int(bbox.top * h))
    x1 = min(w, int((bbox.left + bbox.width) * w))
    y1 = min(h, int((bbox.top + bbox.height) * h))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return ImageQuality(brightness=0.0, sharpness=0.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()) / 255.0 * 100.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(100.0, lap_var / _SHARP_FULL_SCALE * 100.0)
    return ImageQuality(brightness=brightness, sharpness=sharpness)
