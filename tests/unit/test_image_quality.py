import numpy as np

from domain.face import BoundingBox
from infrastructure.cv.quality import assess_quality

_FULL = BoundingBox(width=1.0, height=1.0, left=0.0, top=0.0)


def test_bright_vs_dark():
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    bright = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert assess_quality(dark, _FULL).brightness < 5
    assert assess_quality(bright, _FULL).brightness > 95


def test_sharp_vs_blurred():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    import cv2
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=6)
    assert assess_quality(sharp, _FULL).sharpness > assess_quality(blurred, _FULL).sharpness
