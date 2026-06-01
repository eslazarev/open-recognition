"""opencv_zoo facial expression recognition via cv2.dnn (pooled).

Preprocessing mirrors the opencv_zoo reference: align to 112x112 (done by the
caller via SFace alignCrop), BGR->RGB, /255, subtract mean .5, divide std .5,
NCHW blob. Output logits -> softmax over 7 classes.
"""

from __future__ import annotations

import queue

import cv2
import numpy as np
from numpy.typing import NDArray

from infrastructure.cv.model_loader import fer_path
from infrastructure.cv.yunet_detector import _default_pool_size

EXPRESSIONS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    e = np.exp(x - x.max())
    return e / e.sum()


class FERRecognizer:
    """Queue-backed pool of cv2.dnn FER nets."""

    def __init__(self, pool_size: int | None = None) -> None:
        size = pool_size if pool_size is not None else _default_pool_size()
        self._pool: queue.SimpleQueue = queue.SimpleQueue()
        path = str(fer_path())
        for _ in range(size):
            self._pool.put(cv2.dnn.readNetFromONNX(path))
        self.pool_size = size

    def predict(self, aligned_bgr_112: NDArray[np.uint8]) -> list[float]:
        rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb -= _MEAN
        rgb /= _STD
        blob = cv2.dnn.blobFromImage(rgb)  # NCHW, scale 1.0, no mean
        net = self._pool.get()
        try:
            net.setInput(blob)
            logits = net.forward().ravel().astype(np.float32)
        finally:
            self._pool.put(net)
        return [float(p) for p in _softmax(logits)]
