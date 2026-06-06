"""MediaPipe Face Mesh (478 pts) via onnxruntime (pooled sessions).

cv2.dnn can't run this NHWC/TFLite-origin graph, so we use onnxruntime — the
only non-opencv inference runtime in the project, isolated here. Input is a
square, border-padded crop around the YuNet bbox, resized to 256x256, RGB/255.
Returns 478 landmarks in image-relative [0, 1] coordinates, or None when the
face-presence logit is negative.
"""

from __future__ import annotations

import queue

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from application.ports import DetectedFace
from infrastructure.cv.model_loader import facemesh_path
from infrastructure.cv.yunet_detector import _default_pool_size

_SIZE = 256
_PAD = 0.75  # half-extent multiplier around max(bbox side)


class FaceMeshRecognizer:
    """Queue-backed pool of onnxruntime sessions for the face-mesh model."""

    def __init__(self, pool_size: int | None = None) -> None:
        size = pool_size if pool_size is not None else _default_pool_size()
        self._pool: queue.SimpleQueue = queue.SimpleQueue()
        path = str(facemesh_path())
        first = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input = first.get_inputs()[0].name
        self._pool.put(first)
        for _ in range(size - 1):
            self._pool.put(ort.InferenceSession(path, providers=["CPUExecutionProvider"]))
        self.pool_size = size

    def landmarks(
        self, image: NDArray[np.uint8], detected: DetectedFace
    ) -> NDArray[np.float64] | None:
        h, w = image.shape[:2]
        b = detected.face.bbox
        bw, bh = b.width * w, b.height * h
        cx, cy = (b.left + b.width / 2) * w, (b.top + b.height / 2) * h
        side = int(max(bw, bh) * 2 * _PAD)
        if side <= 0:
            return None
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        sq = np.zeros((side, side, 3), dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
        sq[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
        resized = cv2.cvtColor(cv2.resize(sq, (_SIZE, _SIZE)), cv2.COLOR_BGR2RGB)
        inp = resized.astype(np.float32) / 255.0

        sess = self._pool.get()
        try:
            outs = sess.run(None, {self._input: inp[None, ...]})
        finally:
            self._pool.put(sess)
        lm = np.asarray(outs[0], dtype=np.float64).ravel()[: 478 * 3].reshape(478, 3)
        presence = float(np.asarray(outs[1]).ravel()[0])
        if presence < 0:
            return None
        xy = lm[:, :2].copy()
        xy[:, 0] = (x0 + xy[:, 0] / _SIZE * side) / w
        xy[:, 1] = (y0 + xy[:, 1] / _SIZE * side) / h
        return xy
