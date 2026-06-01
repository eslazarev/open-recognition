"""SFace embedding adapter (cv2.FaceRecognizerSF).

Same queue-backed pool pattern as YuNetDetector — multiple recognizer
instances let concurrent embed() calls run in parallel; cv2 releases
the GIL inside alignCrop/feature.
"""

from __future__ import annotations

import queue

import cv2
import numpy as np
from numpy.typing import NDArray

from application.ports import DetectedFace
from domain.embedding import Embedding
from infrastructure.cv.model_loader import sface_path
from infrastructure.cv.yunet_detector import _default_pool_size


class SFaceRecognizer:
    """Queue-backed pool of cv2.FaceRecognizerSF instances."""

    def __init__(self, pool_size: int | None = None) -> None:
        size = pool_size if pool_size is not None else _default_pool_size()
        self._pool: queue.SimpleQueue[cv2.FaceRecognizerSF] = queue.SimpleQueue()
        path = str(sface_path())
        for _ in range(size):
            self._pool.put(
                cv2.FaceRecognizerSF.create(model=path, config="")
            )
        self.pool_size = size

    def align(self, image: NDArray[np.uint8], detected: DetectedFace) -> NDArray[np.uint8]:
        """Return the 112x112 aligned BGR crop (reuses SFace's aligner)."""
        rec = self._pool.get()
        try:
            return rec.alignCrop(image, detected.raw_row)
        finally:
            self._pool.put(rec)

    def embed(self, image: NDArray[np.uint8], detected: DetectedFace) -> Embedding:
        rec = self._pool.get()
        try:
            aligned = rec.alignCrop(image, detected.raw_row)
            feature = rec.feature(aligned)
        finally:
            self._pool.put(rec)
        return Embedding.from_array(np.asarray(feature, dtype=np.float32).ravel())
