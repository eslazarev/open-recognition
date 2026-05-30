"""YuNet face detector adapter (cv2.FaceDetectorYN).

Holds a pool of pre-loaded detector instances so concurrent callers
do not serialize on a single mutable cv2 object. Each `detect()` call
checks one instance out, runs inference (cv2 releases the GIL during
the C++ call), and returns it to the pool.
"""

from __future__ import annotations

import os
import queue

import cv2
import numpy as np
from numpy.typing import NDArray

from application.ports import DetectedFace
from domain.face import BoundingBox, Face, Landmark
from infrastructure.cv.model_loader import yunet_path

# YuNet's 5-landmark order (per opencv_zoo docs):
#   right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner
# Mapping to AWS Rekognition landmark types is mirror-swapped because
# AWS names them from the viewer's perspective.
_LANDMARK_NAMES = ("eyeLeft", "eyeRight", "nose", "mouthLeft", "mouthRight")


def _default_pool_size() -> int:
    env = os.environ.get("OPEN_RECOGNITION_CV_POOL_SIZE")
    if env:
        return max(1, int(env))
    return max(1, min(4, os.cpu_count() or 1))


class YuNetDetector:
    """Queue-backed pool of cv2.FaceDetectorYN instances."""

    def __init__(
        self,
        pool_size: int | None = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        size = pool_size if pool_size is not None else _default_pool_size()
        self._pool: queue.SimpleQueue[cv2.FaceDetectorYN] = queue.SimpleQueue()
        path = str(yunet_path())
        for _ in range(size):
            self._pool.put(
                cv2.FaceDetectorYN.create(
                    model=path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    top_k=5000,
                )
            )
        self.pool_size = size

    def detect(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        if image.ndim != 3 or image.shape[2] != 3:
            return []
        h, w = image.shape[:2]
        det = self._pool.get()
        try:
            det.setInputSize((w, h))
            _, faces = det.detect(image)
        finally:
            self._pool.put(det)
        if faces is None:
            return []

        results: list[DetectedFace] = []
        for row in faces:
            row = row.astype(np.float32, copy=False)
            x, y, fw, fh = row[0:4]
            confidence = float(np.clip(row[14] * 100.0, 0.0, 100.0))
            bbox = BoundingBox(
                width=float(np.clip(fw / w, 0.0, 1.0)),
                height=float(np.clip(fh / h, 0.0, 1.0)),
                left=float(np.clip(x / w, 0.0, 1.0)),
                top=float(np.clip(y / h, 0.0, 1.0)),
            )
            landmarks = tuple(
                Landmark(
                    type=_LANDMARK_NAMES[i],
                    x=float(np.clip(row[4 + 2 * i] / w, 0.0, 1.0)),
                    y=float(np.clip(row[5 + 2 * i] / h, 0.0, 1.0)),
                )
                for i in range(5)
            )
            results.append(
                DetectedFace(
                    face=Face(bbox=bbox, confidence=confidence, landmarks=landmarks),
                    raw_row=row.copy(),
                )
            )
        return results
