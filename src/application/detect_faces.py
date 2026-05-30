"""DetectFaces use case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from application.ports import FaceDetector
from domain.face import Face


@dataclass(frozen=True, slots=True)
class DetectFacesResult:
    faces: list[Face]


def detect_faces(image: NDArray[np.uint8], detector: FaceDetector) -> DetectFacesResult:
    detected = detector.detect(image)
    return DetectFacesResult(faces=[d.face for d in detected])
