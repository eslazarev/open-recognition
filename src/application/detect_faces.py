"""DetectFaces use case."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from application.ports import FaceAnalyzer, FaceDetector
from domain.face import Face
from domain.face_attributes import FaceAttributes


@dataclass(frozen=True, slots=True)
class DetectFacesResult:
    faces: list[Face]
    attributes: list[FaceAttributes] = field(default_factory=list)


def detect_faces(
    image: NDArray[np.uint8],
    detector: FaceDetector,
    analyzer: FaceAnalyzer | None = None,
    requested: set[str] | None = None,
) -> DetectFacesResult:
    detected = detector.detect(image)
    faces = [d.face for d in detected]
    if analyzer is not None and requested:
        attrs = [analyzer.analyze(image, d, set(requested)) for d in detected]
    else:
        attrs = [FaceAttributes() for _ in detected]
    return DetectFacesResult(faces=faces, attributes=attrs)
