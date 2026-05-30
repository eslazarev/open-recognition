"""CompareFaces use case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from application.ports import FaceDetector, FaceRecognizer
from domain.errors import InvalidParameterValueError
from domain.face import Face
from domain.similarity import cosine, similarity_pct


@dataclass(frozen=True, slots=True)
class CompareFaceMatch:
    face: Face
    similarity: float  # AWS percentage


@dataclass(frozen=True, slots=True)
class CompareFacesResult:
    source_face: Face
    face_matches: list[CompareFaceMatch]
    unmatched_faces: list[Face]


def _largest(detected_faces: list[Face]) -> Face:
    return max(detected_faces, key=lambda f: f.bbox.width * f.bbox.height)


def compare_faces(
    source: NDArray[np.uint8],
    target: NDArray[np.uint8],
    similarity_threshold: float,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
) -> CompareFacesResult:
    if not 0.0 <= similarity_threshold <= 100.0:
        raise InvalidParameterValueError(
            f"SimilarityThreshold must be in [0, 100], got {similarity_threshold}"
        )

    source_detected = detector.detect(source)
    if not source_detected:
        raise InvalidParameterValueError("No face detected in SourceImage")

    src = max(source_detected, key=lambda d: d.face.bbox.width * d.face.bbox.height)
    src_embedding = recognizer.embed(source, src)

    matches: list[CompareFaceMatch] = []
    unmatched: list[Face] = []
    for d in detector.detect(target):
        tgt_embedding = recognizer.embed(target, d)
        pct = similarity_pct(cosine(src_embedding, tgt_embedding))
        if pct >= similarity_threshold:
            matches.append(CompareFaceMatch(face=d.face, similarity=pct))
        else:
            unmatched.append(d.face)

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return CompareFacesResult(
        source_face=src.face, face_matches=matches, unmatched_faces=unmatched
    )
