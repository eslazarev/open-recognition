"""SearchFacesByImage use case."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from application.ports import (
    CollectionRepository,
    FaceDetector,
    FaceMatch,
    FaceRecognizer,
    FaceRepository,
)
from domain.collection import validate_collection_id
from domain.errors import (
    InvalidParameterValueError,
    ResourceNotFoundError,
)
from domain.face import Face
from domain.quality import QualityFilter, assess_face
from domain.similarity import cosine_threshold_from_pct


@dataclass(frozen=True, slots=True)
class SearchFacesByImageResult:
    searched_face: Face
    matches: list[FaceMatch]


async def search_faces_by_image(
    collection_id: str,
    image: NDArray[np.uint8],
    max_faces: int,
    face_match_threshold: float,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    collection_repo: CollectionRepository,
    face_repo: FaceRepository,
    quality_filter: QualityFilter = QualityFilter.AUTO,
) -> SearchFacesByImageResult:
    cid = validate_collection_id(collection_id)
    if await collection_repo.get(cid) is None:
        raise ResourceNotFoundError(f"Collection {cid!r} not found")
    if not 0.0 <= face_match_threshold <= 100.0:
        raise InvalidParameterValueError(
            f"FaceMatchThreshold must be in [0, 100], got {face_match_threshold}"
        )

    detected = await asyncio.to_thread(detector.detect, image)
    qualifying = [d for d in detected if not assess_face(d.face, quality_filter)]
    if not qualifying:
        if not detected:
            raise InvalidParameterValueError(
                "No face detected in the supplied image"
            )
        raise InvalidParameterValueError(
            f"No face in the image meets the QualityFilter={quality_filter.value} bar"
        )

    largest = max(qualifying, key=lambda d: d.face.bbox.width * d.face.bbox.height)
    query = await asyncio.to_thread(recognizer.embed, image, largest)
    cos_threshold = cosine_threshold_from_pct(face_match_threshold)
    matches = await face_repo.search(
        cid,
        query=query,
        max_faces=max(1, min(max_faces, 4096)),
        cosine_threshold=cos_threshold,
    )
    return SearchFacesByImageResult(searched_face=largest.face, matches=matches)
