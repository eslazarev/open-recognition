"""IndexFaces use case."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from application.ports import (
    CollectionRepository,
    DetectedFace,
    FaceDetector,
    FaceRecognizer,
    FaceRepository,
)
from domain.collection import validate_collection_id
from domain.errors import ResourceNotFoundError
from domain.face import Face
from domain.face_record import FaceRecord
from domain.quality import QualityFilter, assess_face


@dataclass(frozen=True, slots=True)
class UnindexedFace:
    face: Face
    reasons: list[str]


@dataclass(frozen=True, slots=True)
class IndexFacesResult:
    records: list[FaceRecord]
    unindexed: list[UnindexedFace]


async def index_faces(
    collection_id: str,
    image: NDArray[np.uint8],
    external_image_id: str | None,
    max_faces: int,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    collection_repo: CollectionRepository,
    face_repo: FaceRepository,
    quality_filter: QualityFilter = QualityFilter.AUTO,
) -> IndexFacesResult:
    cid = validate_collection_id(collection_id)
    if await collection_repo.get(cid) is None:
        raise ResourceNotFoundError(f"Collection {cid!r} not found")

    detected = await asyncio.to_thread(detector.detect, image)

    # AWS order: QualityFilter first (those become UnindexedFaces with
    # specific Reasons), then sort by size, then MaxFaces cap (the cap
    # overflow gets reason EXCEEDS_MAX_FACES).
    passed: list[DetectedFace] = []
    unindexed: list[UnindexedFace] = []
    for d in detected:
        reasons = assess_face(d.face, quality_filter)
        if reasons:
            unindexed.append(UnindexedFace(face=d.face, reasons=reasons))
        else:
            passed.append(d)

    passed.sort(
        key=lambda d: d.face.bbox.width * d.face.bbox.height,
        reverse=True,
    )
    cap = max(1, max_faces)
    keep, drop = passed[:cap], passed[cap:]
    unindexed.extend(
        UnindexedFace(face=d.face, reasons=["EXCEEDS_MAX_FACES"]) for d in drop
    )

    embeddings = await asyncio.gather(
        *(asyncio.to_thread(recognizer.embed, image, d) for d in keep)
    )

    image_id = uuid4()
    records = [
        FaceRecord(
            face_id=uuid4(),
            collection_id=cid,
            image_id=image_id,
            face=d.face,
            embedding=emb,
            external_image_id=external_image_id,
        )
        for d, emb in zip(keep, embeddings, strict=True)
    ]
    await face_repo.add_many(records)

    return IndexFacesResult(records=records, unindexed=unindexed)
