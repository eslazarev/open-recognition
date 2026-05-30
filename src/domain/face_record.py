"""FaceRecord aggregate: a single indexed face inside a collection."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from domain.embedding import Embedding
from domain.face import Face


@dataclass(frozen=True, slots=True)
class FaceRecord:
    face_id: UUID
    collection_id: str
    image_id: UUID
    face: Face
    embedding: Embedding
    external_image_id: str | None = None
