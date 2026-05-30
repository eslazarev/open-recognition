"""Ports — protocols the application layer depends on.

Implementations live under `infrastructure.*`. Use-case
functions accept these as parameters; tests substitute fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from domain.collection import Collection
from domain.embedding import Embedding
from domain.face import Face
from domain.face_record import FaceRecord


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """Raw detector output, still carrying the data needed for alignment."""

    face: Face
    raw_row: NDArray[np.float32]  # 15-element row produced by FaceDetectorYN


class FaceDetector(Protocol):
    def detect(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        ...


class FaceRecognizer(Protocol):
    def embed(self, image: NDArray[np.uint8], detected: DetectedFace) -> Embedding:
        ...


class CollectionRepository(Protocol):
    async def create(self, collection: Collection) -> None: ...
    async def delete(self, collection_id: str) -> None: ...
    async def get(self, collection_id: str) -> Collection | None: ...
    async def list(self, limit: int, offset: int) -> tuple[list[Collection], int]: ...


@dataclass(frozen=True, slots=True)
class FaceMatch:
    face_record: FaceRecord
    similarity: float  # AWS percentage [0, 100]


class FaceRepository(Protocol):
    async def add_many(self, records: list[FaceRecord]) -> None: ...
    async def delete_many(self, collection_id: str, face_ids: list[UUID]) -> list[UUID]: ...
    async def count(self, collection_id: str) -> int: ...
    async def list(
        self, collection_id: str, limit: int, offset: int
    ) -> tuple[list[FaceRecord], int]: ...
    async def search(
        self,
        collection_id: str,
        query: Embedding,
        max_faces: int,
        cosine_threshold: float,
    ) -> list[FaceMatch]: ...
