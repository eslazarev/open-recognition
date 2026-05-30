"""Shared fakes for application unit tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from application.ports import DetectedFace
from domain.embedding import EMBEDDING_DIM, Embedding
from domain.face import BoundingBox, Face


def make_face(left: float = 0.0, width: float = 0.5, confidence: float = 99.0) -> Face:
    return Face(
        bbox=BoundingBox(width=width, height=width, left=left, top=0.1),
        confidence=confidence,
    )


def make_detected(face: Face) -> DetectedFace:
    return DetectedFace(face=face, raw_row=np.zeros(15, dtype=np.float32))


def make_embedding(seed: int) -> Embedding:
    rng = np.random.default_rng(seed)
    return Embedding.from_array(rng.standard_normal(EMBEDDING_DIM).astype(np.float32))


@dataclass
class FakeDetector:
    """Returns a fixed list of detected faces, regardless of input image."""

    detections: list[DetectedFace]

    def detect(self, image: NDArray[np.uint8]) -> list[DetectedFace]:  # noqa: ARG002
        return list(self.detections)


@dataclass
class FakeRecognizer:
    """Maps each detected face row to a deterministic embedding via index."""

    embeddings_by_index: dict[int, Embedding]
    detections: list[DetectedFace]

    def embed(
        self, image: NDArray[np.uint8], detected: DetectedFace  # noqa: ARG002
    ) -> Embedding:
        for i, d in enumerate(self.detections):
            if d is detected:
                return self.embeddings_by_index[i]
        raise AssertionError("Detected face not registered with FakeRecognizer")
