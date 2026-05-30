"""Embedding value object: 128-dim L2-normalised face descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

EMBEDDING_DIM = 128


@dataclass(frozen=True, slots=True)
class Embedding:
    """Unit-norm float32 vector of length EMBEDDING_DIM."""

    vector: NDArray[np.float32]

    @classmethod
    def from_array(cls, arr: NDArray[np.floating]) -> Embedding:
        if arr.ndim != 1:
            raise ValueError(f"Embedding must be 1-D, got shape {arr.shape}")
        if arr.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must have length EMBEDDING_DIM={EMBEDDING_DIM}, got {arr.shape[0]}"
            )
        v = arr.astype(np.float32, copy=True)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise ValueError("Cannot normalise a zero vector")
        v /= norm
        return cls(vector=v)

    def to_list(self) -> list[float]:
        return [float(x) for x in self.vector]
