"""Face value object: bounding box + landmarks + confidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Relative coordinates in [0, 1], matching the AWS shape."""

    width: float
    height: float
    left: float
    top: float

    def __post_init__(self) -> None:
        for name, v in (("width", self.width), ("height", self.height)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"BoundingBox.{name} must be in [0, 1], got {v}")


@dataclass(frozen=True, slots=True)
class Landmark:
    type: str  # eyeLeft | eyeRight | nose | mouthLeft | mouthRight
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Face:
    bbox: BoundingBox
    confidence: float  # in [0, 100], AWS-style
    landmarks: tuple[Landmark, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 100.0:
            raise ValueError(
                f"Face.confidence must be in [0, 100], got {self.confidence}"
            )
