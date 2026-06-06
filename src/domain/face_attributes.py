"""Pure value objects for DetectFaces attributes (no I/O, no cv2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Pose:
    roll: float
    yaw: float
    pitch: float


@dataclass(frozen=True, slots=True)
class ImageQuality:
    brightness: float  # 0..100
    sharpness: float   # 0..100


@dataclass(frozen=True, slots=True)
class Emotion:
    type: str          # AWS enum: HAPPY, SAD, ANGRY, CALM, DISGUSTED, SURPRISED, FEAR
    confidence: float  # 0..100


@dataclass(frozen=True, slots=True)
class BinaryAttr:
    value: bool
    confidence: float  # 0..100


@dataclass(frozen=True, slots=True)
class FaceAttributes:
    pose: Pose | None = None
    quality: ImageQuality | None = None
    emotions: tuple[Emotion, ...] = field(default_factory=tuple)
    smile: BinaryAttr | None = None
    landmarks: tuple = field(default_factory=tuple)  # tuple[domain.face.Landmark, ...]
    eyes_open: BinaryAttr | None = None
    mouth_open: BinaryAttr | None = None
