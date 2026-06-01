"""Runs only the analyzers needed for the requested attribute set.

pose/quality are cheap pure-CV; emotions/smile use the pooled FER net plus an
aligner (the SFaceRecognizer, reused for its alignCrop). Smile is derived from
the 'happy' emotion probability.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from application.ports import DetectedFace
from domain.face_attributes import BinaryAttr, Emotion, FaceAttributes
from infrastructure.cv.fer_recognizer import EXPRESSIONS
from infrastructure.cv.pose import estimate_pose
from infrastructure.cv.quality import assess_quality

_EMOTION_MAP = {
    "angry": "ANGRY", "disgust": "DISGUSTED", "fearful": "FEAR",
    "happy": "HAPPY", "neutral": "CALM", "sad": "SAD", "surprised": "SURPRISED",
}


class _FER(Protocol):
    def predict(self, aligned_bgr_112: NDArray[np.uint8]) -> list[float]: ...


class _Aligner(Protocol):
    def align(self, image: NDArray[np.uint8], detected: DetectedFace) -> NDArray[np.uint8]: ...


class CompositeFaceAnalyzer:
    def __init__(self, fer: _FER, aligner: _Aligner) -> None:
        self._fer = fer
        self._aligner = aligner

    def analyze(
        self, image: NDArray[np.uint8], detected: DetectedFace, requested: set[str]
    ) -> FaceAttributes:
        face = detected.face
        pose = None
        quality = None
        emotions: tuple[Emotion, ...] = ()
        smile = None

        if "pose" in requested:
            h, w = image.shape[:2]
            pose = estimate_pose(face.landmarks, w, h)
        if "quality" in requested:
            quality = assess_quality(image, face.bbox)

        if "emotions" in requested or "smile" in requested:
            crop = self._aligner.align(image, detected)
            probs = self._fer.predict(crop)
            by_label = dict(zip(EXPRESSIONS, probs, strict=False))
            if "emotions" in requested:
                emotions = tuple(
                    sorted(
                        (
                            Emotion(type=_EMOTION_MAP[lbl], confidence=p * 100.0)
                            for lbl, p in by_label.items()
                        ),
                        key=lambda e: e.confidence,
                        reverse=True,
                    )
                )
            if "smile" in requested:
                happy = by_label["happy"]
                smile = BinaryAttr(value=happy >= 0.5, confidence=happy * 100.0)

        return FaceAttributes(pose=pose, quality=quality, emotions=emotions, smile=smile)
