"""Runs only the analyzers needed for the requested attribute set.

pose/quality are cheap pure-CV; emotions/smile use the pooled FER net + an
aligner (SFaceRecognizer.alignCrop); landmarks/eyes_open/mouth_open use the
face-mesh recognizer (478 pts) -> AWS-named landmarks + EAR/MAR.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from application.ports import DetectedFace
from domain.face_attributes import BinaryAttr, Emotion, FaceAttributes
from domain.landmarks import eye_aspect_ratio, mouth_open_ratio, named_landmarks
from infrastructure.cv.fer_recognizer import EXPRESSIONS
from infrastructure.cv.pose import estimate_pose
from infrastructure.cv.quality import assess_quality

_EMOTION_MAP = {
    "angry": "ANGRY", "disgust": "DISGUSTED", "fearful": "FEAR",
    "happy": "HAPPY", "neutral": "CALM", "sad": "SAD", "surprised": "SURPRISED",
}
_EAR_OPEN = 0.18
# MouthOpen = inner-lip gap / inter-pupil distance. Threshold 0.10 validated
# against AWS on 25 faces (closed <=0.095, open >=0.104) — catches wide
# teeth-showing smiles that the old gap/mouth-width metric missed.
_MOUTH_OPEN = 0.10
# Logistic confidence slopes (per ratio unit): tuned so the typical open/closed
# clusters read ~95%+ while values right at the threshold read ~50%.
_EAR_SLOPE = 50.0
_MOUTH_SLOPE = 45.0


def _confidence(value: float, threshold: float, slope: float) -> float:
    """Logistic confidence in the binary decision, centred at the threshold.

    50% exactly at the boundary, rising smoothly toward 100% as the value moves
    clearly into the open or closed region. `slope` is calibrated per attribute
    so the typical clusters read ~95%+. Always >= 50 by construction (it scores
    the chosen — more probable — side).
    """
    p_above = 1.0 / (1.0 + math.exp(-slope * (value - threshold)))
    conf = p_above if value >= threshold else 1.0 - p_above
    return float(conf * 100.0)


class _FER(Protocol):
    def predict(self, aligned_bgr_112: NDArray[np.uint8]) -> list[float]: ...


class _Aligner(Protocol):
    def align(self, image: NDArray[np.uint8], detected: DetectedFace) -> NDArray[np.uint8]: ...


class _Mesh(Protocol):
    def landmarks(
        self, image: NDArray[np.uint8], detected: DetectedFace
    ) -> NDArray[np.float64] | None: ...


class CompositeFaceAnalyzer:
    def __init__(self, fer: _FER, aligner: _Aligner, mesh: _Mesh) -> None:
        self._fer = fer
        self._aligner = aligner
        self._mesh = mesh

    def analyze(
        self, image: NDArray[np.uint8], detected: DetectedFace, requested: set[str]
    ) -> FaceAttributes:
        face = detected.face
        h, w = image.shape[:2]
        pose = quality = smile = eyes_open = mouth_open = None
        emotions: tuple[Emotion, ...] = ()
        landmarks: tuple = ()

        if "pose" in requested:
            pose = estimate_pose(face.landmarks, w, h)
        if "quality" in requested:
            quality = assess_quality(image, face.bbox)

        if requested & {"landmarks", "eyes_open", "mouth_open"}:
            mesh = self._mesh.landmarks(image, detected)
            if mesh is not None:
                if "landmarks" in requested:
                    landmarks = tuple(named_landmarks(mesh))
                if "eyes_open" in requested:
                    ear = eye_aspect_ratio(mesh, w, h)
                    eyes_open = BinaryAttr(
                        value=ear >= _EAR_OPEN,
                        confidence=_confidence(ear, _EAR_OPEN, _EAR_SLOPE),
                    )
                if "mouth_open" in requested:
                    mor = mouth_open_ratio(mesh, w, h)
                    mouth_open = BinaryAttr(
                        value=mor >= _MOUTH_OPEN,
                        confidence=_confidence(mor, _MOUTH_OPEN, _MOUTH_SLOPE),
                    )

        if "emotions" in requested or "smile" in requested:
            crop = self._aligner.align(image, detected)
            probs = self._fer.predict(crop)
            by_label = dict(zip(EXPRESSIONS, probs, strict=False))
            if "emotions" in requested:
                emotions = tuple(
                    sorted(
                        (Emotion(type=_EMOTION_MAP[lbl], confidence=p * 100.0)
                         for lbl, p in by_label.items()),
                        key=lambda e: e.confidence, reverse=True,
                    )
                )
            if "smile" in requested:
                happy = by_label["happy"]
                smile = BinaryAttr(value=happy >= 0.5, confidence=happy * 100.0)

        return FaceAttributes(
            pose=pose, quality=quality, emotions=emotions, smile=smile,
            landmarks=landmarks, eyes_open=eyes_open, mouth_open=mouth_open,
        )
