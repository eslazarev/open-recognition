"""Map MediaPipe Face Mesh (478 pts) to AWS-named landmarks, and compute
eye/mouth aspect ratios. Pure: operates on a numpy [478, 2] array of
image-normalized coordinates. No cv2, no I/O.

Index map: eyes/pupils/nose/mouth are verified positions; eyebrow/jaw indices
are approximate (validate visually). AWS eyeLeft/eyeRight follow the subject's
perspective, matching YuNet's convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from domain.face import Landmark

AWS_LANDMARK_INDEX: dict[str, int] = {
    "eyeLeft": 468, "eyeRight": 473, "leftPupil": 468, "rightPupil": 473,
    "nose": 1, "mouthLeft": 61, "mouthRight": 291, "mouthUp": 13, "mouthDown": 14,
    "noseLeft": 48, "noseRight": 278,
    "leftEyeLeft": 33, "leftEyeRight": 133, "leftEyeUp": 159, "leftEyeDown": 145,
    "rightEyeLeft": 362, "rightEyeRight": 263, "rightEyeUp": 386, "rightEyeDown": 374,
    "leftEyeBrowLeft": 46, "leftEyeBrowUp": 105, "leftEyeBrowRight": 107,
    "rightEyeBrowLeft": 336, "rightEyeBrowUp": 334, "rightEyeBrowRight": 276,
    "upperJawlineLeft": 234, "midJawlineLeft": 58, "chinBottom": 152,
    "midJawlineRight": 288, "upperJawlineRight": 454,
}

_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def named_landmarks(mesh: NDArray[np.float64]) -> list[Landmark]:
    out: list[Landmark] = []
    for name, idx in AWS_LANDMARK_INDEX.items():
        x, y = float(mesh[idx, 0]), float(mesh[idx, 1])
        out.append(Landmark(type=name, x=min(1.0, max(0.0, x)), y=min(1.0, max(0.0, y))))
    return out


def _ear_one(mesh: NDArray[np.float64], idx: tuple[int, ...], w: int, h: int) -> float:
    p = [np.array([mesh[i, 0] * w, mesh[i, 1] * h]) for i in idx]
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz == 0:
        return 0.0
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(vert / (2.0 * horiz))


def eye_aspect_ratio(mesh: NDArray[np.float64], w: int, h: int) -> float:
    return (_ear_one(mesh, _LEFT_EYE, w, h) + _ear_one(mesh, _RIGHT_EYE, w, h)) / 2.0


def mouth_open_ratio(mesh: NDArray[np.float64], w: int, h: int) -> float:
    """Vertical inner-lip gap (13,14) normalized by inter-pupil distance (468,473).

    Normalizing by inter-pupil distance — NOT mouth width — is deliberate: a
    teeth-showing smile has a modest lip gap but a wide mouth, so dividing by
    mouth width under-reads it (it overlaps closed mouths around 0.12).
    gap/inter-pupil separates open from closed with a wide margin (validated
    against AWS on 25 faces).
    """
    up = np.array([mesh[13, 0] * w, mesh[13, 1] * h])
    down = np.array([mesh[14, 0] * w, mesh[14, 1] * h])
    lpupil = np.array([mesh[468, 0] * w, mesh[468, 1] * h])
    rpupil = np.array([mesh[473, 0] * w, mesh[473, 1] * h])
    iod = np.linalg.norm(lpupil - rpupil)
    if iod == 0:
        return 0.0
    return float(np.linalg.norm(up - down) / iod)
