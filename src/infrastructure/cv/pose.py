"""Approximate head pose (Roll/Yaw/Pitch, degrees) from the 5 landmarks via
solvePnP against a generic 3D face model. Approximate — sign/scale not
calibrated to AWS exactly. Landmark order: eyeLeft, eyeRight, nose,
mouthLeft, mouthRight.
"""

from __future__ import annotations

import cv2
import numpy as np

from domain.face import Landmark
from domain.face_attributes import Pose

# Generic 3D face model in millimetres.  Y axis is flipped relative to the
# "Y-up" academic convention so it aligns with OpenCV's image-plane coordinate
# system (Y increases downward).  Nose tip is the origin; eyes and mouth sit
# 30 mm behind it in Z.
_MODEL = np.array(
    [
        [-30.0, -30.0, -30.0],  # eyeLeft
        [ 30.0, -30.0, -30.0],  # eyeRight
        [  0.0,   0.0,   0.0],  # nose tip
        [-25.0,  30.0, -30.0],  # mouthLeft
        [ 25.0,  30.0, -30.0],  # mouthRight
    ],
    dtype=np.float64,
)


def estimate_pose(landmarks: tuple[Landmark, ...], width: int, height: int) -> Pose | None:
    if len(landmarks) < 5:
        return None
    img_pts = np.array(
        [[lm.x * width, lm.y * height] for lm in landmarks[:5]], dtype=np.float64
    )
    focal = float(width)
    cam = np.array(
        [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1))
    # Two-stage solve: EPnP (closed-form, works for n>=4) gives a good initial
    # estimate; ITERATIVE then refines it to sub-degree accuracy.
    ok, rvec, tvec = cv2.solvePnP(_MODEL, img_pts, cam, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    ok, rvec, _ = cv2.solvePnP(
        _MODEL, img_pts, cam, dist,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = (float(a) for a in cv2.RQDecomp3x3(rmat)[0])
    return Pose(roll=roll, yaw=yaw, pitch=pitch)
