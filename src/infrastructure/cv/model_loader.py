"""Lazy, checksum-verified downloader for opencv_zoo ONNX weights.

Hashes are pinned at code time. On every load we hash the file on
disk and compare against the expected SHA256. If a downloaded file
fails verification we delete it and try once more; persistent
mismatch raises so a poisoned upstream never silently propagates.

Models land in `OPEN_RECOGNITION_MODELS_DIR` (default `./models`). To rotate
to a newer release, update both the URL and the SHA256 here in one
commit.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
FER_FILENAME = "facial_expression_recognition_mobilefacenet_2022july.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/" + YUNET_FILENAME
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/" + SFACE_FILENAME
)
FER_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "facial_expression_recognition/" + FER_FILENAME
)

# SHA256 of the artifacts we've validated against (Mar 2026 snapshot of
# opencv_zoo). Rotating to a different model version requires updating
# both the URL above and the hash here in one commit.
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
# opencv_zoo facial_expression_recognition (Apache 2.0), verified 2026-06-01.
FER_SHA256 = "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9"

FACEMESH_FILENAME = "face_mesh_478.onnx"

FACEMESH_URL = "https://huggingface.co/astaileyyoung/FaceMeshONNX/resolve/main/mesh.onnx"

# MediaPipe Face Mesh ONNX re-host (MIT); upstream Apache-2.0. Verified 2026-06-01.
FACEMESH_SHA256 = "c334c26b128dfc709da0f353c6157ac340b785f5f25cdab5dbea12bbea243172"


class ChecksumError(RuntimeError):
    """Raised when a model file's hash does not match the pinned value."""


def models_dir() -> Path:
    p = Path(os.environ.get("OPEN_RECOGNITION_MODELS_DIR", "models"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(64 * 1024):
            fh.write(chunk)
    tmp.replace(path)


def _ensure(filename: str, url: str, expected_sha: str) -> Path:
    path = models_dir() / filename
    for attempt in (1, 2):
        if not (path.exists() and path.stat().st_size > 0):
            _download(url, path)
        actual = _sha256(path)
        if actual == expected_sha:
            return path
        # Hash mismatch: nuke and retry once. If the upstream is
        # genuinely poisoned the second download will also fail and
        # we raise.
        path.unlink(missing_ok=True)
        if attempt == 2:
            raise ChecksumError(
                f"{filename}: expected SHA256 {expected_sha}, got {actual}; "
                f"refused to use the file."
            )
    raise AssertionError("unreachable")  # pragma: no cover


def yunet_path() -> Path:
    return _ensure(YUNET_FILENAME, YUNET_URL, YUNET_SHA256)


def sface_path() -> Path:
    return _ensure(SFACE_FILENAME, SFACE_URL, SFACE_SHA256)


def fer_path() -> Path:
    return _ensure(FER_FILENAME, FER_URL, FER_SHA256)


def facemesh_path() -> Path:
    return _ensure(FACEMESH_FILENAME, FACEMESH_URL, FACEMESH_SHA256)
