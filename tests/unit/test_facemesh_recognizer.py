import numpy as np
import pytest

from infrastructure.cv.facemesh_recognizer import FaceMeshRecognizer
from infrastructure.cv.image_decoder import decode_image
from infrastructure.cv.yunet_detector import YuNetDetector


@pytest.fixture(scope="module")
def mesh():
    try:
        return FaceMeshRecognizer(pool_size=1)
    except Exception as exc:
        pytest.skip(f"face mesh model unavailable: {exc}")


def test_landmarks_on_real_face(mesh):
    img = decode_image(open("tests/fixtures/face_a.jpg", "rb").read())
    det = YuNetDetector(pool_size=1).detect(img)
    assert det, "fixture should contain a face"
    pts = mesh.landmarks(img, det[0])
    assert pts is not None
    assert pts.shape == (478, 2)
    assert float(pts.min()) >= -0.1 and float(pts.max()) <= 1.1
