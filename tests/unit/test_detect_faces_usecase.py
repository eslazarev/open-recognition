import numpy as np

from application.detect_faces import detect_faces
from application.ports import DetectedFace
from domain.face import BoundingBox, Face
from domain.face_attributes import FaceAttributes, Pose


class _Detector:
    def detect(self, image):
        f = Face(bbox=BoundingBox(0.1, 0.1, 0.1, 0.1), confidence=99.0)
        return [DetectedFace(face=f, raw_row=np.zeros(15, dtype=np.float32))]


class _Analyzer:
    def analyze(self, image, detected, requested):
        return FaceAttributes(pose=Pose(1.0, 2.0, 3.0))


def test_without_analyzer_attributes_empty():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    res = detect_faces(img, _Detector())
    assert len(res.faces) == 1
    assert res.attributes[0] == FaceAttributes()


def test_with_analyzer_runs_per_face():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    res = detect_faces(img, _Detector(), analyzer=_Analyzer(), requested={"pose"})
    assert res.attributes[0].pose == Pose(1.0, 2.0, 3.0)
