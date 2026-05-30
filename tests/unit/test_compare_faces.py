import numpy as np
import pytest

from application.compare_faces import compare_faces
from domain.errors import InvalidParameterValueError
from tests.unit.conftest import (
    FakeDetector,
    FakeRecognizer,
    make_detected,
    make_embedding,
    make_face,
)


def _blank() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_compare_faces_matches_identical_embedding() -> None:
    src = make_detected(make_face())
    tgt = make_detected(make_face(left=0.5))
    emb = make_embedding(42)

    src_det = FakeDetector(detections=[src])
    tgt_det = FakeDetector(detections=[tgt])

    # When the same detector is used for both .detect() calls, swap
    # via a wrapper that returns src first then tgt:
    calls: list = [src_det.detections, tgt_det.detections]

    class TwoCallDetector:
        def detect(self, image):  # noqa: ARG002
            return list(calls.pop(0))

    rec = FakeRecognizer(
        detections=[src, tgt], embeddings_by_index={0: emb, 1: emb}
    )

    result = compare_faces(
        source=_blank(),
        target=_blank(),
        similarity_threshold=80.0,
        detector=TwoCallDetector(),
        recognizer=rec,
    )

    assert len(result.face_matches) == 1
    assert result.face_matches[0].similarity == pytest.approx(100.0, abs=1e-4)
    assert result.unmatched_faces == []


def test_compare_faces_drops_below_threshold() -> None:
    src = make_detected(make_face())
    tgt = make_detected(make_face(left=0.5))
    src_emb = make_embedding(1)
    tgt_emb = make_embedding(2)  # different random vector, similarity ~50 %

    calls = [[src], [tgt]]

    class TwoCallDetector:
        def detect(self, image):  # noqa: ARG002
            return list(calls.pop(0))

    rec = FakeRecognizer(
        detections=[src, tgt], embeddings_by_index={0: src_emb, 1: tgt_emb}
    )

    result = compare_faces(
        source=_blank(),
        target=_blank(),
        similarity_threshold=99.0,
        detector=TwoCallDetector(),
        recognizer=rec,
    )

    assert result.face_matches == []
    assert len(result.unmatched_faces) == 1


def test_compare_faces_rejects_bad_threshold() -> None:
    with pytest.raises(InvalidParameterValueError):
        compare_faces(
            source=_blank(),
            target=_blank(),
            similarity_threshold=150.0,
            detector=FakeDetector(detections=[]),
            recognizer=FakeRecognizer(detections=[], embeddings_by_index={}),
        )


def test_compare_faces_raises_if_no_source_face() -> None:
    with pytest.raises(InvalidParameterValueError, match="SourceImage"):
        compare_faces(
            source=_blank(),
            target=_blank(),
            similarity_threshold=80.0,
            detector=FakeDetector(detections=[]),
            recognizer=FakeRecognizer(detections=[], embeddings_by_index={}),
        )
