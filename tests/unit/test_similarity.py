import numpy as np
import pytest

from domain.embedding import EMBEDDING_DIM, Embedding
from domain.similarity import (
    AWS_CALIBRATION_C,
    cosine,
    cosine_threshold_from_pct,
    similarity_pct,
)


def _emb(seed: int) -> Embedding:
    rng = np.random.default_rng(seed)
    return Embedding.from_array(rng.standard_normal(EMBEDDING_DIM).astype(np.float32))


def test_cosine_identical_vectors_is_one() -> None:
    e = _emb(0)
    assert pytest.approx(cosine(e, e), abs=1e-6) == 1.0


def test_cosine_opposite_vectors_is_minus_one() -> None:
    e = _emb(0)
    opposite = Embedding.from_array(-e.vector)
    assert pytest.approx(cosine(e, opposite), abs=1e-6) == -1.0


def test_similarity_pct_extremes() -> None:
    # Logistic saturates: cos = 1 → ~100, cos = -1 → ~0.
    assert similarity_pct(1.0) == pytest.approx(100.0)
    assert similarity_pct(-1.0) == pytest.approx(0.0)


def test_similarity_pct_midpoint_is_fifty() -> None:
    # By construction of the logistic, pct(c) = 50.
    assert similarity_pct(AWS_CALIBRATION_C) == pytest.approx(50.0)


def test_similarity_pct_is_monotonic() -> None:
    last = -1.0
    for cos in (-0.5, 0.0, 0.2, 0.3, 0.325, 0.35, 0.4, 0.6, 0.9):
        v = similarity_pct(cos)
        assert v >= last
        last = v


def test_cosine_threshold_from_pct_is_inverse_of_similarity_pct() -> None:
    # Round-trip away from the saturated extremes where precision is lost.
    for pct in (10.0, 25.0, 50.0, 80.0, 95.0, 99.0):
        cos = cosine_threshold_from_pct(pct)
        assert similarity_pct(cos) == pytest.approx(pct, abs=1e-4)


def test_cosine_threshold_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        cosine_threshold_from_pct(-1.0)
    with pytest.raises(ValueError):
        cosine_threshold_from_pct(100.5)


def test_aws_calibrated_thresholds_match_documented_cutoffs() -> None:
    # Smoke check on the AWS-style thresholds we promise in similarity.py
    # docstring; if these slip, the linked calibration data is stale.
    assert cosine_threshold_from_pct(80.0) == pytest.approx(0.353, abs=0.005)
    assert cosine_threshold_from_pct(90.0) == pytest.approx(0.370, abs=0.005)
    assert cosine_threshold_from_pct(95.0) == pytest.approx(0.385, abs=0.005)
    assert cosine_threshold_from_pct(99.0) == pytest.approx(0.419, abs=0.005)
