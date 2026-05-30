import numpy as np
import pytest

from domain.embedding import EMBEDDING_DIM, Embedding


def test_embedding_normalizes_to_unit_length() -> None:
    raw = np.arange(EMBEDDING_DIM, dtype=np.float32) + 1.0
    e = Embedding.from_array(raw)
    assert pytest.approx(float(np.linalg.norm(e.vector)), abs=1e-6) == 1.0


def test_embedding_preserves_dtype_float32() -> None:
    raw = np.ones(EMBEDDING_DIM, dtype=np.float64)
    e = Embedding.from_array(raw)
    assert e.vector.dtype == np.float32


def test_embedding_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        Embedding.from_array(np.ones(64, dtype=np.float32))


def test_embedding_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        Embedding.from_array(np.ones((1, EMBEDDING_DIM), dtype=np.float32))


def test_embedding_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero"):
        Embedding.from_array(np.zeros(EMBEDDING_DIM, dtype=np.float32))


def test_embedding_to_list_returns_python_floats() -> None:
    raw = np.ones(EMBEDDING_DIM, dtype=np.float32)
    e = Embedding.from_array(raw)
    lst = e.to_list()
    assert isinstance(lst, list)
    assert len(lst) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in lst)
