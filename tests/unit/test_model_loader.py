"""Unit tests for model_loader checksum verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from infrastructure.cv import model_loader


def _sha256_of(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture
def tmp_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("OPEN_RECOGNITION_MODELS_DIR", str(tmp_path))
    return tmp_path


def test_existing_file_with_matching_hash_is_returned_unchanged(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"hello opencv zoo"
    path = tmp_models / "fake.onnx"
    path.write_bytes(payload)

    downloads: list[str] = []
    monkeypatch.setattr(model_loader, "_download", lambda url, p: downloads.append(url))

    out = model_loader._ensure("fake.onnx", "https://example.invalid/x", _sha256_of(payload))
    assert out == path
    assert downloads == []  # never touched the network


def test_missing_file_downloads_and_verifies(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"freshly fetched"
    expected = _sha256_of(payload)

    def fake_download(url: str, p: Path) -> None:
        p.write_bytes(payload)

    monkeypatch.setattr(model_loader, "_download", fake_download)

    out = model_loader._ensure("fresh.onnx", "https://example.invalid/x", expected)
    assert out.read_bytes() == payload


def test_mismatch_after_download_raises_checksum_error(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(url: str, p: Path) -> None:
        p.write_bytes(b"tampered")

    monkeypatch.setattr(model_loader, "_download", fake_download)

    with pytest.raises(model_loader.ChecksumError, match="expected SHA256"):
        model_loader._ensure("bad.onnx", "https://example.invalid/x", "deadbeef" * 8)


def test_existing_corrupted_file_is_replaced_then_verified(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # File on disk is wrong; fresh download is correct.
    bad_path = tmp_models / "rotate.onnx"
    bad_path.write_bytes(b"old wrong file")
    good_payload = b"the correct one"

    def fake_download(url: str, p: Path) -> None:
        p.write_bytes(good_payload)

    monkeypatch.setattr(model_loader, "_download", fake_download)

    out = model_loader._ensure(
        "rotate.onnx", "https://example.invalid/x", _sha256_of(good_payload)
    )
    assert out.read_bytes() == good_payload
