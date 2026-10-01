import hashlib
from pathlib import Path

import pytest

from gazekey.vision.model import ModelIntegrityError, download_model, sha256_file

PAYLOAD = b"fake-model-bytes" * 100_000


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "src.task"
    path.write_bytes(PAYLOAD)
    return path


def test_sha256_file(source: Path) -> None:
    assert sha256_file(source) == hashlib.sha256(PAYLOAD).hexdigest()


def test_download_and_verify(source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "models" / "m.task"
    expected = hashlib.sha256(PAYLOAD).hexdigest()
    digest = download_model(source.as_uri(), dest, expected_sha256=expected)
    assert dest.read_bytes() == PAYLOAD
    assert digest == expected
    assert not dest.with_name("m.task.part").exists()


def test_download_without_pin_returns_hash(source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "m.task"
    assert download_model(source.as_uri(), dest) == hashlib.sha256(PAYLOAD).hexdigest()


def test_hash_mismatch_leaves_nothing_behind(source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "m.task"
    with pytest.raises(ModelIntegrityError):
        download_model(source.as_uri(), dest, expected_sha256="0" * 64)
    assert list(tmp_path.glob("m.task*")) == []


def test_failed_download_does_not_clobber_existing_file(tmp_path: Path) -> None:
    dest = tmp_path / "m.task"
    dest.write_bytes(b"good")
    with pytest.raises(OSError):
        download_model((tmp_path / "nope.task").as_uri(), dest)
    assert dest.read_bytes() == b"good"
