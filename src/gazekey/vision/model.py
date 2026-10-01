import hashlib
import os
import urllib.request
from pathlib import Path

_CHUNK = 1 << 20


class ModelIntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(url: str, dest: Path, *, expected_sha256: str | None = None) -> str:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as response, part.open("wb") as out:
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256.lower():
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
        os.replace(part, dest)
        return actual
    finally:
        part.unlink(missing_ok=True)
