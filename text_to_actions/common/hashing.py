"""SHA256 helpers for data files and serialized cache keys."""

import hashlib
import json
from pathlib import Path


def sha256_bytes(value) -> str:
    """Calculate the SHA256 digest of an existing byte sequence."""
    return hashlib.sha256(value).hexdigest()


def file_sha256(path) -> str:
    """Calculate the SHA256 digest of a file's contents."""
    return sha256_bytes(Path(path).read_bytes())


def fingerprint(value, *, ensure_ascii=False) -> str:
    """Hash sorted JSON while preserving the caller's cache serialization format."""
    serialized = json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True)
    return sha256_bytes(serialized.encode())
