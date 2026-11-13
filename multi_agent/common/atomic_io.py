"""Crash-safe file publication: write beside the target, fsync, then publish."""

import os
from pathlib import Path
from uuid import uuid4

from common.hashing import canonical_json


def fsync_path(path: Path) -> None:
    """Flush a file or directory entry to stable storage."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Publish complete bytes; `exclusive` refuses to replace an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        fsync_path(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, exclusive: bool = False) -> None:
    """Publish complete UTF-8 text atomically."""
    atomic_write_bytes(path, text.encode(), exclusive=exclusive)


def atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    """Publish canonical JSON plus a newline atomically."""
    atomic_write_text(path, canonical_json(value) + "\n", exclusive=exclusive)
