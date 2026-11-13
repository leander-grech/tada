"""Canonical JSON identities, SHA-256 helpers and immutable transport values."""

import hashlib
import json
from pathlib import Path

_HEX = frozenset("0123456789abcdef")


def canonical_json(value: object) -> str:
    """Canonical encoding shared by configuration, streams and artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return bytes_sha256(Path(path).read_bytes())


def content_digest(value: object) -> str:
    """Hash canonical JSON, never Python's process-randomized hash()."""
    return bytes_sha256(canonical_json(value).encode())


def is_sha256(value: object) -> bool:
    """True only for a lowercase 64-character hexadecimal digest string."""
    return isinstance(value, str) and len(value) == 64 and _HEX.issuperset(value)


def nonnegative_int(value: object, label: str) -> int:
    """Reject lossy casts (including bool) at persisted-state boundaries."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


class FrozenDict(dict):
    """Deep immutable JSON object, also safe to pickle for remote workers."""

    def __init__(self, value=(), **kwargs):
        dict.__init__(
            self, ((key, freeze_json(item)) for key, item in dict(value, **kwargs).items())
        )

    def _immutable(self, *_args, **_kwargs):
        raise TypeError("resolved payload is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __reduce__(self):
        return FrozenDict, (dict(self),)

    def copy(self):
        return self


def freeze_json(value):
    if isinstance(value, dict):
        return FrozenDict(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item) for item in value)
    return value
