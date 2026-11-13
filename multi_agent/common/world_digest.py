"""Canonical content identity of a serialized native world."""

import json
from typing import Any

from common.hashing import bytes_sha256


def world_data(world) -> dict[str, Any] | None:
    """Unsupported serialization cannot supply arrival-check provenance."""
    try:
        data = json.loads(world.json())
    except (AttributeError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def world_digest(world) -> str | None:
    data = world_data(world)
    return data_digest(data) if data is not None else None


def data_digest(data: dict[str, Any]) -> str:
    return bytes_sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
