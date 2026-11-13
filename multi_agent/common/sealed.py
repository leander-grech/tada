"""Consumer side of a sealed run-config payload."""

from collections.abc import Mapping
from typing import Any


def consume(payload: Mapping[str, Any], expected: str, *, role: str) -> str:
    """Consumers only compare the digest the driver sealed; tokens carry the semantics."""
    if payload["role"] != role or payload["semantic_digest"] != expected:
        raise ValueError(f"{role} payload semantic_digest does not match the sealed run config")
    return expected
