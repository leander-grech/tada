"""Shared schema declarations for the normalized sector catalog."""

from __future__ import annotations

from typing import TypedDict

PolygonXY = list[list[float]]


class SectorDef(TypedDict):
    polygon: PolygonXY
    bbox: list[float]


class SectorCatalog(TypedDict):
    sectors: dict[str, SectorDef]
    meta: dict[str, object]
