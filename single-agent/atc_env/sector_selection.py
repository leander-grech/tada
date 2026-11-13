"""Sector-based aircraft grouping utilities."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from utils.sector_catalog_schema import SectorCatalog
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt


_REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = _REPO_ROOT / "static/sector_catalog/tada_sector_catalog.json"
def resolve_render_sector_polygons():
    """Return sector polygons and colors for rendering."""

    catalog = load_sector_catalog_cached()
    sectors = catalog["sectors"]

    cmap = plt.get_cmap("Set2")

    polygons = []
    colors = []

    for idx, (sector_name, sector_def) in enumerate(sectors.items()):
        polygon = [
            (float(point[0]), float(point[1]))
            for point in sector_def["polygon"]
        ]

        polygons.append(polygon)
        colors.append(to_rgba(cmap(idx % cmap.N)))

    return polygons, colors

def load_sector_catalog() -> SectorCatalog:
    """Load the normalized sector catalog from the fixed static path."""
    _ensure_catalog_exists()
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_catalog_exists() -> None:
    if CATALOG_PATH.exists():
        return

    from utils.sector_catalog_builder import ensure_sector_catalog_up_to_date

    ensure_sector_catalog_up_to_date()
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Sector catalog not found: {CATALOG_PATH}")


@lru_cache(maxsize=1)
def load_sector_catalog_cached() -> SectorCatalog:
    """Load and memoize the sector catalog once per process."""
    return load_sector_catalog()


def clear_sector_selection_caches() -> None:
    """Clear the process-local catalog cache."""
    load_sector_catalog_cached.cache_clear()
