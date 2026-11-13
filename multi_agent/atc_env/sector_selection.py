"""Sector-based aircraft grouping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Collection, Mapping, Sequence, cast

from atc_env.sector_catalog_schema import PolygonXY, SectorCatalog, SectorDef
from common.hashing import content_digest
from common.jsonio import read_json

_REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = _REPO_ROOT / "static/sector_catalog/tada_sector_catalog.json"
_EPSILON = 1e-9
_LOADED_CATALOG_IDENTITY: dict = {}

PointXY = tuple[float, float]
SectorGroup = tuple[str, ...]


@dataclass
class SelectedGroupIndex:
    """Pre-validated selection index for fast per-step assignment."""

    selected_groups: tuple[SectorGroup, ...]
    sector_to_group: dict[str, SectorGroup]
    sector_defs: dict[str, SectorDef]
    sector_order: tuple[str, ...]


def load_sector_catalog() -> SectorCatalog:
    """Load the normalized sector catalog from the fixed static path."""
    _ensure_catalog_exists()
    return read_json(CATALOG_PATH)


def _ensure_catalog_exists() -> None:
    if CATALOG_PATH.exists():
        return

    from atc_env.sector_catalog import ensure_sector_catalog_up_to_date

    ensure_sector_catalog_up_to_date()
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Sector catalog not found: {CATALOG_PATH}")


@lru_cache(maxsize=1)
def load_sector_catalog_cached() -> SectorCatalog:
    """Load and memoize the sector catalog once per process."""
    from atc_env.sector_catalog import capture_sector_catalog_identity

    catalog = load_sector_catalog()
    _LOADED_CATALOG_IDENTITY.clear()
    _LOADED_CATALOG_IDENTITY.update(capture_sector_catalog_identity())
    return catalog


def capture_loaded_catalog_identity() -> dict:
    """Authenticate cached geometry and the CSV inputs observed when it was loaded."""
    from atc_env.sector_catalog import capture_sector_catalog_identity

    catalog = load_sector_catalog_cached()
    loaded = _LOADED_CATALOG_IDENTITY
    current = capture_sector_catalog_identity()
    geometry = content_digest(catalog["sectors"])
    if geometry != loaded["sectors_sha256"] or any(
        loaded[key] != current[key] for key in ("sectors_sha256", "source_csv_sha256")
    ):
        raise ValueError("loaded sector catalog differs from authenticated current inputs")
    return {"sectors_sha256": geometry, "source_csv_sha256": loaded["source_csv_sha256"]}


def _normalize_bbox(
    polygon: PolygonXY, bbox: Sequence[float] | None
) -> tuple[float, float, float, float]:
    if bbox is not None and len(bbox) == 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _coerce_polygon(value: object, sector_name: str) -> PolygonXY:
    if not isinstance(value, list):
        raise ValueError(f"Sector polygon must contain at least 3 points: {sector_name}")

    points: PolygonXY = []
    for point in value:
        if (not isinstance(point, list) and not isinstance(point, tuple)) or len(point) < 2:
            raise ValueError(f"Sector polygon contains invalid vertex: {sector_name}")
        x_raw = point[0]
        y_raw = point[1]
        if not isinstance(x_raw, int | float) or not isinstance(y_raw, int | float):
            raise ValueError(f"Sector polygon contains non-numeric vertex: {sector_name}")
        points.append([float(x_raw), float(y_raw)])

    if len(points) < 3:
        raise ValueError(f"Sector polygon must contain at least 3 points: {sector_name}")
    return points


def _coerce_bbox(value: object) -> Sequence[float] | None:
    if not isinstance(value, list):
        return None
    if len(value) != 4:
        return None
    bbox_values: list[float] = []
    for raw in value:
        if not isinstance(raw, int | float):
            return None
        bbox_values.append(float(raw))
    return bbox_values


def _normalize_sector_group(raw_group: object, group_idx: int) -> SectorGroup:
    if isinstance(raw_group, str):
        raise ValueError(
            f"SECTOR_SELECTION_GROUPS[{group_idx}] must be an iterable of sector names, not a string."
        )
    if not isinstance(raw_group, list | tuple | set | frozenset):
        raise ValueError(
            f"SECTOR_SELECTION_GROUPS[{group_idx}] must be a list/tuple/set of sector names."
        )

    raw_items = list(raw_group)
    if any(not isinstance(item, str) for item in raw_items):
        raise ValueError(f"SECTOR_SELECTION_GROUPS[{group_idx}] entries must be strings.")
    raw_sectors = sorted(raw_items) if isinstance(raw_group, set | frozenset) else raw_items

    seen: set[str] = set()
    normalized: list[str] = []
    for raw_sector in raw_sectors:
        sector_name = raw_sector.strip()
        if not sector_name:
            continue
        if sector_name in seen:
            raise ValueError(
                f"Duplicate sector '{sector_name}' inside SECTOR_SELECTION_GROUPS[{group_idx}]."
            )
        seen.add(sector_name)
        normalized.append(sector_name)

    if not normalized:
        raise ValueError(f"SECTOR_SELECTION_GROUPS[{group_idx}] must contain at least one sector.")
    return tuple(normalized)


def normalize_selected_sector_groups(
    selected_groups: Sequence[Collection[str]],
) -> tuple[SectorGroup, ...]:
    if not selected_groups:
        raise ValueError("Selected sector groups must be non-empty for sector-based usage")

    return tuple(
        _normalize_sector_group(cast(object, raw_group), group_idx=group_idx)
        for group_idx, raw_group in enumerate(selected_groups)
    )


def build_selected_group_index(
    catalog: Mapping[str, object], selected_groups: Sequence[Collection[str]]
) -> SelectedGroupIndex:
    """Validate selected sector groups and build immutable lookup structures."""
    normalized_groups = normalize_selected_sector_groups(selected_groups)
    return _build_selected_group_index_from_normalized(catalog, normalized_groups)


def _build_selected_group_index_from_normalized(
    catalog: Mapping[str, object], normalized_groups: tuple[SectorGroup, ...]
) -> SelectedGroupIndex:
    """Build selected-group index from normalized sector groups."""

    raw_sectors = catalog.get("sectors")
    if not isinstance(raw_sectors, dict):
        raise ValueError("Sector catalog is missing required sectors section")

    sector_defs: dict[str, SectorDef] = {}
    sector_to_group: dict[str, SectorGroup] = {}
    sectors_map = cast(Mapping[str, object], raw_sectors)
    for group in normalized_groups:
        group_label = " + ".join(group)
        for sector_name in group:
            if sector_name in sector_to_group:
                overlap_with = " + ".join(sector_to_group[sector_name])
                raise ValueError(
                    f"Selected sector groups overlap on sector '{sector_name}' "
                    f"({overlap_with} vs {group_label})"
                )

            sector_def_obj = sectors_map.get(sector_name)
            if not isinstance(sector_def_obj, dict):
                raise ValueError(f"Selected sector group references unknown sector: {sector_name}")
            sector_def_mapping = cast(Mapping[str, object], sector_def_obj)

            polygon = _coerce_polygon(sector_def_mapping.get("polygon"), sector_name=sector_name)
            bbox_raw = _coerce_bbox(sector_def_mapping.get("bbox"))

            bbox = _normalize_bbox(
                polygon=polygon,
                bbox=bbox_raw,
            )

            sector_defs[sector_name] = {
                "polygon": polygon,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            }
            sector_to_group[sector_name] = group

    return SelectedGroupIndex(
        selected_groups=normalized_groups,
        sector_to_group=sector_to_group,
        sector_defs=sector_defs,
        # Later configured groups take precedence in overlapping areas.
        sector_order=tuple(
            sector_name for group in reversed(normalized_groups) for sector_name in group
        ),
    )


@lru_cache(maxsize=64)
def _get_selected_group_index_cached(
    selected_groups: tuple[SectorGroup, ...],
) -> SelectedGroupIndex:
    return _build_selected_group_index_from_normalized(
        load_sector_catalog_cached(), selected_groups
    )


def get_selected_group_index_cached_from_normalized(
    selected_groups: tuple[SectorGroup, ...],
) -> SelectedGroupIndex:
    """Build and memoize selected-group index from already normalized groups."""
    if not selected_groups:
        raise ValueError("Selected sector groups must be non-empty for sector-based usage")
    return _get_selected_group_index_cached(selected_groups)


def clear_sector_selection_caches() -> None:
    """Clear process-local catalog and selected-group cache entries."""
    load_sector_catalog_cached.cache_clear()
    _get_selected_group_index_cached.cache_clear()


def _point_on_segment(point: PointXY, a: PointXY, b: PointXY) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b

    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > _EPSILON:
        return False

    min_x = min(ax, bx) - _EPSILON
    max_x = max(ax, bx) + _EPSILON
    min_y = min(ay, by) - _EPSILON
    max_y = max(ay, by) + _EPSILON
    return min_x <= px <= max_x and min_y <= py <= max_y


def _contains_point_with_boundary(
    point_xy: PointXY, polygon_xy: PolygonXY, *, include_boundary: bool
) -> bool:
    if len(polygon_xy) < 3:
        return False

    inside = False
    px, py = float(point_xy[0]), float(point_xy[1])

    for idx in range(len(polygon_xy)):
        x1, y1 = float(polygon_xy[idx][0]), float(polygon_xy[idx][1])
        x2, y2 = (
            float(polygon_xy[(idx + 1) % len(polygon_xy)][0]),
            float(polygon_xy[(idx + 1) % len(polygon_xy)][1]),
        )

        if _point_on_segment((px, py), (x1, y1), (x2, y2)):
            return include_boundary

        edge_crosses_ray = (y1 > py) != (y2 > py)
        if not edge_crosses_ray:
            continue

        x_at_py = x1 + (x2 - x1) * (py - y1) / (y2 - y1)
        if abs(x_at_py - px) <= _EPSILON:
            return include_boundary
        if x_at_py > px:
            inside = not inside

    return inside


def contains_point(point_xy: PointXY, polygon_xy: PolygonXY) -> bool:
    """Return True when a point is inside or exactly on the polygon boundary."""
    return _contains_point_with_boundary(point_xy, polygon_xy, include_boundary=True)


def _point_in_bbox(point_xy: PointXY, bbox: Sequence[float]) -> bool:
    px, py = float(point_xy[0]), float(point_xy[1])
    min_x, min_y, max_x, max_y = [float(value) for value in bbox]
    return (min_x - _EPSILON) <= px <= (max_x + _EPSILON) and (min_y - _EPSILON) <= py <= (
        max_y + _EPSILON
    )


def sector_for_point(point_xy: PointXY, index: SelectedGroupIndex) -> str | None:
    """Resolve point to a selected sector using bbox prefilter and polygon containment."""
    for sector_name in index.sector_order:
        sector_def = index.sector_defs[sector_name]
        bbox = sector_def["bbox"]
        if not _point_in_bbox(point_xy, bbox):
            continue

        polygon = sector_def["polygon"]
        if contains_point(point_xy, polygon):
            return sector_name

    return None


def assign_aircraft_to_selected_groups(
    aircraft_positions: Mapping[str, PointXY],
    index: SelectedGroupIndex,
) -> dict[SectorGroup, list[str]]:
    """Assign aircraft to selected groups deterministically via sector containment."""
    grouped: dict[SectorGroup, list[str]] = {group: [] for group in index.selected_groups}

    for callsign in sorted(aircraft_positions.keys()):
        sector_name = sector_for_point(aircraft_positions[callsign], index)
        if sector_name is None:
            continue

        grouped[index.sector_to_group[sector_name]].append(callsign)

    return {group: callsigns for group, callsigns in grouped.items() if callsigns}
