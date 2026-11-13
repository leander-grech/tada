"""Builds a normalized sector catalog from local sector source snapshot files."""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from atc_env.sector_catalog_schema import SectorCatalog, SectorDef
from common.hashing import bytes_sha256, content_digest, file_sha256

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CATALOG_SOURCE_DIR = _REPO_ROOT / "static/sector_source/tada_scenario_sectors"
_CATALOG_OUTPUT_PATH = _REPO_ROOT / "static/sector_catalog/tada_sector_catalog.json"
_LIML_LAT = 45.438334
_LIML_LON = 9.279015
_EARTH_RADIUS_M = 6_371_000.0
_METERS_PER_NM = 1852.0


def ensure_sector_catalog_up_to_date() -> Path:
    """Always rebuild the sector catalog from local source snapshot files."""
    catalog_path = build_sector_catalog(
        source_dir=_CATALOG_SOURCE_DIR, output_path=_CATALOG_OUTPUT_PATH
    )
    _invalidate_sector_selection_cache()
    return catalog_path


def capture_sector_catalog_identity() -> dict:
    """Separate generated artifact bytes from portable geometry and CSV input identity."""
    raw = _CATALOG_OUTPUT_PATH.read_bytes()
    sectors = json.loads(raw)["sectors"]
    return {
        "artifact_path": str(_CATALOG_OUTPUT_PATH.resolve()),
        "artifact_sha256": bytes_sha256(raw),
        "sectors_sha256": content_digest(sectors),
        "source_csv_sha256": {
            path.name: file_sha256(path)
            for path in sorted(_CATALOG_SOURCE_DIR.glob("*.csv"))
            if path.name != "airspace_outline.csv"
        },
    }


def build_sector_catalog(source_dir: Path, output_path: Path) -> Path:
    """Build sector catalog JSON from source files and return the written path."""
    source_dir = source_dir.resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Sector source directory not found: {source_dir}")

    sectors = _parse_all_sector_polygons(source_dir)

    meta: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_dir": (source_dir.relative_to(_REPO_ROOT).as_posix()
                       if source_dir.is_relative_to(_REPO_ROOT) else source_dir.name),
    }
    payload: SectorCatalog = {
        "sectors": sectors,
        "meta": meta,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _parse_all_sector_polygons(
    source_dir: Path,
) -> dict[str, SectorDef]:
    sectors: dict[str, SectorDef] = {}

    for csv_path in sorted(source_dir.glob("*.csv")):
        if csv_path.name == "airspace_outline.csv":
            continue
        csv_polygons = _parse_sector_csv(csv_path)
        for sector_name, polygon in csv_polygons.items():
            if sector_name in sectors:
                raise ValueError(f"Duplicate sector polygon encountered: {sector_name}")

            bbox = _polygon_bbox(polygon)
            sectors[sector_name] = {
                "polygon": [[point[0], point[1]] for point in polygon],
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            }

    if not sectors:
        raise ValueError(f"No sector polygons parsed from {source_dir}")

    return sectors


def _parse_sector_csv(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        required = {"poly_id", "Lat_dd", "Lon_dd"}
        fieldnames = set(reader.fieldnames)
        if not required.issubset(fieldnames):
            missing = sorted(required - fieldnames)
            raise ValueError(f"Missing required headers in {csv_path}: {', '.join(missing)}")

        has_point_order = "point_order" in fieldnames
        polygons_ordered: dict[str, list[tuple[int, tuple[float, float]]]] = {}
        for row_index, row in enumerate(reader):
            name = (row.get("poly_id") or "").strip()
            lat_raw = (row.get("Lat_dd") or "").strip()
            lon_raw = (row.get("Lon_dd") or "").strip()
            if not name or not lat_raw or not lon_raw:
                raise ValueError(f"Malformed sector row in {csv_path}: {row}")

            if has_point_order:
                point_order_raw = (row.get("point_order") or "").strip()
                if not point_order_raw:
                    raise ValueError(f"Missing point_order value in {csv_path}: {row}")
                point_order_float = float(point_order_raw)
                if not point_order_float.is_integer():
                    raise ValueError(f"Non-integer point_order value in {csv_path}: {row}")
                point_order = int(point_order_float)
            else:
                point_order = row_index

            lat = float(lat_raw)
            lon = float(lon_raw)
            polygons_ordered.setdefault(name, []).append(
                (point_order, _haversine_offset_nm(_LIML_LAT, _LIML_LON, lat, lon))
            )

    polygons: dict[str, list[tuple[float, float]]] = {}
    for name, ordered_points in polygons_ordered.items():
        polygon = [point for _, point in sorted(ordered_points, key=lambda item: item[0])]
        if len(polygon) < 3:
            raise ValueError(f"Sector polygon must contain at least 3 points in {csv_path}: {name}")
        polygons[name] = polygon

    return polygons


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (
        math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_M * c


def _js_sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _haversine_offset_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    distance_x_m = _haversine_distance_m(lat1, lon1, lat1, lon2) * _js_sign(lon2 - lon1)
    distance_y_m = _haversine_distance_m(lat1, lon1, lat2, lon1) * _js_sign(lat2 - lat1)
    # Simulator world coordinates use north-positive Y.
    return distance_x_m / _METERS_PER_NM, distance_y_m / _METERS_PER_NM


def _polygon_bbox(points: list[tuple[float, float]]) -> list[float]:
    if not points:
        raise ValueError("Polygon has no points")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _invalidate_sector_selection_cache() -> None:
    """Best-effort cache invalidation for runtime sector selection helpers."""
    try:
        from atc_env.sector_selection import clear_sector_selection_caches
    except ImportError:
        return
    clear_sector_selection_caches()
