"""Tests for sector catalog builder."""

import json
from pathlib import Path

import pytest

from atc_env.sector_catalog import build_sector_catalog


def _equirectangular_local_nm(
    origin_lat: float, origin_lon: float, lat: float, lon: float
) -> tuple[float, float]:
    """Independent approximation for expected local NM coordinates."""
    import math

    x_nm = (lon - origin_lon) * 60.0 * math.cos(math.radians(origin_lat))
    y_nm = (lat - origin_lat) * 60.0
    return (x_nm, y_nm)


def test_build_sector_catalog_computes_bbox_from_polygon_vertices(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "TEST.csv").write_text(
        "poly_id,point_order,Lat_dd,Lon_dd\n"
        "TEST,1,45.438334,9.279015\n"
        "TEST,2,45.438334,9.379015\n"
        "TEST,3,45.338334,9.379015\n"
        "TEST,4,45.338334,9.279015\n"
        "TEST,5,45.438334,9.279015\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "catalog.json"

    build_sector_catalog(source_dir=source_dir, output_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    sector = payload["sectors"]["TEST"]
    polygon = sector["polygon"]
    expected_polygon = [
        _equirectangular_local_nm(45.438334, 9.279015, 45.438334, 9.279015),
        _equirectangular_local_nm(45.438334, 9.279015, 45.438334, 9.379015),
        _equirectangular_local_nm(45.438334, 9.279015, 45.338334, 9.379015),
        _equirectangular_local_nm(45.438334, 9.279015, 45.338334, 9.279015),
        _equirectangular_local_nm(45.438334, 9.279015, 45.438334, 9.279015),
    ]

    assert len(polygon) == len(expected_polygon)
    for built_point, expected_point in zip(polygon, expected_polygon, strict=True):
        assert built_point[0] == pytest.approx(expected_point[0], abs=0.03)
        assert built_point[1] == pytest.approx(expected_point[1], abs=0.03)

    expected_bbox = [
        min(point[0] for point in expected_polygon),
        min(point[1] for point in expected_polygon),
        max(point[0] for point in expected_polygon),
        max(point[1] for point in expected_polygon),
    ]
    assert sector["bbox"] == pytest.approx(expected_bbox, abs=0.03)


def test_build_sector_catalog_respects_point_order_when_csv_rows_are_shuffled(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "TEST.csv").write_text(
        "poly_id,point_order,Lat_dd,Lon_dd\n"
        "TEST,2,45.438334,9.379015\n"
        "TEST,1,45.438334,9.279015\n"
        "TEST,3,45.338334,9.379015\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "catalog.json"

    build_sector_catalog(source_dir=source_dir, output_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    polygon = payload["sectors"]["TEST"]["polygon"]
    expected = [
        _equirectangular_local_nm(45.438334, 9.279015, 45.438334, 9.279015),
        _equirectangular_local_nm(45.438334, 9.279015, 45.438334, 9.379015),
        _equirectangular_local_nm(45.438334, 9.279015, 45.338334, 9.379015),
    ]
    assert len(polygon) == len(expected)
    for built_point, expected_point in zip(polygon, expected, strict=True):
        assert built_point[0] == pytest.approx(expected_point[0], abs=0.03)
        assert built_point[1] == pytest.approx(expected_point[1], abs=0.03)


def test_build_sector_catalog_ignores_airspace_outline_csv_for_polygon_parsing(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "TEST.csv").write_text(
        "poly_id,point_order,Lat_dd,Lon_dd\n"
        "TEST,1,45.438334,9.279015\n"
        "TEST,2,45.438334,9.379015\n"
        "TEST,3,45.338334,9.379015\n",
        encoding="utf-8",
    )
    # Intentionally present in source snapshots; this file does not include lat/lon columns.
    (source_dir / "airspace_outline.csv").write_text(
        "poly_id,point_start,point_end\nTEST,1,2\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "catalog.json"

    build_sector_catalog(source_dir=source_dir, output_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "TEST" in payload["sectors"]


def test_build_sector_catalog_raises_for_non_excluded_csv_missing_required_headers(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "BAD.csv").write_text(
        "poly_id,point_start,point_end\nTEST,1,2\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "catalog.json"

    with pytest.raises(ValueError, match="Missing required headers"):
        build_sector_catalog(source_dir=source_dir, output_path=output_path)


def test_build_sector_catalog_raises_for_non_integer_point_order(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "TEST.csv").write_text(
        "poly_id,point_order,Lat_dd,Lon_dd\n"
        "TEST,1.5,45.438334,9.279015\n"
        "TEST,2,45.438334,9.379015\n"
        "TEST,3,45.338334,9.379015\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "catalog.json"

    with pytest.raises(ValueError, match="Non-integer point_order"):
        build_sector_catalog(source_dir=source_dir, output_path=output_path)


def test_build_sector_catalog_raises_for_polygon_with_too_few_points(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "TEST.csv").write_text(
        "poly_id,point_order,Lat_dd,Lon_dd\nTEST,1,45.438334,9.279015\nTEST,2,45.438334,9.379015\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "catalog.json"

    with pytest.raises(ValueError, match="at least 3 points"):
        build_sector_catalog(source_dir=source_dir, output_path=output_path)
