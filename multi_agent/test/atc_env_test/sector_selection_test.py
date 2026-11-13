"""Tests for sector selection catalog validation and geometry assignment."""

import json
from pathlib import Path

import pytest

import atc_env.sector_selection as sector_selection_module
from atc_env.sector_selection import (
    SectorCatalog,
    assign_aircraft_to_selected_groups,
    build_selected_group_index,
    clear_sector_selection_caches,
    contains_point,
    load_sector_catalog,
)
from config.config import Config


def _catalog_with_non_overlapping_roles() -> SectorCatalog:
    return {
        "sectors": {
            "ALPHA": {
                "polygon": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                "bbox": [0.0, 0.0, 10.0, 10.0],
            },
            "BRAVO": {
                "polygon": [[10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]],
                "bbox": [10.0, 0.0, 20.0, 10.0],
            },
        },
        "meta": {},
    }


@pytest.mark.parametrize(
    ("selected_groups", "error_match"),
    [
        ([["DOES_NOT_EXIST"]], "unknown sector"),
        ([["ALPHA"], ["ALPHA", "BRAVO"]], "overlap on sector"),
        ([["ALPHA", "ALPHA"]], "Duplicate sector"),
        ([[" ", ""]], "must contain at least one sector"),
        ([[123]], "entries must be strings"),
    ],
    ids=[
        "unknown_sector",
        "duplicate_across_groups",
        "duplicate_within_group",
        "whitespace_group",
        "non_string_entry",
    ],
)
def test_build_selected_group_index_rejects_invalid_group_configurations(
    selected_groups: list[list[str]],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        build_selected_group_index(
            catalog=_catalog_with_non_overlapping_roles(),
            selected_groups=selected_groups,
        )


def test_set_group_input_is_normalized_deterministically() -> None:
    index = build_selected_group_index(
        catalog=_catalog_with_non_overlapping_roles(),
        selected_groups=[{"BRAVO", "ALPHA"}],
    )

    assert index.selected_groups == (("ALPHA", "BRAVO"),)


def test_boundary_assignment_is_deterministic_by_precedence_order() -> None:
    catalog = _catalog_with_non_overlapping_roles()
    index = build_selected_group_index(catalog=catalog, selected_groups=[["ALPHA"], ["BRAVO"]])

    assert contains_point((10.0, 5.0), catalog["sectors"]["ALPHA"]["polygon"])
    assert contains_point((10.0, 5.0), catalog["sectors"]["BRAVO"]["polygon"])

    assignments = assign_aircraft_to_selected_groups(
        aircraft_positions={
            "AC-1": (1.0, 1.0),
            "AC-2": (15.0, 5.0),
            "AC-3": (10.0, 5.0),
            "AC-4": (99.0, 99.0),
        },
        index=index,
    )

    assert assignments == {
        ("BRAVO",): ["AC-2", "AC-3"],
        ("ALPHA",): ["AC-1"],
    }


def test_multi_sector_group_assignment_collects_aircraft_from_all_group_sectors() -> None:
    index = build_selected_group_index(
        catalog=_catalog_with_non_overlapping_roles(),
        selected_groups=[["ALPHA", "BRAVO"]],
    )

    assignments = assign_aircraft_to_selected_groups(
        aircraft_positions={
            "AC-1": (1.0, 1.0),
            "AC-2": (15.0, 5.0),
            "AC-3": (99.0, 99.0),
        },
        index=index,
    )

    assert assignments == {
        ("ALPHA", "BRAVO"): ["AC-1", "AC-2"],
    }


def test_load_catalog_reads_static_sector_catalog_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "static" / "sector_catalog" / "tada_sector_catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    expected = _catalog_with_non_overlapping_roles()
    catalog_path.write_text(json.dumps(expected), encoding="utf-8")

    monkeypatch.setattr(sector_selection_module, "CATALOG_PATH", catalog_path)
    clear_sector_selection_caches()

    loaded = load_sector_catalog()

    assert loaded == expected


def test_load_catalog_ensures_catalog_exists_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "static" / "sector_catalog" / "tada_sector_catalog.json"
    expected = _catalog_with_non_overlapping_roles()

    def fake_ensure() -> None:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(expected), encoding="utf-8")

    monkeypatch.setattr(sector_selection_module, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(sector_selection_module, "_ensure_catalog_exists", fake_ensure)
    clear_sector_selection_caches()

    loaded = load_sector_catalog()

    assert loaded == expected


def test_default_sector_selection_groups_builds_valid_index() -> None:
    catalog = load_sector_catalog()
    index = build_selected_group_index(
        catalog=catalog, selected_groups=Config.SECTOR_SELECTION_GROUPS
    )
    assert index.selected_groups
