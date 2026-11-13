"""Fresh runs prepare current derived geometry; refused requests have no side effects."""

import json

import pytest


class _BeforeActors(RuntimeError):
    """Stop the lifecycle at its actor construction boundary."""


def _catalog_fixture(tmp_path, monkeypatch):
    from atc_env import sector_catalog as builder

    source = tmp_path / "sources"
    source.mkdir()
    csv = source / "sector.csv"
    csv.write_text("poly_id,Lat_dd,Lon_dd\nTEST,45.0,9.0\nTEST,45.1,9.0\nTEST,45.0,9.1\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"sectors": {"stale": {}}, "meta": {}}))
    monkeypatch.setattr(builder, "_CATALOG_SOURCE_DIR", source)
    monkeypatch.setattr(builder, "_CATALOG_OUTPUT_PATH", catalog)
    return builder, csv, catalog


def test_fresh_rebuilds_stale_geometry_before_actor_construction(
    native_run, config_path, tmp_path, monkeypatch
):
    from training import lifecycle
    from training.lifecycle import RunRequest

    builder, csv, catalog = _catalog_fixture(tmp_path, monkeypatch)
    before = builder.capture_sector_catalog_identity()
    csv.write_text(csv.read_text().replace("45.1", "45.2"))

    def before_actors(*args, **kwargs):
        current = builder.capture_sector_catalog_identity()
        assert current["sectors_sha256"] != before["sectors_sha256"]
        assert current["source_csv_sha256"] != before["source_csv_sha256"]
        assert set(json.loads(catalog.read_text())["sectors"]) == {"TEST"}
        raise _BeforeActors

    monkeypatch.setattr(lifecycle.tune, "Tuner", before_actors)
    with pytest.raises(_BeforeActors):
        lifecycle.run_training(RunRequest("fresh", config_path, None))


def test_catalog_semantics_ignore_only_volatile_metadata(tmp_path, monkeypatch):
    builder, _, catalog = _catalog_fixture(tmp_path, monkeypatch)
    builder.ensure_sector_catalog_up_to_date()
    first = builder.capture_sector_catalog_identity()
    document = json.loads(catalog.read_text())
    document["meta"] = {"generated_at_utc": "another timestamp", "source_dir": "/another/root"}
    catalog.write_text(json.dumps(document))
    second = builder.capture_sector_catalog_identity()
    assert first["artifact_sha256"] != second["artifact_sha256"]
    assert first["sectors_sha256"] == second["sectors_sha256"]
    assert first["source_csv_sha256"] == second["source_csv_sha256"]


def test_fresh_request_refused_by_native_verification_does_not_prepare_catalog(
    native_run, config_path, tmp_path, monkeypatch
):
    from training import lifecycle
    from training.lifecycle import RunRequest

    _, csv, catalog = _catalog_fixture(tmp_path, monkeypatch)
    csv.write_text(csv.read_text().replace("45.1", "45.2"))
    before = (catalog.read_bytes(), csv.read_bytes())

    def refuse(run):
        raise ValueError("native artifact refused")

    monkeypatch.setattr(lifecycle, "verify_native_conditions", refuse)
    with pytest.raises(ValueError, match="native artifact refused"):
        lifecycle.run_training(RunRequest("fresh", config_path, None))
    assert before == (catalog.read_bytes(), csv.read_bytes())


def test_loaded_catalog_identity_rejects_stale_cached_geometry(tmp_path, monkeypatch):
    from atc_env import sector_selection

    builder, csv, catalog = _catalog_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sector_selection, "CATALOG_PATH", catalog)
    builder.ensure_sector_catalog_up_to_date()
    sector_selection.load_sector_catalog_cached.cache_clear()
    try:
        loaded = sector_selection.load_sector_catalog_cached()
        assert hasattr(sector_selection, "capture_loaded_catalog_identity"), (
            "actual loaded catalog authentication is missing"
        )
        first = sector_selection.capture_loaded_catalog_identity()
        csv.write_text(csv.read_text().replace("45.1", "45.2"))
        builder.build_sector_catalog(csv.parent, catalog)
        assert sector_selection.load_sector_catalog_cached() is loaded
        with pytest.raises(ValueError, match="catalog"):
            sector_selection.capture_loaded_catalog_identity()
        sector_selection.load_sector_catalog_cached.cache_clear()
        second = sector_selection.capture_loaded_catalog_identity()
        assert first["sectors_sha256"] != second["sectors_sha256"]
    finally:
        sector_selection.clear_sector_selection_caches()
