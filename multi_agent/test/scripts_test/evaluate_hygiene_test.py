"""Tests for offline evaluation hygiene helpers."""

import json

import pytest

from evaluation.run import (
    build_summary,
    load_seed_manifest,
    resolve_attempt_workers,
    resolve_solve_threshold,
)
from scripts.evaluate import _print_attempts_report


def test_load_seed_manifest_accepts_json_list(tmp_path) -> None:
    manifest = tmp_path / "seeds.json"
    manifest.write_text(json.dumps([11, "22", 33]), encoding="utf-8")

    assert load_seed_manifest(manifest) == [11, 22, 33]


def test_load_seed_manifest_accepts_newline_text(tmp_path) -> None:
    manifest = tmp_path / "seeds.txt"
    manifest.write_text("11\n# held-out validation\n22\n\n33\n", encoding="utf-8")

    assert load_seed_manifest(manifest) == [11, 22, 33]


def test_resolve_solve_threshold_accepts_a_positive_window() -> None:
    assert resolve_solve_threshold(60) == 60.0


def test_resolve_solve_threshold_rejects_non_positive_windows() -> None:
    for window in (0.0, -30.0):
        with pytest.raises(ValueError, match="positive"):
            resolve_solve_threshold(window)


def test_resolve_solve_threshold_rejects_non_finite_windows() -> None:
    """`nan <= 0` is False, so a bare sign check would grade nothing as solved."""
    for window in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            resolve_solve_threshold(window)


def test_load_seed_manifest_accepts_csv_seed_column(tmp_path) -> None:
    manifest = tmp_path / "attempts_per_seed.csv"
    manifest.write_text(
        "seed,solved,attempts_used\n11,True,1\n22,False,8\n33,True,2\n",
        encoding="utf-8",
    )

    assert load_seed_manifest(manifest) == [11, 22, 33]


def test_load_seed_manifest_reads_seed_column_by_name(tmp_path) -> None:
    manifest = tmp_path / "report.csv"
    manifest.write_text("attempt,seed\n1,11\n1,22\n", encoding="utf-8")

    assert load_seed_manifest(manifest) == [11, 22]


def test_load_seed_manifest_accepts_quoted_csv_fields(tmp_path) -> None:
    manifest = tmp_path / "quoted.csv"
    manifest.write_text('source,seed\n"top,up",11\n', encoding="utf-8")

    assert load_seed_manifest(manifest) == [11]


def test_load_seed_manifest_rejects_malformed_csv_rows(tmp_path) -> None:
    manifest = tmp_path / "malformed.csv"
    manifest.write_text("source,seed\n11\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid"):
        load_seed_manifest(manifest)


def test_load_seed_manifest_rejects_duplicates(tmp_path) -> None:
    manifest = tmp_path / "seeds.json"
    manifest.write_text(json.dumps([11, 22, 11]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_seed_manifest(manifest)


def test_build_summary_reports_selection_axes() -> None:
    scenarios = [
        {
            "seed": 1,
            "outcome": "success",
            "init_dev": 20.0,
            "final_dev": 4.0,
            "forecast_infringement_severity": 0.0,
            "steps": 10,
            "nz_actions": 5,
            "action_counts": {"learned_command_count": 5},
            "nz_ex_trb": 3,
            "spd": 2,
            "aircraft": 12,
        },
        {
            "seed": 2,
            "outcome": "delayed",
            "init_dev": 30.0,
            "final_dev": 20.0,
            "forecast_infringement_severity": 0.0,
            "steps": 20,
            "nz_actions": 1,
            "action_counts": {"learned_command_count": 1},
            "nz_ex_trb": 1,
            "spd": 1,
            "aircraft": 12,
        },
        {
            "seed": 3,
            "outcome": "infringement",
            "init_dev": 30.0,
            "final_dev": 10.0,
            "forecast_infringement_severity": 2.0,
            "steps": 15,
            "nz_actions": 6,
            "action_counts": {"learned_command_count": 6},
            "nz_ex_trb": 4,
            "spd": 1,
            "aircraft": 12,
        },
    ]

    summary = build_summary(
        scenarios,
        sub_episode_count=1,
        checkpoint="/tmp/checkpoint",
        seed_manifest=[1, 2, 3],
        solve_threshold_s=60.0,
    )

    assert summary["num_scenarios"] == 3
    assert summary["solve_threshold_s"] == 60.0
    assert summary["seed_manifest"] == [1, 2, 3]
    assert summary["scenario_seeds"] == [1, 2, 3]
    assert summary["outcomes"]["infringement"]["rate"] == 1 / 3
    assert summary["selection_axes"]["success_rate"] == 1 / 3
    assert summary["selection_axes"]["infringement_rate"] == 1 / 3
    assert summary["selection_axes"]["delayed_rate"] == 1 / 3
    assert summary["diagnostics"]["non_trombone_command_mean"] == 8 / 3
    assert summary["selection_axes"]["learned_command_mean"] == 4
    assert "forecast_infringement_severity_mean" not in summary["selection_axes"]
    assert "infringement_mean" not in summary["non_dominated_axes"]
    assert summary["selection_axes"]["speed_action_mean"] == 4 / 3


def test_resolve_attempt_workers_defaults_to_available_scenarios() -> None:
    assert resolve_attempt_workers(None, 3, configured_workers=8) == 3


def test_resolve_attempt_workers_accepts_explicit_count() -> None:
    assert resolve_attempt_workers(2, 3, configured_workers=8) == 2


def test_resolve_attempt_workers_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        resolve_attempt_workers(4, 3, configured_workers=8)

    with pytest.raises(ValueError, match="positive"):
        resolve_attempt_workers(0, 3, configured_workers=8)


def test_print_attempts_report_renders_missing_mean_as_na(capsys) -> None:
    report = {
        "scenarios": 1,
        "attempt_cap": 1,
        "episodes_run": 1,
        "solve_threshold_s": 60.0,
        "deterministic_solved_rate": None,
        "pass_at_1": 0.0,
        "pass_at_cap": 0.0,
        "censored": 1,
        "mean_attempts_to_success": None,
        "pass_at_k": [0.0],
    }

    _print_attempts_report(report, [])
    output = capsys.readouterr().out

    assert "mean attempts to success" in output
    assert "n/a" in output
