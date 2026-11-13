"""Tests for the pass@k benchmark driver's grid, argv and comparison table."""

import json
import sys
from pathlib import Path

import pytest

import scripts.attempts_benchmark as benchmark
from evaluation.attempts import AttemptRecord, write_attempts_raw
from scripts.attempts_benchmark import (
    BenchmarkCell,
    build_cells,
    checkpoint_slugs,
    collect_cell,
    evaluate_argv,
    pass_at,
    render_comparison,
)


def _attempt(seed: int, attempt: int, *, end_reason: str) -> AttemptRecord:
    return AttemptRecord(
        seed=seed,
        attempt=attempt,
        solved=end_reason == "success",
        end_reason=end_reason,
        max_dev=50.0,
        init_dev_per_ac=300.0,
        final_dev_per_ac=40.0,
        infr_severity=2.0,
        ep_return=1.0,
        steps=40,
        nz_actions=12,
        nz_ex_trb=6,
        spd=3,
        aircraft=10,
    )


def _write_successful_cell_artifacts(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_attempts_raw(
        out_dir / "attempts_raw.csv",
        [_attempt(1, 1, end_reason="success")],
    )
    (out_dir / "attempts_summary.json").write_text(
        json.dumps(
            {
                "attempt_cap": 1,
                "pass_at_k": [1.0],
                "pass_at_cap": 1.0,
                "censored": 0,
                "deterministic_solved_rate": 1.0,
                "mean_attempts_to_success": 1.0,
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_slugs_use_directory_names_when_unique() -> None:
    slugs = checkpoint_slugs([Path("good/f51a9/f51a9_3_000054"), Path("good/898b8/898b8_1_000137")])

    assert slugs == ["f51a9_3_000054", "898b8_1_000137"]


def test_checkpoint_slugs_qualify_unique_parent_names() -> None:
    """Mixed qualified and bare names in one output tree are hard to read."""
    slugs = checkpoint_slugs(
        [Path("runs/trial_1/checkpoint_000060"), Path("runs/trial_3/checkpoint_000060")]
    )

    assert slugs == ["trial_1_checkpoint_000060", "trial_3_checkpoint_000060"]


def test_checkpoint_slugs_reject_ambiguous_qualified_names() -> None:
    checkpoints = [
        Path("runs_a/f51a9/checkpoint_000060"),
        Path("runs_b/f51a9/checkpoint_000060"),
    ]

    with pytest.raises(ValueError, match="checkpoint output slug collision") as raised:
        build_cells(checkpoints, [10], out_root=Path("/tmp/bench"))

    message = str(raised.value)
    assert all(str(checkpoint) in message for checkpoint in checkpoints)


def test_build_cells_completes_each_aircraft_count_before_the_next() -> None:
    """An interrupted benchmark should still leave a comparable set."""
    cells = build_cells(
        [Path("/ck/a"), Path("/ck/b")],
        [12, 15],
        out_root=Path("/out"),
    )

    assert [(cell.max_aircraft, cell.checkpoint.name) for cell in cells] == [
        (12, "a"),
        (12, "b"),
        (15, "a"),
        (15, "b"),
    ]
    assert cells[0].out_dir == Path("/out/a12_a")


def test_evaluate_argv_threads_every_shared_setting() -> None:
    cell = BenchmarkCell(checkpoint=Path("/ck/a"), max_aircraft=15, out_dir=Path("/out/a15_a"))

    argv = evaluate_argv(
        cell,
        attempts=20,
        solve_threshold_s=60.0,
        seed_manifest=Path("config/evaluation_seed_manifest.csv"),
        max_steps=128,
        workers=None,
    )

    assert argv[1:3] == ["-m", "scripts.evaluate"]
    assert "--workers" not in argv
    for flag, value in (
        ("-d", "/ck/a"),
        ("--max-aircraft", "15"),
        ("--max-steps", "128"),
        ("--attempts", "20"),
        ("--solve-threshold", "60.0"),
        ("--seed-manifest", "config/evaluation_seed_manifest.csv"),
        ("--attempts-out-dir", "/out/a15_a"),
    ):
        assert argv[argv.index(flag) + 1] == value


def test_evaluate_argv_passes_an_explicit_worker_count() -> None:
    cell = BenchmarkCell(checkpoint=Path("/ck/a"), max_aircraft=12, out_dir=Path("/out/a12_a"))

    argv = evaluate_argv(
        cell,
        attempts=8,
        solve_threshold_s=30.0,
        seed_manifest=Path("seeds.csv"),
        max_steps=128,
        workers=4,
    )

    assert argv[argv.index("--workers") + 1] == "4"


def test_pass_at_returns_none_beyond_the_cap() -> None:
    report = {"pass_at_k": [0.5, 0.75]}

    assert pass_at(report, 1) == 0.5
    assert pass_at(report, 2) == 0.75
    assert pass_at(report, 4) is None


def test_collect_cell_reads_the_stored_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "a10_ck"
    out_dir.mkdir()
    write_attempts_raw(
        out_dir / "attempts_raw.csv",
        [_attempt(1, 1, end_reason="success"), _attempt(1, 2, end_reason="infringement")],
    )
    (out_dir / "attempts_summary.json").write_text(
        json.dumps({"attempt_cap": 2, "pass_at_k": [0.5, 0.5], "censored": 1}), encoding="utf-8"
    )
    cell = BenchmarkCell(checkpoint=Path("/ck/ck"), max_aircraft=10, out_dir=out_dir)

    row = collect_cell(cell)

    assert row["max_aircraft"] == 10
    assert row["report"]["attempt_cap"] == 2
    assert row["attempt_metrics"]["episodes"] == 2
    assert row["attempt_metrics"]["end_reason_share"]["success"] == 0.5


def test_render_comparison_clamps_columns_to_the_cap() -> None:
    rows = [
        {
            "checkpoint": "/ck/f51a9_3_000054",
            "label": "a12_f51a9_3_000054",
            "max_aircraft": 12,
            "report": {
                "attempt_cap": 2,
                "pass_at_k": [0.4, 0.6],
                "pass_at_cap": 0.6,
                "censored": 40,
                "deterministic_solved_rate": 0.42,
                "mean_attempts_to_success": 1.5,
            },
            "attempt_metrics": {
                "episodes": 200,
                "end_reason_share": {"success": 0.4, "infringement": 0.35, "delayed": 0.25},
                "infringement_severity_mean": 3.5,
                "max_dev_mean": 70.5,
                "steps_mean": 44.0,
                "nz_actions_mean": 14.0,
                "nz_ex_trb_mean": 7.0,
            },
        }
    ]

    table = render_comparison(rows)

    header = table.splitlines()[0]
    assert "pass@1" in header and "pass@2" in header
    assert "pass@4" not in header
    assert "| `a12_f51a9_3_000054` | 12 | 42.0% | 40.0% | 60.0% | 40 | 1.50 |" in table


def test_render_comparison_handles_a_baseline_free_run() -> None:
    rows = [
        {
            "checkpoint": "/ck/x",
            "label": "a10_x",
            "max_aircraft": 10,
            "report": {
                "attempt_cap": 1,
                "pass_at_k": [0.2],
                "pass_at_cap": 0.2,
                "censored": 80,
                "deterministic_solved_rate": None,
                "mean_attempts_to_success": None,
            },
            "attempt_metrics": {
                "episodes": 100,
                "end_reason_share": {"delayed": 1.0},
                "infringement_severity_mean": 0.0,
                "max_dev_mean": 90.0,
                "steps_mean": 40.0,
                "nz_actions_mean": 10.0,
                "nz_ex_trb_mean": 5.0,
            },
        }
    ]

    table = render_comparison(rows)

    assert "| `a10_x` | 10 | - | 20.0% |" in table
    assert "| `a10_x` | 10 | - | 20.0% | 80 | n/a |" in table


def test_render_comparison_reports_an_empty_grid() -> None:
    assert render_comparison([]) == "No completed benchmark runs.\n"


def test_render_comparison_uses_qualified_labels_for_colliding_paths() -> None:
    def row(checkpoint: str, label: str) -> dict[str, object]:
        return {
            "checkpoint": checkpoint,
            "label": label,
            "max_aircraft": 10,
            "report": {
                "attempt_cap": 1,
                "pass_at_k": [0.5],
                "pass_at_cap": 0.5,
                "censored": 1,
                "deterministic_solved_rate": 0.5,
                "mean_attempts_to_success": 1.0,
            },
            "attempt_metrics": {
                "episodes": 2,
                "end_reason_share": {"success": 0.5, "infringement": 0.5},
                "infringement_severity_mean": 1.0,
                "max_dev_mean": 50.0,
                "nz_actions_mean": 2.0,
                "nz_ex_trb_mean": 1.0,
            },
        }

    table = render_comparison(
        [
            row(
                "/runs_a/f51a9/checkpoint_000060",
                "a10_f51a9_checkpoint_000060",
            ),
            row(
                "/runs_b/f51a9/checkpoint_000060",
                "a10_f51a9_checkpoint_000060_2",
            ),
        ]
    )

    assert "`a10_f51a9_checkpoint_000060`" in table
    assert "`a10_f51a9_checkpoint_000060_2`" in table
    assert "`checkpoint_000060`" not in table


def test_main_publishes_incomplete_grid_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    manifest = tmp_path / "seeds.csv"
    manifest.write_text("seed\n1\n", encoding="utf-8")
    out_root = tmp_path / "benchmark"

    def fake_run_cell(argv: list[str], *, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(argv[argv.index("-d") + 1])
        if checkpoint == bad:
            log_path.write_text("failed", encoding="utf-8")
            return 7
        log_path.write_text("completed", encoding="utf-8")
        _write_successful_cell_artifacts(log_path.parent)
        return 0

    monkeypatch.setattr(benchmark, "force_runtime_log_color", lambda: None)
    monkeypatch.setattr(benchmark, "run_cell", fake_run_cell)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attempts_benchmark",
            "-d",
            str(good),
            "-d",
            str(bad),
            "-a",
            "10",
            "--attempts",
            "1",
            "--seed-manifest",
            str(manifest),
            "-o",
            str(out_root),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.main()

    assert raised.value.code == 1
    markdown = (out_root / "benchmark_comparison.md").read_text(encoding="utf-8")
    assert "INCOMPLETE GRID: 1/2 cells completed." in markdown
    assert "- `a10_bad` failed (exit code 7)." in markdown
    published = json.loads((out_root / "benchmark_comparison.json").read_text(encoding="utf-8"))
    assert published["cells_attempted"] == 2
    assert published["cells_completed"] == 1
    assert published["failures"] == [
        {
            "checkpoint": str(bad),
            "exit_code": 7,
            "label": "a10_bad",
            "max_aircraft": 10,
        }
    ]
    assert [row["label"] for row in published["rows"]] == ["a10_good"]


def test_main_treats_malformed_zero_exit_summary_as_failed_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "seeds.csv"
    manifest.write_text("seed\n1\n", encoding="utf-8")
    out_root = tmp_path / "benchmark"

    def fake_run_cell(argv: list[str], *, log_path: Path) -> int:
        del argv
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("completed", encoding="utf-8")
        (log_path.parent / "attempts_summary.json").write_text("{not valid json", encoding="utf-8")
        return 0

    monkeypatch.setattr(benchmark, "force_runtime_log_color", lambda: None)
    monkeypatch.setattr(benchmark, "run_cell", fake_run_cell)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attempts_benchmark",
            "-d",
            str(checkpoint),
            "-a",
            "10",
            "--attempts",
            "1",
            "--seed-manifest",
            str(manifest),
            "-o",
            str(out_root),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.main()

    assert raised.value.code == 1
    published = json.loads((out_root / "benchmark_comparison.json").read_text(encoding="utf-8"))
    assert published["cells_attempted"] == 1
    assert published["cells_completed"] == 0
    assert published["failures"][0]["exit_code"] == 0
    assert "JSONDecodeError" in published["failures"][0]["error"]
