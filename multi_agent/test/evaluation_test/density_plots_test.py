"""Tests for the cross-density figure helpers."""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from evaluation.attempts import AttemptRecord, write_attempts_per_seed
from evaluation.report.density_cells import (
    DensityCell,
    action_count_bins,
    crossing_k,
    discover_cells,
    exposure_curves,
    forced_los_curve,
    load_cell,
    parse_cell_name,
)
from evaluation.report.density_plots import (
    density_caption,
    greedy_note,
    render_action_count_density,
    render_exposure_density,
    render_forced_los_density,
    render_passk_density,
)
from evaluation.stats import summarize_seed


def _attempt(
    seed: int,
    *,
    nz_actions: int = 5,
    attempt: int = 1,
    end_reason: str = "success",
) -> AttemptRecord:
    return AttemptRecord(
        seed=seed,
        attempt=attempt,
        solved=end_reason == "success",
        end_reason=end_reason,
        max_dev=10.0,
        init_dev_per_ac=100.0,
        final_dev_per_ac=10.0,
        infr_severity=0.0,
        ep_return=1.0,
        steps=40,
        nz_actions=nz_actions,
        nz_ex_trb=2,
        spd=1,
        aircraft=10,
    )


def _write_cell(
    root: Path,
    name: str,
    *,
    cap: int,
    nz: list[int],
    det: float | None,
    scenarios: int | None = None,
    solve_threshold_s: float = 60.0,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "attempts_summary.json").write_text(
        json.dumps(
            {
                "attempt_cap": cap,
                "scenarios": len(nz) if scenarios is None else scenarios,
                "pass_at_k": [0.5] * cap,
                "deterministic_solved_rate": det,
                "solve_threshold_s": solve_threshold_s,
            }
        ),
        encoding="utf-8",
    )
    fields = [f.name for f in dataclasses.fields(AttemptRecord)]
    with (directory / "attempts_raw.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, actions in enumerate(nz):
            writer.writerow(dataclasses.asdict(_attempt(index, nz_actions=actions)))
    return directory


def _cell(
    aircraft: int,
    nz: list[int],
    *,
    cap: int = 2,
    scenarios: int | None = None,
    solve_threshold_s: float = 60.0,
) -> DensityCell:
    return DensityCell(
        aircraft=aircraft,
        checkpoint="ckpt",
        cap=cap,
        scenarios=len(nz) if scenarios is None else scenarios,
        solve_threshold_s=solve_threshold_s,
        curve=[0.4, 0.8][:cap],
        deterministic_rate=0.5,
        attempts=[_attempt(i, nz_actions=n) for i, n in enumerate(nz)],
    )


def _safety_cell(
    rows: list[tuple[int, int, str]],
    *,
    cap: int,
    scenarios: int,
    solve_threshold_s: float = 60.0,
) -> DensityCell:
    """Build a cell from ``(seed, attempt, end_reason)`` triples."""
    return DensityCell(
        aircraft=10,
        checkpoint="ckpt",
        cap=cap,
        scenarios=scenarios,
        solve_threshold_s=solve_threshold_s,
        curve=[0.0] * cap,
        deterministic_rate=0.5,
        attempts=[
            _attempt(seed, attempt=attempt, end_reason=reason) for seed, attempt, reason in rows
        ],
    )


@pytest.fixture
def saved_figures(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture figures while retaining the real save-to-disk smoke check."""
    from matplotlib.figure import Figure

    figures = []
    original_savefig = Figure.savefig

    def capture(fig, *args, **kwargs):
        figures.append(fig)
        return original_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return figures


def _figure_text(fig) -> list[str]:
    return [
        *(text.get_text() for text in fig.texts),
        *(text.get_text() for ax in fig.axes for text in ax.texts),
        *(
            ax.get_title(loc=location)
            for ax in fig.axes
            for location in ("left", "center", "right")
        ),
    ]


def test_exposure_curves_accumulate_first_solve_and_first_loss() -> None:
    """pass@k counts a scenario from its first solve on; LoS-free drops it from its first loss on."""
    cell = _safety_cell(
        [
            (1, 1, "delayed"),
            (1, 2, "success"),
            (2, 1, "infringement"),
            (2, 2, "success"),
        ],
        cap=3,
        scenarios=2,
    )
    passk, los_free = exposure_curves(cell)
    assert passk == [0.0, 1.0, 1.0]
    assert los_free == [0.5, 0.5, 0.5]


def test_exposure_curves_count_a_repeat_offender_once() -> None:
    """Two separation losses in one scenario are one unsafe scenario, not two."""
    cell = _safety_cell(
        [(1, 1, "infringement"), (1, 2, "infringement"), (2, 1, "success")],
        cap=2,
        scenarios=2,
    )
    _, los_free = exposure_curves(cell)
    assert los_free == [0.5, 0.5]


def test_exposure_curves_ignore_severity_without_a_separation_loss() -> None:
    """Near-conflict severity is not a breach: only the end reason terminates on LoS."""
    cell = _safety_cell([(1, 1, "success"), (2, 1, "delayed")], cap=1, scenarios=2)
    noisy = dataclasses.replace(
        cell,
        attempts=[dataclasses.replace(a, infr_severity=22.03) for a in cell.attempts],
    )

    passk, los_free = exposure_curves(noisy)

    assert passk == [0.5]
    assert los_free == [1.0]


def test_forced_los_curve_retires_a_scenario_at_its_first_clean_attempt() -> None:
    """One attempt that keeps separation is enough: the loop can deliver that one."""
    cell = _safety_cell(
        [(1, 1, "infringement"), (1, 2, "infringement"), (1, 3, "delayed"), (2, 1, "success")],
        cap=3,
        scenarios=2,
    )

    assert forced_los_curve(cell) == [0.5, 0.5, 0.0]


def test_forced_los_curve_never_rises_with_the_budget() -> None:
    """The whole point of the metric: retries can only remove forced losses."""
    cell = _safety_cell(
        [
            (1, 1, "infringement"),
            (1, 2, "success"),
            (2, 1, "infringement"),
            (2, 2, "infringement"),
            (3, 1, "delayed"),
            (3, 2, "infringement"),
        ],
        cap=2,
        scenarios=3,
    )

    curve = forced_los_curve(cell)

    assert curve == pytest.approx([2 / 3, 1 / 3])


def test_forced_los_curve_prefers_a_clean_attempt_over_a_faster_infringing_one() -> None:
    """Safety-first selection: a low-deviation infringement must not win the scenario."""
    cell = _safety_cell([(1, 1, "delayed"), (1, 2, "infringement")], cap=2, scenarios=1)
    tempting = dataclasses.replace(
        cell,
        attempts=[
            dataclasses.replace(a, max_dev=1.0 if a.end_reason == "infringement" else 900.0)
            for a in cell.attempts
        ],
    )

    assert forced_los_curve(tempting) == [0.0, 0.0]


def test_forced_los_curve_keeps_a_solved_scenario_in_the_denominator() -> None:
    """A solved scenario stops writing rows, but still counts as one of the scenarios."""
    cell = _safety_cell(
        [(1, 1, "success"), (2, 1, "infringement"), (2, 2, "infringement")],
        cap=2,
        scenarios=2,
    )

    assert forced_los_curve(cell) == [0.5, 0.5]


def test_crossing_k_reports_the_first_strict_lead() -> None:
    """The rule and the shaded band both open where pass@k first exceeds LoS-free@k."""
    assert crossing_k([0.1, 0.5, 0.9], [0.9, 0.7, 0.4]) == 3
    assert crossing_k([0.9, 0.9], [0.5, 0.4]) == 1


def test_crossing_k_does_not_count_a_tie() -> None:
    """Equal shares are not a lead: the curves have to separate before the figure says so."""
    assert crossing_k([0.5, 0.6, 0.7], [0.5, 0.6, 0.7]) is None


def test_crossing_k_returns_none_when_retries_never_catch_up() -> None:
    """A policy that stays behind on solves gets no rule and no band."""
    assert crossing_k([0.1, 0.2, 0.3], [0.9, 0.8, 0.7]) is None


def test_greedy_note_prints_both_levels_in_metric_order() -> None:
    """The dashed lines share one texture, so the panel spells their values out."""
    cell = dataclasses.replace(
        _cell(10, [3, 5]),
        deterministic_rate=0.65,
        deterministic_los_free_rate=0.77,
    )
    assert greedy_note(cell) == "greedy 65% / 77%"


def test_greedy_note_marks_a_missing_level_rather_than_shifting_the_other() -> None:
    """A lone dashed line must not be read as the metric it is not."""
    cell = dataclasses.replace(
        _cell(10, [3, 5]),
        deterministic_rate=0.65,
        deterministic_los_free_rate=None,
    )
    assert greedy_note(cell) == "greedy 65% / -"


def test_greedy_note_is_empty_without_a_greedy_pass() -> None:
    cell = dataclasses.replace(
        _cell(10, [3, 5]),
        deterministic_rate=None,
        deterministic_los_free_rate=None,
    )
    assert greedy_note(cell) == ""


def test_exposure_curves_divide_by_the_recorded_scenario_count() -> None:
    """Unrun attempts are neither a solve nor an exposure, but the run still had 4 scenarios."""
    cell = _safety_cell([(1, 1, "success"), (2, 1, "infringement")], cap=2, scenarios=4)
    passk, los_free = exposure_curves(cell)
    assert passk == [0.25, 0.25]
    assert los_free == [0.75, 0.75]


def test_exposure_curves_are_monotone_in_opposite_directions() -> None:
    """The whole point of the figure: retries can only add solves and only add exposure."""
    rows = [
        (seed, k, "infringement" if (seed + k) % 3 else "success")
        for seed in range(5)
        for k in (1, 2, 3)
    ]
    passk, los_free = exposure_curves(_safety_cell(rows, cap=3, scenarios=5))
    assert passk == [0.2, 0.6, 1.0]
    assert los_free == pytest.approx([0.2, 0.0, 0.0])
    assert all(a <= b for a, b in zip(passk, passk[1:], strict=False))
    assert all(a >= b for a, b in zip(los_free, los_free[1:], strict=False))


def test_parse_cell_name_splits_aircraft_and_slug() -> None:
    assert parse_cell_name("a15_898b8_1_checkpoint_000137") == (15, "898b8_1_checkpoint_000137")


def test_parse_cell_name_rejects_foreign_directories() -> None:
    with pytest.raises(ValueError, match="benchmark cell"):
        parse_cell_name("benchmark_comparison.md")


def test_load_cell_reads_summary_and_raw_rows(tmp_path: Path) -> None:
    directory = _write_cell(tmp_path, "a12_ckpt_x", cap=3, nz=[4, 6, 8], det=0.25)
    cell = load_cell(directory)
    assert (cell.aircraft, cell.checkpoint, cell.cap, cell.scenarios) == (12, "ckpt_x", 3, 3)
    assert cell.solve_threshold_s == 60.0
    assert cell.curve == [0.5, 0.5, 0.5]
    assert cell.deterministic_rate == 0.25
    assert cell.nz_actions_mean == 6.0


def test_load_cell_keeps_missing_baselines_as_none(tmp_path: Path) -> None:
    """A run without a greedy pass draws neither dashed level."""
    directory = _write_cell(tmp_path, "a12_ckpt_x", cap=1, nz=[4], det=None)
    cell = load_cell(directory)
    assert cell.deterministic_rate is None
    assert cell.deterministic_los_free_rate is None


def test_load_cell_reads_binary_deterministic_los_free_rate(tmp_path: Path) -> None:
    """Greedy safety is the end reason, not the solve flag: a delayed pass is safe but unsolved."""
    directory = _write_cell(
        tmp_path,
        "a12_ckpt_x",
        cap=1,
        nz=[4, 6, 8],
        det=1 / 3,
        scenarios=3,
    )
    outcomes = [
        summarize_seed(
            [_attempt(seed)],
            cap=1,
            deterministic=_attempt(seed, attempt=0, end_reason=reason),
        )
        for seed, reason in ((1, "success"), (2, "infringement"), (3, "delayed"))
    ]
    write_attempts_per_seed(directory / "attempts_per_seed.csv", outcomes)

    assert load_cell(directory).deterministic_los_free_rate == 2 / 3


def test_load_cell_ignores_a_per_seed_file_without_a_greedy_pass(tmp_path: Path) -> None:
    """An all-or-nothing gate: one unrun greedy scenario is not a 100% safety line."""
    directory = _write_cell(tmp_path, "a12_ckpt_x", cap=1, nz=[4], det=None)
    write_attempts_per_seed(
        directory / "attempts_per_seed.csv",
        [summarize_seed([_attempt(1)], cap=1)],
    )

    assert load_cell(directory).deterministic_los_free_rate is None


def test_load_cell_survives_a_per_seed_file_missing_the_greedy_columns(tmp_path: Path) -> None:
    """A trimmed archive costs the dashed level, never the whole figure."""
    directory = _write_cell(tmp_path, "a12_ckpt_x", cap=1, nz=[4], det=None)
    (directory / "attempts_per_seed.csv").write_text(
        "seed,solved,attempts_used,first_success_attempt,censored,best_max_dev,best_end_reason\n"
        "1,True,1,1,False,10.0,success\n",
        encoding="utf-8",
    )

    assert load_cell(directory).deterministic_los_free_rate is None


def test_discover_cells_rejects_more_densities_than_the_palette(tmp_path: Path) -> None:
    """Six densities would ship a blank panel and a mislabelled protocol span."""
    for aircraft in (10, 12, 15, 20, 25, 30):
        _write_cell(tmp_path, f"a{aircraft}_ckpt_x", cap=1, nz=[4], det=None)

    with pytest.raises(ValueError, match="exceed"):
        discover_cells([tmp_path], checkpoint="ckpt_x")


def test_discover_cells_orders_by_density_across_roots(tmp_path: Path) -> None:
    """Separate per-density runs are as valid an input as one whole grid."""
    high, low = tmp_path / "high", tmp_path / "low"
    _write_cell(high, "a20_ckpt_x", cap=1, nz=[9], det=0.1)
    _write_cell(low, "a10_ckpt_x", cap=1, nz=[3], det=0.9)
    cells = discover_cells([high, low], checkpoint="ckpt_x")
    assert [cell.aircraft for cell in cells] == [10, 20]


def test_discover_cells_ignores_other_checkpoints_and_stray_files(tmp_path: Path) -> None:
    _write_cell(tmp_path, "a10_wanted", cap=1, nz=[3], det=0.5)
    _write_cell(tmp_path, "a10_other", cap=1, nz=[3], det=0.5)
    (tmp_path / "benchmark_comparison.md").write_text("x", encoding="utf-8")
    cells = discover_cells([tmp_path], checkpoint="wanted")
    assert [cell.checkpoint for cell in cells] == ["wanted"]


def test_discover_cells_rejects_two_cells_at_one_density(tmp_path: Path) -> None:
    """Silently dropping one would plot a mislabelled curve."""
    _write_cell(tmp_path, "a10_ckpt_a", cap=1, nz=[3], det=0.5)
    _write_cell(tmp_path, "a10_ckpt_b", cap=1, nz=[3], det=0.5)
    with pytest.raises(ValueError, match="two cells for 10 aircraft"):
        discover_cells([tmp_path], checkpoint="ckpt")


def test_discover_cells_rejects_a_slug_matching_several_checkpoints(tmp_path: Path) -> None:
    """One curve per density, all from one checkpoint, or the figure lies."""
    ten, twenty = tmp_path / "ten", tmp_path / "twenty"
    _write_cell(ten, "a10_898b8_1_checkpoint_000137", cap=1, nz=[3], det=0.5)
    _write_cell(twenty, "a20_898b8_3_checkpoint_000123", cap=1, nz=[9], det=0.1)
    with pytest.raises(ValueError, match="matches several checkpoints"):
        discover_cells([ten, twenty], checkpoint="898b8")


def test_discover_cells_rejects_disagreeing_solve_windows(tmp_path: Path) -> None:
    """A density grid must not caption cells graded with different windows."""
    _write_cell(tmp_path, "a10_ckpt_x", cap=1, nz=[3], det=0.5, solve_threshold_s=30.0)
    _write_cell(tmp_path, "a20_ckpt_x", cap=1, nz=[9], det=0.5, solve_threshold_s=60.0)

    with pytest.raises(ValueError, match="solve threshold"):
        discover_cells([tmp_path], checkpoint="ckpt_x")


def test_discover_cells_reports_an_empty_match(tmp_path: Path) -> None:
    _write_cell(tmp_path, "a10_ckpt_a", cap=1, nz=[3], det=0.5)
    with pytest.raises(ValueError, match="no benchmark cells"):
        discover_cells([tmp_path], checkpoint="absent")


def test_action_count_bins_span_every_cell_on_one_grid() -> None:
    """Per-cell binning would make the outlines incomparable."""
    edges = action_count_bins([_cell(10, [3, 4]), _cell(20, [11])], width=2)
    assert edges[0] == 2.0
    assert edges[-1] == 12.0
    assert all(b - a == 2.0 for a, b in zip(edges, edges[1:], strict=False))


def test_action_count_bins_rejects_a_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="positive"):
        action_count_bins([_cell(10, [3])], width=0)


def test_action_count_bins_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="no attempts"):
        action_count_bins([], width=2)


def test_density_caption_states_the_episode_composition() -> None:
    """A bare episode count hides whether it is scenarios, retries or both."""
    cells = [_cell(10, [3] * 40, cap=4, scenarios=10), _cell(20, [9] * 40, cap=4, scenarios=10)]
    caption = density_caption(cells, bin_width=4)
    assert "10 pinned scenarios" in caption
    assert "up to 4 stochastic attempts each" in caption
    assert "4-action bins" in caption


def test_density_caption_trusts_the_recorded_scenario_count(tmp_path: Path) -> None:
    """Producers that stop retrying a solved scenario write far fewer rows.

    Dividing rows by the cap would call this run 5 scenarios, not 100.
    """
    directory = _write_cell(tmp_path, "a10_ckpt_x", cap=20, nz=[3] * 100, det=0.9, scenarios=100)
    caption = density_caption([load_cell(directory)], bin_width=4)
    assert "100 pinned scenarios" in caption
    assert "up to 20 stochastic attempts each" in caption


def test_density_caption_spans_grids_that_disagree() -> None:
    """Combining separate runs can mix scenario sets; the caption must not pick one."""
    caption = density_caption(
        [_cell(10, [3], cap=8, scenarios=100), _cell(20, [9], cap=20, scenarios=60)],
        bin_width=2,
    )
    assert "60-100 pinned scenarios" in caption
    assert "up to 8-20 stochastic attempts each" in caption


def test_renderers_write_all_figures_and_expose_artists(
    tmp_path: Path, saved_figures: list
) -> None:
    exposure_base = _safety_cell(
        [
            (1, 1, "infringement"),
            (1, 2, "success"),
            (2, 1, "delayed"),
            (2, 2, "delayed"),
        ],
        cap=2,
        scenarios=2,
    )
    exposure_base = dataclasses.replace(
        exposure_base,
        attempts=[
            dataclasses.replace(record, nz_actions=actions)
            for record, actions in zip(exposure_base.attempts, [3, 5, 9, 13], strict=True)
        ],
    )
    cells = [
        dataclasses.replace(
            exposure_base,
            aircraft=10,
            curve=[0.4, 0.8],
            deterministic_los_free_rate=0.25,
        ),
        dataclasses.replace(
            exposure_base,
            aircraft=20,
            curve=[0.2, 0.6],
            deterministic_los_free_rate=0.25,
        ),
    ]
    passk = tmp_path / "figs" / "passk.png"
    forced_los = tmp_path / "figs" / "forced_los.png"
    exposure = tmp_path / "figs" / "exposure.png"
    actions = tmp_path / "figs" / "actions.png"

    render_passk_density(passk, cells, checkpoint="ckpt")
    render_forced_los_density(forced_los, cells, checkpoint="ckpt")
    render_exposure_density(exposure, cells, checkpoint="ckpt")
    render_action_count_density(actions, cells, checkpoint="ckpt", bin_width=2)

    # Keep the file checks as an exception smoke test: a renderer that raises
    # before writing is still a useful failure even when artist checks change.
    assert passk.stat().st_size > 0
    assert forced_los.stat().st_size > 0
    assert exposure.stat().st_size > 0
    assert actions.stat().st_size > 0
    assert len(saved_figures) == 4

    # Curve/baseline order and values catch dropped or swapped pass@k artists.
    passk_ax = saved_figures[0].axes[0]
    assert [line.get_linestyle() for line in passk_ax.lines] == ["-", "--", "-", "--"]
    assert [list(passk_ax.lines[index].get_ydata()) for index in (0, 2)] == [
        [0.4, 0.8],
        [0.2, 0.6],
    ]
    assert passk_ax.get_ylim() == (0.0, 1.0)
    assert "60 s success window" in passk_ax.get_title(loc="left")

    # Forced-risk curves and complement baselines catch skipped risk rendering.
    forced_ax = saved_figures[1].axes[0]
    assert [list(forced_ax.lines[index].get_ydata()) for index in (0, 2)] == [
        [0.5, 0.0],
        [0.5, 0.0],
    ]
    assert [list(forced_ax.lines[index].get_ydata()) for index in (1, 3)] == [
        [0.75, 0.75],
        [0.75, 0.75],
    ]
    assert forced_ax.get_ylim() == pytest.approx((0.0, 0.75 * 1.18))
    assert "60 s success window" in forced_ax.get_title(loc="left")

    # Exposure artist order catches passing pass@k where LoS-free@k belongs.
    exposure_fig = saved_figures[2]
    assert _figure_text(exposure_fig)[0] == "Retry exposure vs traffic density - ckpt"
    assert len(exposure_fig.axes) == 2
    exposure_ax = exposure_fig.axes[0]
    assert len(exposure_ax.collections) == 1
    assert len(exposure_ax.lines) == 4
    assert list(exposure_ax.lines[-2].get_ydata()) == [0.5, 0.5]
    assert list(exposure_ax.lines[-1].get_ydata()) == [0.0, 0.5]
    assert exposure_ax.get_ylim() == (0.0, 1.0)
    assert "dashed = greedy" in _figure_text(exposure_fig)[1]

    # Two histogram passes per density and shared limits catch bin/normalization drift.
    action_ax = saved_figures[3].axes[0]
    assert len(action_ax.patches) == 2 * len(cells)
    assert action_ax.get_xlim() == (2.0, 14.0)
    assert action_ax.get_ylim()[0] == 0.0
    assert action_ax.get_ylim()[1] > 0.0
    assert action_ax.get_xlabel() == "nonzero actions in the episode"


def test_exposure_renderer_handles_one_density_without_greedy_baselines(
    tmp_path: Path, saved_figures: list
) -> None:
    """A partial archive still yields a genuinely baseline-free exposure panel."""
    path = tmp_path / "exposure.png"
    cell = dataclasses.replace(
        _cell(10, [3, 5]),
        deterministic_rate=None,
        deterministic_los_free_rate=None,
    )

    render_exposure_density(path, [cell], checkpoint="ckpt")

    assert path.stat().st_size > 0
    ax = saved_figures[-1].axes[0]
    assert len(ax.lines) == 2
    assert ax.get_ylim() == (0.0, 1.0)
    assert all("dashed" not in text for text in _figure_text(saved_figures[-1]))
    assert all("greedy" not in text for text in _figure_text(saved_figures[-1]))


def test_exposure_renderer_draws_a_panel_that_never_leads(
    tmp_path: Path, saved_figures: list
) -> None:
    """A policy retries cannot rescue gets no crossing rule and no band."""
    path = tmp_path / "exposure.png"
    cell = dataclasses.replace(
        _safety_cell([(1, 1, "infringement"), (2, 1, "delayed")], cap=2, scenarios=2),
        deterministic_los_free_rate=0.5,
    )

    render_exposure_density(path, [cell], checkpoint="ckpt")

    assert path.stat().st_size > 0
    ax = saved_figures[-1].axes[0]
    assert len(ax.lines) == 4
    assert list(ax.lines[-2].get_ydata()) == [0.5, 0.5]
    assert list(ax.lines[-1].get_ydata()) == [0.0, 0.0]
    assert "pass@k never leads up to k=2" in _figure_text(saved_figures[-1])


def test_passk_renderer_tolerates_a_missing_baseline(tmp_path: Path, saved_figures: list) -> None:
    """A run without a greedy pass still has a curve worth drawing."""
    cell = dataclasses.replace(
        _cell(10, [3, 5]),
        deterministic_rate=None,
        deterministic_los_free_rate=None,
    )
    path = tmp_path / "passk.png"
    render_passk_density(path, [cell], checkpoint="ckpt")

    assert path.stat().st_size > 0
    ax = saved_figures[-1].axes[0]
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_ydata()) == [0.4, 0.8]
    assert ax.get_ylim() == (0.0, 1.0)
    assert all("dashed" not in text for text in _figure_text(saved_figures[-1]))


@pytest.mark.parametrize(
    "renderer",
    [render_passk_density, render_forced_los_density, render_exposure_density],
    ids=["passk", "forced_los", "exposure"],
)
def test_renderers_caption_greedy_clause_only_when_baseline_exists(
    tmp_path: Path, saved_figures: list, renderer
) -> None:
    """Captions never promise a dashed line that the input did not provide."""
    without_baseline = dataclasses.replace(
        _cell(10, [3, 5]),
        deterministic_rate=None,
        deterministic_los_free_rate=None,
    )
    with_baseline = dataclasses.replace(
        without_baseline,
        deterministic_rate=0.5,
        deterministic_los_free_rate=0.25,
    )

    path = tmp_path / "without.png"
    renderer(path, [without_baseline], checkpoint="ckpt")
    without_text = "\n".join(_figure_text(saved_figures[-1]))
    assert "dashed" not in without_text

    path = tmp_path / "with.png"
    renderer(path, [with_baseline], checkpoint="ckpt")
    with_text = "\n".join(_figure_text(saved_figures[-1]))
    assert "dashed" in with_text


def test_renderer_uses_stored_window_unless_explicitly_overridden(
    tmp_path: Path, saved_figures: list
) -> None:
    cell = dataclasses.replace(_cell(10, [3, 5]), solve_threshold_s=60.0)

    render_passk_density(tmp_path / "stored.png", [cell], checkpoint="ckpt")
    assert "60 s success window" in saved_figures[-1].axes[0].get_title(loc="left")

    render_passk_density(
        tmp_path / "override.png",
        [cell],
        checkpoint="ckpt",
        solve_threshold_s=30.0,
    )
    assert "30 s success window" in saved_figures[-1].axes[0].get_title(loc="left")


def test_density_cli_forwards_explicit_threshold_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI flag reaches every renderer; omission forwards the stored-value sentinel."""
    from scripts import density_plots as cli

    calls: list[float | None] = []
    cell = _cell(10, [3, 5])

    monkeypatch.setattr(cli, "force_runtime_log_color", lambda: None)
    monkeypatch.setattr(cli, "discover_cells", lambda roots, checkpoint: [cell])
    monkeypatch.setattr(
        cli,
        "render_passk_density",
        lambda *args, **kwargs: calls.append(kwargs["solve_threshold_s"]),
    )
    monkeypatch.setattr(
        cli,
        "render_forced_los_density",
        lambda *args, **kwargs: calls.append(kwargs["solve_threshold_s"]),
    )
    monkeypatch.setattr(cli, "render_action_count_density", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "render_exposure_density",
        lambda *args, **kwargs: calls.append(kwargs["solve_threshold_s"]),
    )
    monkeypatch.setattr(cli, "runtime_print", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "density-plots",
            str(tmp_path),
            "-c",
            "ckpt",
            "-o",
            str(tmp_path / "out"),
            "--solve-threshold",
            "30",
            "--exposure-figure",
        ],
    )
    cli.main()
    assert calls == [30.0, 30.0, 30.0]

    calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "density-plots",
            str(tmp_path),
            "-c",
            "ckpt",
            "-o",
            str(tmp_path / "out"),
            "--exposure-figure",
        ],
    )
    cli.main()
    assert calls == [None, None, None]


def test_renderer_rejects_disagreeing_stored_windows(tmp_path: Path, saved_figures: list) -> None:
    cells = [
        _cell(10, [3, 5], solve_threshold_s=30.0),
        _cell(20, [9, 13], solve_threshold_s=60.0),
    ]

    with pytest.raises(ValueError, match="solve threshold"):
        render_passk_density(tmp_path / "mismatch.png", cells, checkpoint="ckpt")
