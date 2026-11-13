"""Tests for the retry-until-solved (pass@k) analysis helpers."""

import csv

import pytest

from evaluation.attempts import (
    AttemptRecord,
    read_attempts_per_seed,
    read_attempts_raw,
    write_attempts_per_seed,
    write_attempts_raw,
)
from evaluation.stats import (
    attempt_buckets,
    build_report,
    pass_at_k,
    summarize_attempt_metrics,
    summarize_seed,
)


def _attempt(
    seed: int,
    attempt: int,
    *,
    solved: bool,
    max_dev: float = 50.0,
    init_dev_per_ac: float = 300.0,
    final_dev_per_ac: float = 20.0,
) -> AttemptRecord:
    return AttemptRecord(
        seed=seed,
        attempt=attempt,
        solved=solved,
        end_reason="success" if solved else "delayed",
        max_dev=max_dev,
        init_dev_per_ac=init_dev_per_ac,
        final_dev_per_ac=final_dev_per_ac,
        infr_severity=0.0,
        ep_return=1.0,
        steps=40,
        nz_actions=6,
        nz_ex_trb=4,
        spd=2,
        aircraft=20,
    )


def test_summarize_seed_reports_first_success_not_last() -> None:
    attempts = [
        _attempt(7, 3, solved=True),
        _attempt(7, 1, solved=False),
        _attempt(7, 2, solved=True),
    ]

    outcome = summarize_seed(attempts, cap=3)

    assert outcome.solved is True
    assert outcome.first_success_attempt == 2
    assert outcome.attempts_used == 3
    assert outcome.censored is False


def test_summarize_seed_marks_cap_exhaustion_as_censored() -> None:
    attempts = [_attempt(7, k, solved=False) for k in (1, 2)]

    outcome = summarize_seed(attempts, cap=2)

    assert outcome.solved is False
    assert outcome.first_success_attempt is None
    assert outcome.censored is True


def test_summarize_seed_best_tracks_lowest_deviation() -> None:
    attempts = [
        _attempt(7, 1, solved=False, max_dev=400.0),
        _attempt(7, 2, solved=False, max_dev=90.0),
        _attempt(7, 3, solved=False, max_dev=250.0),
    ]

    assert summarize_seed(attempts, cap=3).best_max_dev == 90.0


def test_summarize_seed_carries_deterministic_baseline() -> None:
    baseline = _attempt(7, 0, solved=False, max_dev=310.0)

    outcome = summarize_seed([_attempt(7, 1, solved=True)], cap=4, deterministic=baseline)

    assert outcome.deterministic_solved is False
    assert outcome.deterministic_max_dev == 310.0
    assert outcome.deterministic_end_reason == "delayed"


def test_summarize_seed_rejects_empty_attempts() -> None:
    with pytest.raises(ValueError, match="at least one attempt"):
        summarize_seed([], cap=3)


def test_pass_at_k_is_monotonic_and_cumulative() -> None:
    outcomes = [
        summarize_seed([_attempt(1, 1, solved=True)], cap=4),
        summarize_seed(
            [
                _attempt(2, 1, solved=False),
                _attempt(2, 2, solved=False),
                _attempt(2, 3, solved=True),
            ],
            cap=4,
        ),
        summarize_seed([_attempt(3, k, solved=False) for k in (1, 2, 3, 4)], cap=4),
    ]

    curve = pass_at_k(outcomes, cap=4)

    assert curve == [1 / 3, 1 / 3, 2 / 3, 2 / 3]
    assert curve == sorted(curve)


def test_pass_at_k_handles_no_scenarios() -> None:
    assert pass_at_k([], cap=3) == [0.0, 0.0, 0.0]


def test_attempt_buckets_partition_every_scenario() -> None:
    outcomes = [
        summarize_seed([_attempt(1, 1, solved=True)], cap=8),
        summarize_seed([_attempt(2, 1, solved=False), _attempt(2, 2, solved=True)], cap=8),
        summarize_seed([_attempt(3, k, solved=False) for k in range(1, 9)], cap=8),
    ]

    buckets = attempt_buckets(outcomes, cap=8)

    assert sum(count for _, count, _ in buckets) == len(outcomes)
    assert buckets[0] == ("1", 1, False)
    assert buckets[1] == ("2-3", 1, False)
    assert buckets[-1][1] == 1
    assert buckets[-1][2] is True


def test_attempt_buckets_clamp_labels_to_cap() -> None:
    outcomes = [summarize_seed([_attempt(1, 1, solved=True)], cap=2)]

    labels = [label for label, _, _ in attempt_buckets(outcomes, cap=2)]

    assert labels == ["1", "2", "not solved\nin 2"]


def test_attempt_buckets_label_full_ranges_when_reachable() -> None:
    outcomes = [summarize_seed([_attempt(1, k, solved=False) for k in range(1, 9)], cap=8)]

    labels = [label for label, _, _ in attempt_buckets(outcomes, cap=8)]

    assert labels == ["1", "2-3", "4-7", "8", "not solved\nin 8"]


def test_attempt_buckets_label_singleton_overflow_range() -> None:
    attempts = [_attempt(1, k, solved=k == 16) for k in range(1, 17)]
    outcome = summarize_seed(attempts, cap=16)

    labels = [label for label, _, _ in attempt_buckets([outcome], cap=16)]

    assert labels == ["1", "2-3", "4-7", "8-15", "16", "not solved\nin 16"]


def test_build_report_counts_episodes_and_censoring() -> None:
    attempts = [
        _attempt(1, 1, solved=True),
        _attempt(2, 1, solved=False),
        _attempt(2, 2, solved=False),
    ]
    outcomes = [
        summarize_seed([attempts[0]], cap=2, deterministic=_attempt(1, 0, solved=True)),
        summarize_seed(attempts[1:], cap=2, deterministic=_attempt(2, 0, solved=False)),
    ]

    report = build_report(outcomes, attempts, cap=2, solve_threshold_s=60.0)

    assert report["scenarios"] == 2
    assert report["episodes_run"] == 3
    assert report["solved_within_cap"] == 1
    assert report["censored"] == 1
    assert report["pass_at_1"] == 0.5
    assert report["pass_at_cap"] == 0.5
    assert report["deterministic_solved_rate"] == 0.5
    assert report["mean_attempts_to_success"] == 1.0
    assert report["solve_threshold_s"] == 60.0


def test_build_report_uses_none_for_mean_when_no_scenario_is_solved() -> None:
    attempts = [_attempt(1, k, solved=False) for k in (1, 2)]
    outcome = summarize_seed(attempts, cap=2)

    report = build_report([outcome], attempts, cap=2, solve_threshold_s=60.0)

    assert report["mean_attempts_to_success"] is None


def test_write_attempts_raw_sorts_by_seed_then_attempt(tmp_path) -> None:
    path = tmp_path / "attempts_raw.csv"

    write_attempts_raw(
        path,
        [_attempt(9, 2, solved=True), _attempt(4, 1, solved=False), _attempt(9, 1, solved=False)],
    )

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [(r["seed"], r["attempt"]) for r in rows] == [("4", "1"), ("9", "1"), ("9", "2")]
    assert rows[0]["init_dev_per_ac"] == "300.0"


def test_read_attempts_raw_round_trips_every_field(tmp_path) -> None:
    """Stored CSVs are the only record of a finished run."""
    path = tmp_path / "attempts_raw.csv"
    written = [_attempt(4, 1, solved=False), _attempt(9, 2, solved=True)]
    write_attempts_raw(path, written)

    assert read_attempts_raw(path) == written


def test_read_attempts_per_seed_round_trips_optional_baseline_fields(tmp_path) -> None:
    """Deterministic end reasons are the source for binary LoS-free baselines."""
    path = tmp_path / "attempts_per_seed.csv"
    outcomes = [
        summarize_seed(
            [_attempt(4, 1, solved=True)],
            cap=2,
            deterministic=_attempt(4, 0, solved=False),
        ),
        summarize_seed([_attempt(9, 1, solved=False)], cap=2),
    ]
    write_attempts_per_seed(path, outcomes)

    assert read_attempts_per_seed(path) == outcomes


def test_summarize_attempt_metrics_reports_shares_and_means() -> None:
    attempts = [
        _attempt(1, 1, solved=True),
        _attempt(1, 2, solved=False, max_dev=150.0),
        _attempt(2, 1, solved=False, max_dev=100.0),
        _attempt(2, 2, solved=False, max_dev=100.0),
    ]

    metrics = summarize_attempt_metrics(attempts)

    assert metrics["episodes"] == 4
    assert metrics["end_reason_share"] == {"delayed": 0.75, "success": 0.25}
    assert list(metrics["end_reason_share"]) == ["delayed", "success"]
    assert metrics["max_dev_mean"] == 100.0
    assert metrics["nz_ex_trb_mean"] == 4.0


def test_summarize_attempt_metrics_counts_reasons_before_dividing() -> None:
    attempts = [_attempt(seed=1, attempt=k, solved=True) for k in range(800)]

    metrics = summarize_attempt_metrics(attempts)

    assert metrics["end_reason_share"] == {"success": 1.0}


def test_summarize_attempt_metrics_survives_an_empty_run() -> None:
    assert summarize_attempt_metrics([]) == {"episodes": 0, "end_reason_share": {}}


def test_write_attempts_per_seed_emits_one_row_per_seed(tmp_path) -> None:
    path = tmp_path / "attempts_per_seed.csv"
    outcomes = [
        summarize_seed([_attempt(9, 1, solved=True)], cap=1),
        summarize_seed([_attempt(4, 1, solved=False)], cap=1),
    ]

    write_attempts_per_seed(path, outcomes)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["seed"] for r in rows] == ["4", "9"]
    assert [r["solved"] for r in rows] == ["False", "True"]
    assert rows[0]["censored"] == "True"


def test_plot_separates_exact_episode_means_unverified_and_missing_coverage(tmp_path, monkeypatch):
    import dataclasses

    import matplotlib.pyplot as plt

    from evaluation.stats import render_attempts_plot

    partial = dataclasses.replace(
        _attempt(1, 1, solved=False),
        max_dev=None,
        final_dev_per_ac=10.0,
        final_timing_exact_count=1,
        final_timing_target_count=10,
        final_timing_interval_count=2,
        final_timing_unknown_count=7,
        final_timing_source="owned_forecast",
    )
    unknown = dataclasses.replace(
        partial,
        seed=2,
        final_dev_per_ac=None,
        final_timing_exact_count=0,
        final_timing_interval_count=0,
        final_timing_unknown_count=10,
        final_timing_source="owned_forecast",
    )
    exact = dataclasses.replace(
        partial,
        seed=3,
        final_dev_per_ac=30.0,
        final_timing_exact_count=9,
        final_timing_target_count=9,
        final_timing_interval_count=0,
        final_timing_unknown_count=0,
        final_timing_source="owned_forecast",
    )
    unverified = _attempt(4, 1, solved=False, final_dev_per_ac=90.0)
    attempts = [partial, unknown, exact, unverified]
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    path = tmp_path / "mixed-coverage.png"
    render_attempts_plot(
        path, [], attempts, cap=2, title="Synthetic coverage", solve_threshold_s=60
    )
    fig = figures[-1]
    labels = [line.get_label() for line in fig.axes[2].lines]
    text = "\n".join(labels + [item.get_text() for item in fig.axes[2].texts])
    assert "episode-weighted exact-known mean" in text
    assert "20s" in text  # (10 + 30) / 2, not target-pooled 28s or unverified-pooled mean.
    assert "legacy/unverified" in text and "90s" in text
    assert "exact 10 / targets 29" in text and "interval 2" in text and "unknown 17" in text
    assert "missing coverage 1/4 episodes" in text
    assert "2 episodes" in text
    summary = summarize_attempt_metrics(attempts)
    assert summary["exact_known_deviation_mean"] == 20.0
    assert summary["legacy_unverified_deviation_mean"] == 90.0
    assert summary["final_timing_target_coverage"]["episodes_missing_coverage"] == 1
    assert path.stat().st_size > 0


def test_all_unknown_plot_does_not_draw_zero_final_mean(tmp_path, monkeypatch):
    import dataclasses

    import matplotlib.pyplot as plt

    from evaluation.stats import render_attempts_plot

    attempt = dataclasses.replace(
        _attempt(1, 1, solved=False),
        max_dev=None,
        init_dev_per_ac=None,
        final_dev_per_ac=None,
        final_timing_exact_count=0,
        final_timing_target_count=10,
        final_timing_interval_count=0,
        final_timing_unknown_count=10,
        final_timing_source="owned_forecast",
    )
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    render_attempts_plot(
        tmp_path / "unknown.png", [], [attempt], cap=2, title="All unknown", solve_threshold_s=60
    )
    axis = figures[-1].axes[2]
    assert len(axis.lines) == 1  # Only the operational threshold, no invented mean.
    assert "unknown 10" in "\n".join(t.get_text() for t in axis.texts)
    assert summarize_attempt_metrics([attempt])["exact_known_deviation_mean"] is None


@pytest.mark.parametrize("indices", [(1, 2), (0, 0)])
def test_canonical_roundtrip_ranks_safe_interval_and_reports_completed_exact(tmp_path, indices):
    import json
    from dataclasses import replace

    from atc_env.completed_scenario import CompletedScenarioResult
    from evaluation.attempts import completed_attempt
    from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
    from outcomes.ledger import OutcomeSnapshot
    from outcomes.verification import VerificationResult, classify_outcome

    def record(attempt_id, low, high, unsafe=False):
        live = OutcomeSnapshot(
            (ArrivalTargetSpec("A", 0),),
            (ArrivalEvidence("A", True, True, TimingEvidence(low, high, "fixture")),),
            (),
            (),
            False,
        )
        verification = (
            VerificationResult(replace(live, failed=True), 1, False, "verification_hazard")
            if unsafe
            else None
        )
        result = CompletedScenarioResult(
            attempt_id,
            classify_outcome(live, None, threshold_s=60, policy_limit_reached=True),
            live,
            verification,
            (),
            0,
            {"complete": True, "seed": 1717171717},
        )
        return completed_attempt(
            CompletedScenarioResult.from_dict(json.loads(json.dumps(result.to_dict())))
        )

    unsafe = record("unsafe", 1, 1, True)
    safe = record("safe", 70, 80)
    if indices != (0, 0):
        unsafe = replace(unsafe, attempt=indices[0])
        safe = replace(safe, attempt=indices[1])
    path = tmp_path / "canonical.csv"
    write_attempts_raw(path, [unsafe, safe])
    restored = read_attempts_raw(path)
    assert restored == [unsafe, safe]
    summary = summarize_seed(restored, cap=2)
    assert summary.best_end_reason == "delayed"
    assert summary.best_max_dev is None
    metrics = summarize_attempt_metrics(restored)
    assert metrics["exact_known_deviation_mean"] == 1
    assert metrics["legacy_unverified_deviation_episode_count"] == 0
    assert metrics["forecast_infringement_severity_mean"] is None
    assert metrics["infringement_severity_mean"] is None
    assert metrics["forecast_infringement_severity_known_count"] == 0
    assert metrics["end_reason_share"].get("success", 0) == 0
    with pytest.raises(ValueError, match="identity"):
        summarize_seed([replace(safe, attempt_id="swapped")], cap=1)
    with pytest.raises(json.JSONDecodeError):
        summarize_attempt_metrics([replace(safe, completed_result_json="{")])
    with pytest.raises(ValueError, match="duplicate canonical attempt"):
        summarize_seed([safe, safe], cap=2)
    with pytest.raises(ValueError, match="duplicate canonical attempt"):
        summarize_attempt_metrics([safe, safe])


def test_current_attempt_missing_evidence_is_not_canonical_success():
    from dataclasses import replace

    missing = replace(_attempt(7, 1, solved=True), attempt_id="current")
    with pytest.raises(ValueError, match="missing canonical evidence"):
        summarize_seed([missing], cap=1)
    with pytest.raises(ValueError, match="missing canonical evidence"):
        summarize_attempt_metrics([missing])
