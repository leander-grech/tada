"""Unknown coverage is explicit in control, features and numerical reports."""

from importlib import import_module
from typing import Any, cast

from config.settings import settings_from_config
from outcomes.evidence import ArrivalEvidence, TimingEvidence
from outcomes.ledger import OutcomeSnapshot


def test_coverage_maps_keep_unknown_absent_and_report_intervals_honestly():
    consumers = import_module("atc_env.coverage_consumers")
    snapshot = OutcomeSnapshot(
        (),
        (
            ArrivalEvidence("exact", True, True, TimingEvidence(-40, -40, "event")),
            ArrivalEvidence("interval", True, True, TimingEvidence(-10, 10, "history")),
            ArrivalEvidence("unknown", True, False, None),
        ),
        (),
        (),
        False,
    )
    assert consumers.proxy_map(snapshot) == {"exact": -40, "interval": 10}
    assert consumers.exact_map(snapshot) == {"exact": -40}
    assert consumers.format_timing(snapshot.arrivals[1].timing) == "[-10, +10] s"
    assert consumers.format_timing(None) == "unknown"
    assert consumers.success_margin({}, {"missing"}, 30) == 0
    assert consumers.success_margin({"exact": 0}, {"exact", "missing"}, 30) == 0


def test_zero_timing_credit_never_falls_back_to_legacy_delta(monkeypatch):
    import actions.rewards as rewards
    from simulator.commands import TimedCommandList

    monkeypatch.setattr(rewards, "get_world_infringements", lambda *_: {})
    result = rewards.calculate_reward(
        TimedCommandList(commands=[]),
        cast(Any, object()),
        cast(Any, object()),
        100,
        0,
        1000,
        0,
        2,
        timing_credit=0.0,
        options=rewards.reward_options(settings_from_config().reward),
        gate_center=settings_from_config().reward.reward_infringement_gate_center,
        safety=settings_from_config().safety,
    )
    assert result.deviation == 0


def test_numerical_report_distinguishes_exact_interval_unknown():
    consumers = import_module("atc_env.coverage_consumers")
    snapshot = OutcomeSnapshot(
        (),
        (
            ArrivalEvidence("exact", True, True, TimingEvidence(-40, -40, "event")),
            ArrivalEvidence("interval", True, True, TimingEvidence(-10, 10, "history")),
            ArrivalEvidence("unknown", False, False, None),
        ),
        (),
        (),
        False,
    )
    report = consumers.timing_report(snapshot)
    assert report["exact_count"] == report["interval_count"] == report["unknown_count"] == 1
    assert report["sign_ambiguous_count"] == 1
    assert report["exact_mean_absolute_error_s"] == 40
    assert report["coverage_fraction"] == 2 / 3
    assert report["arrivals"]["interval"] == {
        "kind": "interval",
        "lower_error_s": -10,
        "upper_error_s": 10,
    }


def test_optional_arrival_metrics_survive_attempt_roundtrip_and_known_only_reducers(tmp_path):
    import json

    from evaluation.attempts import AttemptRecord, read_attempts_raw, write_attempts_raw
    from evaluation.run import mean_metric
    from evaluation.stats import summarize_attempt_metrics

    record = AttemptRecord(1, 1, False, "timeout", None, None, None, 0.0, 0.0, 1, 0, 0, 0, 1)
    path = tmp_path / "attempts.csv"
    write_attempts_raw(path, [record])
    assert read_attempts_raw(path) == [record]
    summary = summarize_attempt_metrics([record])
    assert summary["max_dev_mean"] is None
    assert summary["max_dev_known_count"] == 0
    assert mean_metric([{"final_dev": None}, {"final_dev": 30.0}], "final_dev") == 30
    json.dumps(summary, allow_nan=False)
