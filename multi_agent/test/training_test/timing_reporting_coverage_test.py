"""Published exact-known means retain their target denominator and interval evidence."""

from types import SimpleNamespace

import pandas as pd
import pytest

from evaluation.report import collect as detailed
from evaluation.report.build import build_summary
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.ledger import OutcomeSnapshot


class Logger:
    def __init__(self):
        self.values = {}

    def log_value(self, key, value, **kwargs):
        self.values[key] = value


@pytest.mark.parametrize("complete", [False, True])
def test_callback_and_evaluation_publish_exact_target_coverage(complete, monkeypatch):
    import callbacks.episode_logging as metrics

    evidence = (
        (TimingEvidence(40, 40, "event"),) * 3
        if complete
        else (TimingEvidence(40, 40, "event"), TimingEvidence(-10, 10, "history"), None)
    )
    snapshot = OutcomeSnapshot(
        tuple(ArrivalTargetSpec(str(i), 100) for i in range(3)),
        tuple(
            ArrivalEvidence(str(i), True, value is not None, value)
            for i, value in enumerate(evidence)
        ),
        (),
        (),
        False,
    )
    world = SimpleNamespace(aircraft=[object()] * 3)
    monkeypatch.setattr(metrics, "infringement_severity_total", lambda *_: 0)
    monkeypatch.setattr(metrics, "per_aircraft_infringement_severity", lambda *_: 0)
    logger = Logger()
    metrics.log_final_deviation_and_infringement(logger, world, snapshot=snapshot)
    expected_count = 3 if complete else 1
    prefix = "custom_metrics/forecast_deviation_final"
    assert f"{prefix}_exact_target_count" in logger.values, (
        "published timing mean needs its exact target denominator"
    )
    assert logger.values[f"{prefix}_exact_known_mean_s_mean"] == 40
    assert logger.values[f"{prefix}_exact_target_count"] == expected_count
    assert logger.values[f"{prefix}_target_count"] == 3
    assert logger.values[f"{prefix}_exact_coverage_fraction_mean"] == expected_count / 3

    env = SimpleNamespace(
        last_end_reason="timeout",
        scenario_seed=1,
        step_counter=1,
        initial_agent_count=3,
        prev_world=world,
        forecast_coverage=snapshot,
        outcome_ledger=SimpleNamespace(snapshot=lambda: snapshot),
        last_outcome=None,
        last_verification=None,
        raw_initial_deviation=None,
        _action_history=[],
    )
    monkeypatch.setattr(detailed, "infringement_severity_total", lambda *_: 0)
    env.raw_initial_infringements_total = 0
    report_state = detailed.ReportState(max_episodes=1)
    callback = detailed.make_report_callback(report_state)()
    args = dict(
        episode=None,
        env_runner=SimpleNamespace(worker_index=0),
        metrics_logger=None,
        env=env,
        env_index=0,
        rl_module=None,
    )
    callback.on_episode_start(**args)
    callback.on_episode_end(**args)
    detailed_row = report_state.episode_rows[0]
    report = detailed_row["final_timing"]
    assert report["exact_count"] == expected_count
    assert report["target_count"] == 3
    assert report["exact_coverage_fraction"] == expected_count / 3
    if not complete:
        assert report["interval_count"] == report["unknown_count"] == 1
        assert report["arrivals"]["1"]["lower_error_s"] == -10
    assert detailed_row["final_dev_exact_known_mean_s"] == 40
    detailed_summary = build_summary(pd.DataFrame([detailed_row]), pd.DataFrame())
    assert (
        detailed_summary["final_timing_target_coverage"]["exact_coverage_fraction"]
        == expected_count / 3
    )
