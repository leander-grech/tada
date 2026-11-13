"""Outcome kind metric names are persisted dashboards keys and must never change."""

from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from atc_env import episode_metrics
from callbacks.callback_metric_helpers import log_outcome_summary
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import classify_outcome

OUTCOME_KIND_METRICS = {
    "custom_metrics/outcome_physical_success_count",
    "custom_metrics/outcome_verified_noop_success_count",
    "custom_metrics/outcome_safe_delayed_count",
    "custom_metrics/outcome_infringement_count",
    "custom_metrics/outcome_unresolved_count",
}


def test_outcome_kind_metric_names_are_unchanged():
    live = OutcomeSnapshot(
        (ArrivalTargetSpec("A", 100),),
        (ArrivalEvidence("A", True, True, TimingEvidence(0, 0, "exact")),),
        (),
        (),
        False,
    )
    summary = episode_metrics.summarize_outcome(
        live, classify_outcome(live, None, threshold_s=60, policy_limit_reached=False), None
    )
    logger = MetricsLogger()
    log_outcome_summary(logger, summary)
    report = logger.reduce(compile=True)
    assert OUTCOME_KIND_METRICS <= set(report)
