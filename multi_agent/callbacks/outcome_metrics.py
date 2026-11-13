"""Completed-attempt outcome metrics from the immutable terminal result."""

from typing import Any

from atc_env.episode_metrics import (
    summarize_actions,
    summarize_outcome,
)
from callbacks.callback_metric_helpers import (
    log_outcome_summary,
    log_sum_metric,
)


def log_completed_outcome(metrics_logger: Any, result: Any) -> None:
    """Log one completed attempt's outcome ledger, actions and reward components."""
    summary = summarize_outcome(result.ledger, result.operational_outcome, result.verification)
    log_outcome_summary(metrics_logger, summary)
    for key, value in summarize_actions(result.actions).items():
        log_sum_metric(metrics_logger, f"custom_metrics/actions_{key}", value)
    for key, value in result.diagnostics["reward_components"].items():
        log_sum_metric(metrics_logger, f"custom_metrics/reward_component_{key}", value)
    log_sum_metric(
        metrics_logger, "custom_metrics/learned_policy_steps", result.diagnostics["steps"]
    )
