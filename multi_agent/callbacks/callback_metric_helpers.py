"""Shared helpers for extracting and writing callback metrics in RLlib result dicts."""

import math
from typing import Any, Iterable

from config.config import Config
from outcomes import vocabulary

EPISODE_OUTCOME_REASONS = vocabulary.ALL_END_REASONS


def to_finite_float(value: Any) -> float | None:
    """Convert value to finite float, returning None for invalid input."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def log_mean_min_max_metric(metrics_logger: Any, key: str, value: float) -> None:
    """Log mean/min/max (no percentiles) with a consistent rolling window."""
    v = float(value)
    window = int(Config.CUSTOM_METRICS_WINDOW)
    metrics_logger.log_value(f"{key}_mean", v, reduce="mean", window=window)
    metrics_logger.log_value(f"{key}_min", v, reduce="min", window=window)
    metrics_logger.log_value(f"{key}_max", v, reduce="max", window=window)


def log_fraction_metric(
    metrics_logger: Any, key: str, value: float, window: int | None = None
) -> None:
    """Log a per-episode indicator as a rolling fraction metric."""
    v = 1.0 if value else 0.0
    metrics_logger.log_value(key, v, reduce="mean", window=window)


def log_sum_metric(metrics_logger: Any, key: str, value: float) -> None:
    """Log a per-iteration sum for denominator-aware metrics."""
    metrics_logger.log_value(key, float(value), reduce="sum")


def _metric_key_variants(metric_name: str) -> Iterable[str]:
    """Yield metric keys in preferred lookup order."""
    for suffix in ("_mean", "_sum", ""):
        yield f"{metric_name}{suffix}"


def _first_finite_metric_value(metrics: dict[str, Any], keys: Iterable[str]) -> float | None:
    """Return the first finite metric value found under any of the candidate keys."""
    for key in keys:
        value = to_finite_float(metrics.get(key))
        if value is not None:
            return value
    return None


def get_env_runners_metric_value(result: dict, metric_name: str) -> float | None:
    """Read env_runners metric from nested or flattened result layouts."""
    env_runners = result.get("env_runners", {})
    if isinstance(env_runners, dict):
        value = _first_finite_metric_value(env_runners, (metric_name,))
        if value is not None:
            return value

    return _first_finite_metric_value(result, (f"env_runners/{metric_name}",))


def get_timer_metric_value(result: dict, metric_name: str) -> float | None:
    """Read top-level timer metric from nested or flattened result layouts."""
    timers = result.get("timers", {})
    if isinstance(timers, dict):
        value = _first_finite_metric_value(timers, (metric_name,))
        if value is not None:
            return value

    return _first_finite_metric_value(result, (f"timers/{metric_name}",))


def get_custom_metric_value(result: dict, metric_name: str) -> float | None:
    """Read custom metric from nested or flattened result layouts."""
    env_runners = result.get("env_runners", {})
    if isinstance(env_runners, dict):
        custom_metrics = env_runners.get("custom_metrics", {})
        if isinstance(custom_metrics, dict):
            value = _first_finite_metric_value(custom_metrics, _metric_key_variants(metric_name))
            if value is not None:
                return value
        value = _first_finite_metric_value(
            env_runners,
            (f"custom_metrics/{key}" for key in _metric_key_variants(metric_name)),
        )
        if value is not None:
            return value

    return _first_finite_metric_value(
        result,
        (f"env_runners/custom_metrics/{key}" for key in _metric_key_variants(metric_name)),
    )


def get_learner_metric_value(
    result: dict,
    policy_name: str,
    metric_name: str,
    fallback_policy_names: tuple[str, ...] = (),
) -> float | None:
    """Read learner metric from nested or flattened result layouts."""
    policy_names = (policy_name, *fallback_policy_names)
    learners = result.get("learners", {})
    if isinstance(learners, dict):
        for candidate in policy_names:
            policy_metrics = learners.get(candidate, {})
            if isinstance(policy_metrics, dict):
                value = to_finite_float(policy_metrics.get(metric_name))
                if value is not None:
                    return value

            value = to_finite_float(learners.get(f"{candidate}/{metric_name}"))
            if value is not None:
                return value

    for candidate in policy_names:
        value = to_finite_float(result.get(f"learners/{candidate}/{metric_name}"))
        if value is not None:
            return value

    return None


def set_custom_metric_value(result: dict, metric_name: str, value: float) -> None:
    """Write custom metric to the supported env_runner result shapes."""
    parsed = to_finite_float(value)
    if parsed is None:
        return

    env_runners = result.get("env_runners")
    if not isinstance(env_runners, dict):
        env_runners = {}
        result["env_runners"] = env_runners

    env_custom_metrics = env_runners.get("custom_metrics")
    if not isinstance(env_custom_metrics, dict):
        env_custom_metrics = {}
        env_runners["custom_metrics"] = env_custom_metrics

    env_custom_metrics[metric_name] = parsed
    result[f"env_runners/custom_metrics/{metric_name}"] = parsed


def log_outcome_summary(metrics_logger: Any, summary: dict) -> None:
    """Stage per-report additive outcome evidence for true-scenario capture consumers.

    Means and coverage fractions are derived from pooled target denominators at
    the driver. Bounds use their own min/max reducers, never interval midpoints.
    """
    for key, value in summary.items():
        path = f"custom_metrics/outcome_{key}"
        if key.endswith("_mean_absolute_error_s") or key.endswith("_coverage_fraction"):
            continue
        if key.endswith("_max_absolute_error_s"):
            if value is not None:
                metrics_logger.log_value(path, float(value), reduce="max")
        elif isinstance(value, (bool, int, float)):
            log_sum_metric(metrics_logger, path, float(value))
        elif key.endswith("_interval_bounds_s"):
            for lower, upper in value.values():
                metrics_logger.log_value(path + "_lower", float(lower), reduce="min")
                metrics_logger.log_value(path + "_upper", float(upper), reduce="max")
    reasons = {
        "incomplete_attempt",
        "attempt_error",
        "live_loss_of_separation",
        "live_airspace_exit",
        "verification_target_mismatch",
        "verification_hazard",
        "world_identity_mismatch",
        "empty_targets",
        "missing_target",
        "verification_cap",
        "no_progress",
        "native_age_exceeded_cap",
        "incomplete_arrivals",
        "timing_unresolved",
        "timeout",
        "native_mission_failed",
    }
    for reason in reasons | set(summary["violation_reasons"]):
        log_sum_metric(
            metrics_logger,
            f"custom_metrics/outcome_reason_{reason}_count",
            int(reason in summary["violation_reasons"]),
        )
    statuses = {
        "not_run",
        "completed",
        "world_identity_mismatch",
        "empty_targets",
        "verification_hazard",
        "missing_target",
        "verification_cap",
        "no_progress",
        "native_age_exceeded_cap",
    }
    for status in statuses | {summary["verification_status"]}:
        log_sum_metric(
            metrics_logger,
            f"custom_metrics/outcome_verification_status_{status}_count",
            int(status == summary["verification_status"]),
        )
