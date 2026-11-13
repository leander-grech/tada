"""Derived training-result metrics: outcome ratios and effective learner values."""

from callbacks.callback_metric_helpers import (
    get_custom_metric_value,
    get_learner_metric_value,
    set_custom_metric_value,
)


def log_result_metrics(result: dict) -> None:
    """Derive outcome coverage/error ratios and copy the effective learner schedule."""
    for scope in ("live", "verified"):
        prefix = f"outcome_{scope}_"
        exact = get_custom_metric_value(result, prefix + "exact_count")
        absolute_sum = get_custom_metric_value(result, prefix + "exact_absolute_error_sum_s")
        targets = get_custom_metric_value(result, prefix + "target_count")
        intervals = get_custom_metric_value(result, prefix + "interval_count")
        if exact is not None and exact > 0 and absolute_sum is not None:
            set_custom_metric_value(
                result, prefix + "exact_mean_absolute_error_s", absolute_sum / exact
            )
        if targets is not None and targets > 0 and exact is not None and intervals is not None:
            set_custom_metric_value(
                result, prefix + "coverage_fraction", (exact + intervals) / targets
            )
            set_custom_metric_value(result, prefix + "exact_coverage_fraction", exact / targets)
    for source, target in (
        ("default_optimizer_learning_rate", "effective_learning_rate"),
        ("curr_entropy_coeff", "effective_entropy_coeff"),
    ):
        value = get_learner_metric_value(result, "shared_policy", source)
        if value is not None:
            set_custom_metric_value(result, target, value)
    lambda_val = get_learner_metric_value(result, "shared_policy", "curr_lambda_value")
    if lambda_val is not None:
        set_custom_metric_value(result, "learner_lambda", lambda_val)
