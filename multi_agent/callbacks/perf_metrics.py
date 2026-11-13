"""Training-iteration timing breakdown: sampling, learner update, weight sync."""

from callbacks.callback_metric_helpers import (
    get_env_runners_metric_value,
    get_learner_metric_value,
    get_timer_metric_value,
    set_custom_metric_value,
    to_finite_float,
)


def log_iteration_timing_metrics(result: dict) -> None:
    """Log training-iteration timing breakdown for sampling vs learner update."""
    iteration_time = get_timer_metric_value(result, "training_iteration")
    if iteration_time is None:
        iteration_time = to_finite_float(result.get("time_this_iter_s"))

    sampling_time = get_timer_metric_value(result, "env_runner_sampling_timer") or 0.0
    learner_update_time = get_timer_metric_value(result, "learner_update_timer") or 0.0
    sync_weights_time = get_timer_metric_value(result, "synch_weights") or 0.0

    set_custom_metric_value(result, "perf_time_sampling_s", sampling_time)
    set_custom_metric_value(result, "perf_time_learner_update_s", learner_update_time)
    set_custom_metric_value(result, "perf_time_sync_weights_s", sync_weights_time)

    if iteration_time is not None and iteration_time > 0.0:
        known_time = sampling_time + learner_update_time + sync_weights_time
        iteration_time = float(iteration_time)
        overhead_time = max(0.0, iteration_time - known_time)

        set_custom_metric_value(result, "perf_time_iteration_s", iteration_time)
        set_custom_metric_value(result, "perf_time_overhead_s", overhead_time)
        set_custom_metric_value(result, "perf_frac_sampling", sampling_time / iteration_time)
        set_custom_metric_value(
            result, "perf_frac_learner_update", learner_update_time / iteration_time
        )
        set_custom_metric_value(
            result, "perf_frac_sync_weights", sync_weights_time / iteration_time
        )
        set_custom_metric_value(result, "perf_frac_overhead", overhead_time / iteration_time)

    sampled_env_steps = get_env_runners_metric_value(result, "num_env_steps_sampled")
    if sampled_env_steps is not None and sampling_time > 0.0:
        set_custom_metric_value(
            result,
            "perf_sampling_env_steps_per_s",
            float(sampled_env_steps) / sampling_time,
        )

    trained_env_steps = get_learner_metric_value(
        result,
        "__all_modules__",
        "num_env_steps_trained",
        fallback_policy_names=("shared_policy",),
    )
    if trained_env_steps is not None and learner_update_time > 0.0:
        set_custom_metric_value(
            result,
            "perf_update_env_steps_per_s",
            float(trained_env_steps) / learner_update_time,
        )
