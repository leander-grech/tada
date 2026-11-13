"""Per-episode safety, timing, action and scenario metrics logged at episode end."""

from typing import Any

from actions.actions import SPEED_ACTIONS, TROMBONE_ACTIONS
from atc_env.episode_metrics import (
    episode_timing_report,
    infringement_severity_total,
    peak_forecast_severity,
    per_aircraft_infringement_severity,
)
from callbacks.callback_metric_helpers import (
    EPISODE_OUTCOME_REASONS,
    log_fraction_metric,
    log_mean_min_max_metric,
    log_sum_metric,
)
from config.config import Config
from config.settings import settings_from_config
from outcomes import vocabulary


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0.0 else 0.0


def log_episode_outcome(
    metrics_logger: Any,
    base_env: Any,
) -> None:
    end_reason = getattr(base_env, "last_end_reason", None)
    window = int(Config.CUSTOM_METRICS_WINDOW)
    # Per-episode indicators: rolling mean approximates recent outcome fractions.
    for r in EPISODE_OUTCOME_REASONS:
        log_fraction_metric(
            metrics_logger,
            f"custom_metrics/episodes_end_{r}",
            end_reason == r,
            window=window,
        )
        log_sum_metric(
            metrics_logger,
            f"custom_metrics/episodes_end_{r}_count",
            1.0 if end_reason == r else 0.0,
        )
    log_sum_metric(metrics_logger, "custom_metrics/episodes_end_total_count", 1.0)


def log_scenario_metrics(metrics_logger: Any, base_env: Any) -> None:
    """Log fixed scenario aircraft counts."""
    max_aircraft = getattr(base_env, "max_aircraft_count", None)
    if max_aircraft is not None:
        log_mean_min_max_metric(
            metrics_logger, "custom_metrics/scenario_max_aircraft", max_aircraft
        )
    episode_max_aircraft = getattr(base_env, "episode_max_aircraft_count", None)
    if episode_max_aircraft is not None:
        log_mean_min_max_metric(
            metrics_logger,
            "custom_metrics/scenario_episode_aircraft",
            episode_max_aircraft,
        )


def log_scenario_generation_metrics(metrics_logger: Any, base_env: Any) -> None:
    """Log number of attempts needed to generate the scenario for this episode."""
    simulator = getattr(base_env, "simulator", None)
    attempts = float(getattr(simulator, "last_scenario_generation_attempts", 0.0))
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/scenario_generation_attempts",
        attempts,
    )


def log_diagnostic_metrics(metrics_logger: Any, base_env: Any, world: Any) -> None:
    """Log diagnostic metrics for lockout and conflict analysis."""
    locked_count = len(getattr(base_env, "_locked_out_agents", set()))
    initial_count = max(1, int(getattr(base_env, "initial_agent_count", 1)))
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/locked_fraction",
        float(locked_count) / float(initial_count),
    )

    end_reason = getattr(base_env, "last_end_reason", None)
    if end_reason == vocabulary.INFRINGEMENT:
        step_counter = max(0, int(getattr(base_env, "step_counter", 0)))
        max_steps = max(1, int(getattr(base_env, "max_steps_per_episode", 1)))
        log_mean_min_max_metric(
            metrics_logger,
            "custom_metrics/infringement_step_fraction",
            float(step_counter) / float(max_steps),
        )

    if world is not None:
        log_mean_min_max_metric(
            metrics_logger,
            "custom_metrics/peak_forecast_severity",
            peak_forecast_severity(world, settings_from_config().safety),
        )


def log_episode_steps(metrics_logger: Any, base_env: Any) -> None:
    """Log realized environment steps for the finished episode."""
    episode_steps = float(max(0, int(getattr(base_env, "step_counter", 0))))
    log_mean_min_max_metric(metrics_logger, "custom_metrics/episode_steps", episode_steps)


def log_scheduler_observability(metrics_logger: Any, base_env: Any) -> None:
    """Log scheduler diagnostics around observed/eligible mismatch and sector exclusion."""
    observed_count = max(0, int(getattr(base_env, "last_observation_count", 0)))
    blocked_count = max(0, int(getattr(base_env, "last_blocked_observation_count", 0)))
    controlled_count = max(0, int(getattr(base_env, "last_controlled_candidate_count", 0)))
    outside_excluded_count = max(0, int(getattr(base_env, "last_outside_excluded_count", 0)))

    blocked_fraction = _safe_ratio(float(blocked_count), float(observed_count))
    outside_excluded_fraction = _safe_ratio(float(outside_excluded_count), float(controlled_count))

    log_mean_min_max_metric(
        metrics_logger, "custom_metrics/blocked_observation_fraction", blocked_fraction
    )
    log_mean_min_max_metric(
        metrics_logger, "custom_metrics/outside_excluded_count", float(outside_excluded_count)
    )
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/outside_excluded_fraction",
        outside_excluded_fraction,
    )


def _log_optional_timing(metrics_logger, key, value):
    log_fraction_metric(metrics_logger, f"{key}_known_fraction", value is not None)
    log_sum_metric(metrics_logger, f"{key}_known_count", int(value is not None))
    if value is not None:
        log_mean_min_max_metric(metrics_logger, key, value)


def _log_timing_report(metrics_logger, key, report):
    _log_optional_timing(metrics_logger, key, report["exact_mean_absolute_error_s"])
    if report["exact_mean_absolute_error_s"] is not None:
        log_mean_min_max_metric(
            metrics_logger, f"{key}_exact_known_mean_s", report["exact_mean_absolute_error_s"]
        )
    for field, suffix in (
        ("exact_count", "exact_target_count"),
        ("target_count", "target_count"),
        ("interval_count", "interval_target_count"),
        ("unknown_count", "unknown_target_count"),
    ):
        log_sum_metric(metrics_logger, f"{key}_{suffix}", report[field])
    for field in ("exact_coverage_fraction", "coverage_fraction"):
        log_mean_min_max_metric(metrics_logger, f"{key}_{field}", report[field])
    log_sum_metric(metrics_logger, f"{key}_source_{report['source']}_count", 1)


def log_final_deviation_and_infringement(metrics_logger: Any, world: Any, *, snapshot=None) -> None:
    report = episode_timing_report(world, snapshot=snapshot)
    final_dev = report["exact_mean_absolute_error_s"]
    final_infr = infringement_severity_total(world, settings_from_config().safety)
    final_infr_per_ac = per_aircraft_infringement_severity(world, settings_from_config().safety)

    _log_timing_report(metrics_logger, "custom_metrics/forecast_deviation_final", report)
    if final_dev is not None:
        log_sum_metric(metrics_logger, "custom_metrics/forecast_deviation_final_total", final_dev)
    log_mean_min_max_metric(
        metrics_logger, "custom_metrics/forecast_infringement_total_final", final_infr
    )
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/forecast_infringement_total_final_per_ac",
        final_infr_per_ac,
    )
    log_sum_metric(
        metrics_logger,
        "custom_metrics/forecast_infringement_total_final_per_ac_total",
        final_infr_per_ac,
    )


def _aggregate_action_stats(
    history: Any,
) -> tuple[list[int], list[float], list[int], int, int]:
    """Aggregate action counts and reward-bearing action counts separately."""

    counts = [0] * Config.NUM_ACTIONS
    rewards_by_action = [0.0] * Config.NUM_ACTIONS
    reward_counts = [0] * Config.NUM_ACTIONS
    for entry in history or []:
        act = entry.get("action", -1)
        if 0 <= act < Config.NUM_ACTIONS:
            counts[act] += 1
            if entry.get("phase") != "pre_episode":
                rewards_by_action[act] += entry.get("reward", 0.0)
                reward_counts[act] += 1

    nonzero_count = sum(counts[1:])
    trombone_count = sum(counts[a] for a in TROMBONE_ACTIONS)
    return (
        counts,
        rewards_by_action,
        reward_counts,
        nonzero_count,
        nonzero_count - trombone_count,
    )


def log_action_usage(metrics_logger: Any, base_env: Any) -> None:
    history = getattr(base_env, "_action_history", []) or []
    counts, rewards_by_action, reward_counts, nonzero_count, nonzero_ex_trombone = (
        _aggregate_action_stats(history)
    )

    for idx, cnt in enumerate(counts):
        log_mean_min_max_metric(metrics_logger, f"custom_metrics/{idx}_action", cnt)
        avg_reward = _safe_ratio(rewards_by_action[idx], float(reward_counts[idx]))
        log_mean_min_max_metric(metrics_logger, f"custom_metrics/{idx}_action_reward", avg_reward)

    ac_count = max(1, int(getattr(base_env, "initial_agent_count", 1)))
    actions_per_ac = nonzero_ex_trombone / ac_count
    log_mean_min_max_metric(metrics_logger, "custom_metrics/nonzero_action", nonzero_count)
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/diagnostic_nonzero_ex_trombone_action",
        nonzero_ex_trombone,
    )
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/diagnostic_nonzero_ex_trombone_action_per_ac",
        actions_per_ac,
    )
    log_sum_metric(
        metrics_logger,
        "custom_metrics/diagnostic_nonzero_ex_trombone_action_per_ac_total",
        actions_per_ac,
    )
    speed_count = sum(counts[a] for a in SPEED_ACTIONS)
    log_mean_min_max_metric(metrics_logger, "custom_metrics/speed_action", speed_count)
    nonzero_reward_total = sum(rewards_by_action[1:])
    nonzero_reward_avg = _safe_ratio(nonzero_reward_total, float(sum(reward_counts[1:])))
    log_mean_min_max_metric(
        metrics_logger, "custom_metrics/nonzero_reward_action", nonzero_reward_avg
    )


def log_episode_start(metrics_logger: Any, base_env: Any) -> None:
    # Raw baselines reflect scenario difficulty before pre-episode auto-solve.
    raw_initial_total_dev = base_env.raw_initial_deviation
    raw_initial_infr_total = float(base_env.raw_initial_infringements_total)
    policy_start_total_dev = base_env.initial_deviation
    policy_start_infr_total = float(base_env.initial_infringements_total)

    aircraft_count = len(base_env.agents) if base_env.agents else 1
    raw_initial_per_aircraft_dev = (
        raw_initial_total_dev / aircraft_count if raw_initial_total_dev is not None else None
    )
    policy_start_per_aircraft_dev = (
        policy_start_total_dev / aircraft_count if policy_start_total_dev is not None else None
    )

    _log_optional_timing(
        metrics_logger, "custom_metrics/deviation_initial", raw_initial_per_aircraft_dev
    )
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/forecast_infringement_total_initial",
        raw_initial_infr_total,
    )
    _log_optional_timing(
        metrics_logger, "custom_metrics/deviation_policy_start", policy_start_per_aircraft_dev
    )
    log_mean_min_max_metric(
        metrics_logger,
        "custom_metrics/forecast_infringement_total_policy_start",
        policy_start_infr_total,
    )
