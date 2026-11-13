"""Raw episode safety, timing, action and performance diagnostics."""

from typing import Any

from ray.rllib.callbacks.callbacks import RLlibCallback

from callbacks.callback_metric_helpers import (
    EPISODE_OUTCOME_REASONS,
)
from callbacks.episode_logging import (
    log_action_usage,
    log_diagnostic_metrics,
    log_episode_outcome,
    log_episode_start,
    log_episode_steps,
    log_final_deviation_and_infringement,
    log_scenario_generation_metrics,
    log_scenario_metrics,
    log_scheduler_observability,
)
from callbacks.outcome_metrics import log_completed_outcome
from callbacks.perf_metrics import log_iteration_timing_metrics
from callbacks.result_metrics import log_result_metrics
from common.env_utils import get_base_env
from common.sealed import consume


class ATCEpisodeMetricsCallback(RLlibCallback):
    """Logs per-episode initial/current total deviation and infringement severity sums"""

    def on_environment_created(self, *, env_runner, **kwargs):
        config = env_runner.config.env_config
        payload = config.get("resolved_metrics")
        if config.get("resolved_env") is not None:
            if payload is None:
                raise ValueError("bound metrics require immutable resolved_metrics payload")
            consume(payload, config["expected_semantic_digest"], role="metrics")
            self.consumed_metrics_payload = payload

    def _take_completed_outcome(self, metrics_logger: Any, base_env: Any) -> None:
        """Log the latest completed attempt once, then release it from the collector."""
        result = getattr(base_env, "last_completed_result", None)
        if result is None or getattr(self, "_last_outcome_attempt", None) == result.attempt_id:
            return
        self._last_outcome_attempt = result.attempt_id
        log_completed_outcome(metrics_logger, result)
        base_env.completed_collector.take(result.attempt_id)

    def on_episode_start(
        self,
        *,
        episode: Any,
        env_runner: Any,
        metrics_logger: Any,
        env: Any,
        env_index: int,
        rl_module: Any,
        **kwargs,
    ) -> None:
        base_env = get_base_env(env)

        self._take_completed_outcome(metrics_logger, base_env)
        log_episode_start(metrics_logger, base_env)

    def on_episode_end(
        self,
        *,
        episode: Any,
        env_runner: Any,
        metrics_logger: Any,
        env: Any,
        env_index: int,
        rl_module: Any,
        **kwargs,
    ) -> None:
        base_env = get_base_env(env)

        # RLlib may fire on_episode_end for sub-episodes where all tracked agents
        # terminated individually, even though the env hasn't reached a global end
        # condition (__all__). In that case last_end_reason is None and all metrics
        # would reflect mid-episode state — skip entirely.
        end_reason = getattr(base_env, "last_end_reason", None)
        if end_reason not in EPISODE_OUTCOME_REASONS:
            return

        world = getattr(base_env, "prev_world", None)

        if world is not None:
            log_final_deviation_and_infringement(
                metrics_logger, world, snapshot=getattr(base_env, "forecast_coverage", None)
            )

        self._take_completed_outcome(metrics_logger, base_env)
        log_episode_steps(metrics_logger, base_env)
        log_episode_outcome(metrics_logger, base_env)
        log_action_usage(metrics_logger, base_env)
        log_scenario_metrics(metrics_logger, base_env)
        log_scheduler_observability(metrics_logger, base_env)
        log_scenario_generation_metrics(metrics_logger, base_env)
        log_diagnostic_metrics(metrics_logger, base_env, world)

    def on_train_result(
        self,
        *,
        algorithm: Any,
        metrics_logger: Any | None = None,
        result: dict,
        **kwargs,
    ) -> None:
        log_iteration_timing_metrics(result)
        log_result_metrics(result)
