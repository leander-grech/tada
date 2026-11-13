"""Behavioral raw episode metrics and iteration timing tests."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

import callbacks.episode_logging as episode_logging
from actions.actions import Action
from callbacks.metrics_callback import ATCEpisodeMetricsCallback
from config.config import Config


class _OutcomeLoggerStub:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.reductions: dict[str, str | None] = {}

    def log_value(self, key, value, reduce=None, window=None):  # pylint: disable=unused-argument
        self.values[key] = float(value)
        self.reductions[key] = reduce


def test_on_train_result_logs_iteration_timing_breakdown_metrics() -> None:
    callback = ATCEpisodeMetricsCallback()
    result = {}
    result["timers"] = {
        "training_iteration": 10.0,
        "env_runner_sampling_timer": 4.0,
        "learner_update_timer": 5.0,
        "synch_weights": 0.5,
    }
    result["env_runners"] = {"num_env_steps_sampled": 2000.0}
    result["learners"] = {"__all_modules__": {"num_env_steps_trained": 4000.0}}

    callback.on_train_result(algorithm=None, result=result)

    assert result["env_runners/custom_metrics/perf_time_iteration_s"] == 10.0
    assert result["env_runners/custom_metrics/perf_time_sampling_s"] == 4.0
    assert result["env_runners/custom_metrics/perf_time_learner_update_s"] == 5.0
    assert result["env_runners/custom_metrics/perf_time_sync_weights_s"] == 0.5
    assert result["env_runners/custom_metrics/perf_time_overhead_s"] == pytest.approx(0.5)
    assert result["env_runners/custom_metrics/perf_frac_sampling"] == pytest.approx(0.4)
    assert result["env_runners/custom_metrics/perf_frac_learner_update"] == pytest.approx(0.5)
    assert result["env_runners/custom_metrics/perf_frac_sync_weights"] == pytest.approx(0.05)
    assert result["env_runners/custom_metrics/perf_frac_overhead"] == pytest.approx(0.05)
    assert result["env_runners/custom_metrics/perf_sampling_env_steps_per_s"] == pytest.approx(
        500.0
    )
    assert result["env_runners/custom_metrics/perf_update_env_steps_per_s"] == pytest.approx(800.0)
    assert (
        result["env_runners/custom_metrics/perf_frac_sampling"]
        + result["env_runners/custom_metrics/perf_frac_learner_update"]
        + result["env_runners/custom_metrics/perf_frac_sync_weights"]
        + result["env_runners/custom_metrics/perf_frac_overhead"]
    ) == pytest.approx(1.0)


def test_on_train_result_falls_back_to_time_this_iter_for_iteration_time() -> None:
    callback = ATCEpisodeMetricsCallback()
    result: dict[str, float] = {
        "time_this_iter_s": 20.0,
        "timers/env_runner_sampling_timer": 8.0,
        "timers/learner_update_timer": 6.0,
        "env_runners/num_env_steps_sampled": 800.0,
    }

    callback.on_train_result(algorithm=None, result=result)

    assert result["env_runners/custom_metrics/perf_time_iteration_s"] == 20.0
    assert result["env_runners/custom_metrics/perf_time_sampling_s"] == 8.0
    assert result["env_runners/custom_metrics/perf_time_learner_update_s"] == 6.0
    assert result["env_runners/custom_metrics/perf_time_sync_weights_s"] == 0.0
    assert result["env_runners/custom_metrics/perf_time_overhead_s"] == pytest.approx(6.0)
    assert result["env_runners/custom_metrics/perf_frac_overhead"] == pytest.approx(0.3)
    assert result["env_runners/custom_metrics/perf_sampling_env_steps_per_s"] == pytest.approx(
        100.0
    )


def test_on_train_result_skips_iteration_breakdown_without_iteration_time() -> None:
    callback = ATCEpisodeMetricsCallback()
    result: dict[str, float] = {
        "timers/env_runner_sampling_timer": 3.0,
        "timers/learner_update_timer": 2.0,
    }

    callback.on_train_result(algorithm=None, result=result)

    assert result["env_runners/custom_metrics/perf_time_sampling_s"] == 3.0
    assert result["env_runners/custom_metrics/perf_time_learner_update_s"] == 2.0
    assert result["env_runners/custom_metrics/perf_time_sync_weights_s"] == 0.0
    assert "env_runners/custom_metrics/perf_time_iteration_s" not in result
    assert "env_runners/custom_metrics/perf_time_overhead_s" not in result
    assert "env_runners/custom_metrics/perf_frac_sampling" not in result
    assert "env_runners/custom_metrics/perf_frac_learner_update" not in result
    assert "env_runners/custom_metrics/perf_frac_sync_weights" not in result
    assert "env_runners/custom_metrics/perf_frac_overhead" not in result


def test_on_train_result_skips_throughput_when_timer_denominator_is_zero() -> None:
    callback = ATCEpisodeMetricsCallback()
    result = {}
    result["timers"] = {
        "training_iteration": 10.0,
        "env_runner_sampling_timer": 0.0,
        "learner_update_timer": 0.0,
    }
    result["env_runners"] = {"num_env_steps_sampled": 2000.0}
    result["learners"] = {"__all_modules__": {"num_env_steps_trained": 4000.0}}

    callback.on_train_result(algorithm=None, result=result)

    assert "env_runners/custom_metrics/perf_sampling_env_steps_per_s" not in result
    assert "env_runners/custom_metrics/perf_update_env_steps_per_s" not in result


def test_log_episode_outcome_counts_every_episode_by_end_reason() -> None:
    logger = _OutcomeLoggerStub()
    base_env = SimpleNamespace(
        last_end_reason="success", episode_max_aircraft_count=4, max_aircraft_count=5
    )
    episode_logging.log_episode_outcome(logger, base_env)
    assert logger.values["custom_metrics/episodes_end_success"] == 1.0
    assert logger.values["custom_metrics/episodes_end_success_count"] == 1.0
    assert logger.values["custom_metrics/episodes_end_infringement_count"] == 0.0
    assert logger.values["custom_metrics/episodes_end_total_count"] == 1.0
    assert logger.reductions["custom_metrics/episodes_end_total_count"] == "sum"


def test_on_episode_end_skips_outcome_metrics_for_rllib_sub_episode_without_env_end_reason() -> (
    None
):
    callback = ATCEpisodeMetricsCallback()
    logger = _OutcomeLoggerStub()

    episode = MultiAgentEpisode(
        observation_space=gym.spaces.Dict(
            {"A": gym.spaces.Discrete(2), "B": gym.spaces.Discrete(2)}
        ),
        action_space=gym.spaces.Dict({"A": gym.spaces.Discrete(2), "B": gym.spaces.Discrete(2)}),
        agent_to_module_mapping_fn=lambda _agent_id, _episode: Config.POLICY_NAME,
    )
    episode.add_env_reset(observations={"A": 0}, infos={})
    episode.add_env_step(
        observations={"A": 1},
        actions={"A": 0},
        rewards={"A": 1.0, "B": 0.0},
        infos={},
        terminateds={"A": True, "__all__": False},
        truncateds={"__all__": False},
    )
    assert episode.is_done

    base_env = SimpleNamespace(
        prev_world=None,
        step_counter=4,
        last_end_reason=None,
        _action_history=[],
        max_aircraft_count=5,
        episode_max_aircraft_count=5,
        last_observation_count=0,
        last_blocked_observation_count=0,
        last_controlled_candidate_count=0,
        last_outside_excluded_count=0,
        initial_agent_count=2,
        _locked_out_agents=set(),
        max_steps_per_episode=20,
        simulator=SimpleNamespace(last_scenario_generation_attempts=1),
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=logger,
        env=SimpleNamespace(unwrapped=base_env),
        env_index=0,
        rl_module=None,
    )

    # Guard at the top of on_episode_end skips ALL metric logging for sub-episodes
    assert not logger.values


def test_on_episode_start_uses_raw_initial_metrics_and_logs_policy_start_baseline() -> None:
    callback = ATCEpisodeMetricsCallback()
    logger = _OutcomeLoggerStub()
    base_env = SimpleNamespace(
        agents=["A", "B", "C"],
        raw_initial_deviation=120.0,
        raw_initial_infringements_total=4.5,
        initial_deviation=60.0,
        initial_infringements_total=1.5,
    )
    env = SimpleNamespace(unwrapped=base_env)

    callback.on_episode_start(
        episode=None,
        env_runner=None,
        metrics_logger=logger,
        env=env,
        env_index=0,
        rl_module=None,
    )

    assert logger.values["custom_metrics/deviation_initial_mean"] == 40.0
    assert logger.values["custom_metrics/forecast_infringement_total_initial_mean"] == 4.5
    assert logger.values["custom_metrics/deviation_policy_start_mean"] == 20.0
    assert logger.values["custom_metrics/forecast_infringement_total_policy_start_mean"] == 1.5


def test_log_action_usage_counts_pre_episode_auto_solve_without_zeroing_reward_average() -> None:
    logger = _OutcomeLoggerStub()
    action_num = int(Action.LENGTHEN_TROMBONE)
    base_env = SimpleNamespace(
        _action_history=[
            {"action": action_num, "reward": 0.0, "phase": "pre_episode"},
            {"action": action_num, "reward": 5.0},
        ],
        initial_agent_count=2,
        max_aircraft_count=5,
        episode_max_aircraft_count=3,
    )

    episode_logging.log_action_usage(logger, base_env)

    assert logger.values[f"custom_metrics/{action_num}_action_mean"] == 2.0
    assert logger.values[f"custom_metrics/{action_num}_action_reward_mean"] == 5.0


def test_episode_end_reports_forecast_deviation_once_and_feeds_review(monkeypatch) -> None:
    callback = ATCEpisodeMetricsCallback()

    monkeypatch.setattr(
        episode_logging,
        "episode_timing_report",
        lambda _world, **_: {
            "exact_mean_absolute_error_s": 12.0,
            "exact_count": 1,
            "target_count": 1,
            "interval_count": 0,
            "unknown_count": 0,
            "exact_coverage_fraction": 1.0,
            "coverage_fraction": 1.0,
            "source": "test_forecast",
        },
    )
    monkeypatch.setattr(
        episode_logging,
        "infringement_severity_total",
        lambda *_: 0.25,
    )
    monkeypatch.setattr(
        episode_logging,
        "per_aircraft_infringement_severity",
        lambda *_: 0.05,
    )
    monkeypatch.setattr(episode_logging, "peak_forecast_severity", lambda *_: 0.0)

    logger = _OutcomeLoggerStub()
    base_env = SimpleNamespace(
        prev_world=object(),
        step_counter=4,
        last_end_reason="success",
        _action_history=[],
        max_aircraft_count=5,
        episode_max_aircraft_count=5,
        last_observation_count=0,
        last_blocked_observation_count=0,
        last_controlled_candidate_count=0,
        last_outside_excluded_count=0,
        simulator=SimpleNamespace(last_scenario_generation_attempts=1),
    )
    env = SimpleNamespace(unwrapped=base_env)
    episode = SimpleNamespace(get_return=lambda: 7.5)

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=logger,
        env=env,
        env_index=0,
        rl_module=None,
    )

    assert logger.values["custom_metrics/forecast_deviation_final_mean"] == 12.0
    assert not [k for k in logger.values if "stage_max" in k]
    deviation_reports = {k for k in logger.values if k.endswith("_exact_target_count")}
    assert deviation_reports == {"custom_metrics/forecast_deviation_final_exact_target_count"}
