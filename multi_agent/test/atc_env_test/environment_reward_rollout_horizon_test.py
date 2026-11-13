"""Tests for reward rollout horizon handling in ATCEnv."""

from types import SimpleNamespace
from typing import Any, cast

from support.env_patch import patch_env_global
from support.settings import live_env_settings

from actions.reward_parts import RewardParts
from actions.rewards import reward_options
from actions.timing_credit import TimingCreditState
from atc_env.environment import ATCEnv
from config.settings import settings_from_config
from simulator.commands import TimedCommandList


def _build_test_env() -> ATCEnv:
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env.max_steps_per_episode = 128
    env.episode_max_aircraft_count = 5
    env.step_counter = 1
    env.prev_world = object()
    env.initial_deviation = 0.0
    env.initial_infringements_total = 0.0
    env.episode_time_horizon = 100.0
    env._past_tracks = {}
    env.simulator = SimpleNamespace(world=SimpleNamespace(age_in_seconds=0.0, aircraft=[]))
    env.reward_settings = reward_options(settings=settings_from_config().reward)
    env.outcome_state = SimpleNamespace(
        targets=tuple(SimpleNamespace(callsign=str(i)) for i in range(5)), forecast_snapshot=None
    )
    env.timing_credit = TimingCreditState(
        [str(i) for i in range(5)], {}, scaled_min_baseline=1, delay_max=5
    )
    env._forecast = lambda world, steps: env.simulator.advance_multiple_steps(steps, world)
    env.last_reward = RewardParts(0.0, 0.0, 0.0)
    env.last_step_action_nums = {}
    env._step_prev_world_all_infr = None
    return env


def test_simulate_and_reward_rolls_out_remaining_steps_once(monkeypatch) -> None:
    env = _build_test_env()
    previous_planning_world = env.prev_world

    seen_steps = []
    worlds = {}

    def fake_advance_multiple_steps(steps, world):
        seen_steps.append(steps)
        key = f"w{len(seen_steps)}"
        worlds[key] = object()
        return worlds[key]

    sim = cast(Any, env.simulator)
    sim.advance_multiple_steps = fake_advance_multiple_steps
    sim.apply_action_and_advance = lambda *args, **kwargs: None
    seen_reward_args = {}

    def fake_calculate_reward(
        _interpreted_actions,
        previous_estimation,
        current_estimation,
        _initial_deviation,
        _initial_infringements_total,
        _episode_time_horizon,
        _current_time_s,
        episode_aircraft_count,
        **_kwargs,
    ):
        seen_reward_args["previous"] = previous_estimation
        seen_reward_args["current"] = current_estimation
        seen_reward_args["episode_aircraft_count"] = episode_aircraft_count
        return RewardParts(0.0, 0.0, 0.0)

    patch_env_global(monkeypatch, "get_world_infringements", lambda *_, **__: {})
    patch_env_global(
        monkeypatch,
        "calculate_reward",
        fake_calculate_reward,
    )

    env._simulate_and_reward(TimedCommandList(commands=[]))  # pylint: disable=protected-access
    assert seen_steps == [126]
    assert seen_reward_args["previous"] is previous_planning_world
    assert seen_reward_args["current"] is worlds["w1"]
    assert seen_reward_args["episode_aircraft_count"] == env.episode_max_aircraft_count
    assert env.prev_world is worlds["w1"]


def test_simulate_and_reward_uses_remaining_when_near_end(monkeypatch) -> None:
    env = _build_test_env()
    env.max_steps_per_episode = 5
    env.step_counter = 2

    seen_steps = []

    def fake_advance_multiple_steps(steps, world):
        seen_steps.append(steps)
        return object()

    sim = cast(Any, env.simulator)
    sim.advance_multiple_steps = fake_advance_multiple_steps
    sim.apply_action_and_advance = lambda *args, **kwargs: None
    patch_env_global(monkeypatch, "get_world_infringements", lambda *_, **__: {})
    patch_env_global(
        monkeypatch,
        "calculate_reward",
        lambda *args, **kwargs: RewardParts(0.0, 0.0, 0.0),
    )

    env._simulate_and_reward(TimedCommandList(commands=[]))  # pylint: disable=protected-access
    assert seen_steps == [2]


def test_simulate_and_reward_uses_policy_start_baselines_not_raw_initial_metrics(
    monkeypatch,
) -> None:
    env = _build_test_env()
    env.initial_deviation = 15.0
    env.initial_infringements_total = 3.0
    env.raw_initial_deviation = 150.0
    env.raw_initial_infringements_total = 30.0

    sim = cast(Any, env.simulator)
    sim.advance_multiple_steps = lambda steps, world: object()
    sim.apply_action_and_advance = lambda *args, **kwargs: None
    seen_reward_args: dict[str, float] = {}

    def fake_calculate_reward(
        _interpreted_actions,
        _previous_estimation,
        _current_estimation,
        initial_deviation,
        initial_infringements_total,
        _episode_time_horizon,
        _current_time_s,
        _episode_aircraft_count,
        **_kwargs,
    ):
        seen_reward_args["initial_deviation"] = initial_deviation
        seen_reward_args["initial_infringements_total"] = initial_infringements_total
        return RewardParts(0.0, 0.0, 0.0)

    patch_env_global(monkeypatch, "get_world_infringements", lambda *_, **__: {})
    patch_env_global(monkeypatch, "calculate_reward", fake_calculate_reward)

    env._simulate_and_reward(TimedCommandList(commands=[]))  # pylint: disable=protected-access

    assert seen_reward_args == {
        "initial_deviation": 15.0,
        "initial_infringements_total": 3.0,
    }
