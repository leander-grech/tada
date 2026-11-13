"""Tests for deterministic training scenario seed generation."""

import random

from support.settings import live_env_settings

from atc_env.environment import ATCEnv


def _make_aircraft_count_shell(
    *,
    max_aircraft_count: int,
    scenario_seed: int | None,
    evaluation_mode: bool = False,
) -> ATCEnv:
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env.max_aircraft_count = max_aircraft_count
    env.scenario_seed = scenario_seed
    env.evaluation_mode = evaluation_mode
    return env


def test_scenario_aircraft_count_is_fixed_across_seeds_and_roles():
    for evaluation in (True, False):
        for seed in (None, 7, 19, 29):
            env = _make_aircraft_count_shell(
                max_aircraft_count=7, scenario_seed=seed, evaluation_mode=evaluation
            )
            assert env._resolve_episode_max_aircraft_count() == 7


def test_evaluation_seed_stream_uses_manifest_order_then_round_robin() -> None:
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env._eval_seeds = [101, 202, 303]
    env._training_stream = None
    env._inference_checkpoint = None
    env._eval_episode_index = -1
    env._eval_randomize_count = 0
    env._eval_randomize_used = 0
    env._eval_random = random.Random(42)

    seeds = [env._next_evaluation_scenario_seed() for _ in range(5)]

    assert seeds == [101, 202, 303, 101, 202]


def _make_eval_cursor_shell(*, episode_index: int, scenario_finished: bool) -> ATCEnv:
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env._eval_seeds = [101, 202, 303]
    env._training_stream = None
    env._inference_checkpoint = None
    env._eval_episode_index = episode_index
    env._eval_randomize_count = 0
    env._eval_randomize_used = 0
    env._eval_random = random.Random(42)
    env._scenario_finished = scenario_finished
    env.scenario_seed = (
        env._eval_seeds[episode_index] if 0 <= episode_index < len(env._eval_seeds) else None
    )
    return env


def test_first_evaluation_selection_takes_the_first_manifest_seed() -> None:
    env = _make_eval_cursor_shell(episode_index=-1, scenario_finished=False)

    assert env._select_evaluation_seed() == 101
    assert env._eval_episode_index == 0


def test_finished_scenario_advances_the_evaluation_cursor() -> None:
    env = _make_eval_cursor_shell(episode_index=0, scenario_finished=True)

    assert env._select_evaluation_seed() == 202
    assert env._eval_episode_index == 1


def test_interrupted_scenario_replays_its_seed() -> None:
    """A reset arriving mid-scenario must not consume the next manifest entry."""
    env = _make_eval_cursor_shell(episode_index=1, scenario_finished=False)

    assert env._select_evaluation_seed() == 202
    assert env._eval_episode_index == 1
    # Repeated cuts on the same scenario keep replaying it.
    assert env._select_evaluation_seed() == 202
    assert env._eval_episode_index == 1


def test_finished_last_manifest_seed_wraps_to_the_first() -> None:
    env = _make_eval_cursor_shell(episode_index=2, scenario_finished=True)

    assert env._select_evaluation_seed() == 101
    assert env._eval_episode_index == 3


def test_training_requires_explicit_reservation_before_generation() -> None:
    import pytest

    with pytest.raises(ValueError, match="reserved"):
        ATCEnv({})


def test_pb2_seed_configuration_requires_migration() -> None:
    import pytest

    for option in ("training_trial_slot", "seed_generation"):
        with pytest.raises(ValueError, match="PB2"):
            ATCEnv({option: 0})
