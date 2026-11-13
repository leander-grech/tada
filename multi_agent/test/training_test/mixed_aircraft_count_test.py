"""Training scenarios draw their aircraft count from the seed; evaluation keeps the full count."""

from training.env import TrainingATCEnv


def _shell(*, seed: int, evaluation_mode: bool, minimum: int, maximum: int) -> TrainingATCEnv:
    env = TrainingATCEnv.__new__(TrainingATCEnv)
    env.max_aircraft_count = maximum
    env._min_aircraft_count = minimum
    env.scenario_seed = seed
    env.evaluation_mode = evaluation_mode
    return env


def test_training_count_is_a_deterministic_function_of_the_seed_within_range() -> None:
    counts = [
        _shell(
            seed=seed, evaluation_mode=False, minimum=10, maximum=20
        )._resolve_episode_max_aircraft_count()
        for seed in range(200)
    ]
    repeated = [
        _shell(
            seed=seed, evaluation_mode=False, minimum=10, maximum=20
        )._resolve_episode_max_aircraft_count()
        for seed in range(200)
    ]
    assert counts == repeated
    assert min(counts) == 10 and max(counts) == 20


def test_evaluation_always_uses_the_full_count() -> None:
    for seed in range(50):
        env = _shell(seed=seed, evaluation_mode=True, minimum=10, maximum=20)
        assert env._resolve_episode_max_aircraft_count() == 20
