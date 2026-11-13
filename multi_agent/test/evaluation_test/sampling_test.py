"""Tests for how an attempts round walks one env runner's scenario slice.

A scenario is several RLlib episodes: aircraft enter and leave, so every
aircraft airborne can be terminated on the same step while the scenario runs
on.  RLlib closes the ``MultiAgentEpisode`` there, which makes an episode
budget the wrong unit for "cover this slice once".  These tests pin the
timestep-driven walk that replaced it.
"""

from types import SimpleNamespace

import pytest

from evaluation.attempts import AttemptRecord, completed_attempt
from evaluation.run import sample_round
from evaluation.sampling import sample_completed_shard


def _sample_shard(runner, *, rng_seed):
    return sample_completed_shard(runner, rng_seed=rng_seed, transport=completed_attempt)


HORIZON = 40


def _record(seed: int) -> AttemptRecord:
    return AttemptRecord(
        seed=seed,
        attempt=0,
        solved=False,
        end_reason="delayed",
        max_dev=0.0,
        init_dev_per_ac=0.0,
        final_dev_per_ac=0.0,
        infr_severity=0.0,
        ep_return=0.0,
        steps=1,
        nz_actions=0,
        nz_ex_trb=0,
        spd=0,
        aircraft=1,
    )


class _Runner:
    """An owned attempt can fragment without losing its identity or replaying RNG."""

    def __init__(self, seeds, *, steps_per_scenario, drop=None, fragments_per_scenario=1):
        from atc_env.completed_scenario import CompletedScenarioCollector

        self.worker_index = 1
        self.config = SimpleNamespace(explore=True)
        self.calls = []
        self.env = SimpleNamespace(
            _eval_seeds=seeds,
            evaluation_scenario_count=len(seeds),
            max_steps_per_episode=HORIZON,
            policy_time_limit_steps=HORIZON,
            completed_collector=CompletedScenarioCollector(),
        )
        self._steps_per_scenario = steps_per_scenario
        self._drop = drop or set()
        self._credit = 0
        self._needs_initial_reset = True

    def sample(self, **kwargs):
        from atc_env.completed_scenario import CompletedScenarioResult
        from outcomes.ledger import OutcomeSnapshot
        from outcomes.verification import classify_outcome

        self.calls.append(kwargs)
        if self._needs_initial_reset:
            self._needs_initial_reset = False
            self._credit = 0
            self.env.attempt_id = self.env._next_attempt_id
        self._credit += kwargs["num_timesteps"]
        if (
            self._credit >= self._steps_per_scenario
            and self.env._next_attempt_seed not in self._drop
        ):
            ledger = OutcomeSnapshot((), (), (), (), False)
            result = CompletedScenarioResult(
                self.env.attempt_id,
                classify_outcome(ledger, None, threshold_s=60, policy_limit_reached=True),
                ledger,
                None,
                (),
                float(self._credit),
                {"complete": True, "seed": self.env._next_attempt_seed},
            )
            self.env.completed_collector.capture(result.attempt_id, result)
        return [SimpleNamespace(is_done=False)]


class _RoundGroup:
    def __init__(self, runner: _Runner, *, healthy_remote_workers: int = 1):
        self.runner = runner
        self.healthy_remote_workers = healthy_remote_workers
        self.local_env_runner: bool | None = None

    def num_healthy_remote_workers(self) -> int:
        return self.healthy_remote_workers

    def foreach_env_runner(self, func, *, local_env_runner: bool):
        self.local_env_runner = local_env_runner
        return [func(self.runner)]


def test_shard_starts_new_attempt_without_forced_sampling_reset() -> None:
    runner = _Runner([1, 2, 3], steps_per_scenario=10)

    _sample_shard(runner, rng_seed=7)

    assert runner.calls[0]["force_reset"] is False


def test_shard_bounded_incomplete_does_not_reset_stochastic_fragments() -> None:
    runner = _Runner([1, 2, 3, 4, 5], steps_per_scenario=50)

    records = _sample_shard(runner, rng_seed=7)

    assert sorted(record.seed for record in records) == [1, 2, 3, 4, 5]
    assert all(call["force_reset"] is False for call in runner.calls)


def test_shard_walk_breaks_a_repeating_cut_cycle() -> None:
    runner = _Runner(list(range(12)), steps_per_scenario=30)

    records = _sample_shard(runner, rng_seed=7)

    assert sorted(record.seed for record in records) == list(range(12))


def test_shard_walk_resumes_after_a_clean_chunk_boundary() -> None:
    runner = _Runner([1, 2, 3, 4, 5], steps_per_scenario=40)

    _sample_shard(runner, rng_seed=7)

    assert runner.calls[1]["force_reset"] is False


def test_shard_walk_samples_timesteps_never_episodes() -> None:
    runner = _Runner([1, 2, 3], steps_per_scenario=10)

    _sample_shard(runner, rng_seed=7)

    assert runner.calls, "the shard was never sampled"
    assert all("num_episodes" not in call for call in runner.calls)
    assert all(call["num_timesteps"] > 0 for call in runner.calls)


def test_shard_walk_returns_every_scenario_in_the_slice() -> None:
    runner = _Runner([11, 22, 33], steps_per_scenario=10)

    records = _sample_shard(runner, rng_seed=7)

    assert sorted(record.seed for record in records) == [11, 22, 33]


def test_shard_walk_handles_multiple_fragments_per_scenario() -> None:
    runner = _Runner(
        list(range(1, 9)),
        steps_per_scenario=HORIZON,
        fragments_per_scenario=3,
    )

    records = _sample_shard(runner, rng_seed=7)

    seeds = [record.seed for record in records]
    assert seeds == list(range(1, 9))
    assert len(seeds) == len(set(seeds))


def test_sample_round_stamps_attempt_index_on_every_record() -> None:
    runner = _Runner([1, 2, 3], steps_per_scenario=10)
    group = _RoundGroup(runner)
    algo = SimpleNamespace(eval_env_runner_group=group)

    records = sample_round(algo, scenarios=[1, 2, 3], attempt=7, rng_seed=11)

    assert [(record.seed, record.attempt) for record in records] == [
        (1, 7),
        (2, 7),
        (3, 7),
    ]


def test_sample_round_accepts_exact_seed_coverage() -> None:
    runner = _Runner([11, 22, 33], steps_per_scenario=10)
    group = _RoundGroup(runner)
    algo = SimpleNamespace(eval_env_runner_group=group)

    records = sample_round(algo, scenarios=[11, 22, 33], attempt=1, rng_seed=11)
    assert sorted(record.seed for record in records) == [11, 22, 33]


def test_sample_round_names_a_missing_seed() -> None:
    class _StaticGroup:
        def num_healthy_remote_workers(self) -> int:
            return 1

        def foreach_env_runner(self, func, *, local_env_runner: bool):
            return [[_record(1)]]

    algo = SimpleNamespace(eval_env_runner_group=_StaticGroup())

    with pytest.raises(RuntimeError, match=r"missing=\[2\]"):
        sample_round(algo, scenarios=[1, 2], attempt=1, rng_seed=11)


def test_sample_round_names_unexpected_and_duplicate_seeds() -> None:
    class _StaticGroup:
        def num_healthy_remote_workers(self) -> int:
            return 1

        def foreach_env_runner(self, func, *, local_env_runner: bool):
            return [[_record(1), _record(2), _record(2), _record(99)]]

    algo = SimpleNamespace(eval_env_runner_group=_StaticGroup())

    with pytest.raises(
        RuntimeError,
        match=r"unexpected=\[99\].*duplicates=\[2\]",
    ):
        sample_round(algo, scenarios=[1, 2], attempt=1, rng_seed=11)


def test_sample_round_uses_local_runner_when_no_remote_workers() -> None:
    runner = _Runner([1], steps_per_scenario=10)
    group = _RoundGroup(runner, healthy_remote_workers=0)
    algo = SimpleNamespace(eval_env_runner_group=group)

    sample_round(algo, scenarios=[1], attempt=1, rng_seed=11)

    assert group.local_env_runner is True


def test_shard_walk_stops_once_the_slice_is_covered() -> None:
    """Sampling continues only while scenarios are missing, plus the final chunk."""
    runner = _Runner(list(range(20)), steps_per_scenario=1)

    _sample_shard(runner, rng_seed=7)

    budget = sum(call["num_timesteps"] for call in runner.calls)
    assert budget < 20 * HORIZON, "a covered slice kept sampling to the worst-case budget"


def test_shard_retains_incomplete_without_replaying_dropped_scenario() -> None:
    """An episode RLlib drops leaves a hole only another lap of the slice reaches."""
    runner = _Runner([11, 22, 33], steps_per_scenario=10, drop={22})

    records = _sample_shard(runner, rng_seed=7)

    assert sorted(record.seed for record in records) == [11, 22, 33]


def test_shard_walk_returns_one_incomplete_record_per_requested_seed():
    import json

    runner = _Runner([11, 22], steps_per_scenario=HORIZON + 10)
    records = _sample_shard(runner, rng_seed=7)
    assert [r.seed for r in records] == [11, 22]
    for record in records:
        assert record.completed_result_json is not None
        assert not json.loads(record.completed_result_json)["diagnostics"]["complete"]
    assert len(runner.calls) == 2 * (HORIZON + 2)
    assert all(not call["force_reset"] for call in runner.calls)
