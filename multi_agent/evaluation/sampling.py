"""Transport requested immutable attempts from their owning runner processes."""

from collections import Counter
from dataclasses import replace
from functools import partial
from uuid import uuid4

from atc_env.completed_scenario import (
    CompletedScenarioCollector,
    CompletedScenarioResult,
    capture_environment,
)
from common.env_utils import get_base_env
from common.seeding import seed_all
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import classify_outcome


def start_attempt(runner, env, *, seed, attempt_id):
    """Queue one attempt identity; the runner's next sample resets into it."""
    env.completed_collector = CompletedScenarioCollector()
    env.requested_attempt_id = attempt_id
    env._next_attempt_id = attempt_id
    env._next_attempt_seed = seed
    runner._needs_initial_reset = True


def sample_attempt(runner, env, *, seed, attempt_id, explore):
    """Run one requested attempt to its terminal capture on its owning runner."""
    start_attempt(runner, env, seed=seed, attempt_id=attempt_id)
    return run_completed_scenario(
        runner,
        env.completed_collector,
        attempt_id=attempt_id,
        explore=explore,
        max_sample_steps=int(env.policy_time_limit_steps) + 2,
    )


def sample_completed_shard(runner, *, rng_seed, transport=None):
    """Sample this runner's seed slice once; ``transport`` maps rows in-process."""
    seed_all(rng_seed + runner.worker_index)
    env = get_base_env(runner.env)
    results = [
        sample_attempt(
            runner, env, seed=seed, attempt_id=uuid4().hex, explore=bool(runner.config.explore)
        )
        for seed in env._eval_seeds
    ]
    return results if transport is None else [transport(result) for result in results]


def _result_seed(result):
    return result.diagnostics.get("seed")


def sample_completed_round(algorithm, *, seeds, rng_seed, transport=None, seed_of=_result_seed):
    """One pass over *seeds* across the evaluation runners; exact coverage or raise."""
    group = algorithm.eval_env_runner_group
    records = [
        record
        for shard in group.foreach_env_runner(
            partial(sample_completed_shard, rng_seed=rng_seed, transport=transport),
            local_env_runner=group.num_healthy_remote_workers() == 0,
        )
        for record in shard
    ]
    expected = Counter(seeds)
    found = Counter(seed_of(record) for record in records)
    if found != expected:
        duplicates = sorted(seed for seed, count in found.items() if count > expected[seed] > 0)
        raise RuntimeError(
            "completed attempt transport has missing or unexpected scenario identities: "
            f"covered {len(found.keys() & expected.keys())} of {len(expected)} scenarios "
            f"in {len(records)} episodes; missing={sorted(expected.keys() - found.keys())} "
            f"unexpected={sorted(found.keys() - expected.keys())} duplicates={duplicates}"
        )
    return records


def run_completed_scenario(
    runner, collector, *, attempt_id: str, explore: bool, max_sample_steps: int
) -> CompletedScenarioResult:
    """Continue the same recurrent/stochastic attempt; never reset or replay it."""
    if type(max_sample_steps) is not int or max_sample_steps <= 0:
        raise ValueError("positive sample budget required")
    error = "sample_budget_exhausted"
    from training.env_runner import ATCEnvRunner

    # Owned recurrent connectors cannot safely resume every partial tail. One
    # bounded call preserves continuity; terminal capture survives any later tail.
    chunk = max_sample_steps if isinstance(runner, ATCEnvRunner) else 1
    error_traceback = None

    def transported(result):
        from common.env_utils import get_base_env

        env = get_base_env(runner.env) if getattr(runner, "env", None) is not None else None
        return replace(
            result,
            diagnostics={
                **result.diagnostics,
                "sampling": {
                    "requested_attempt_id": attempt_id,
                    "sample_step_budget": max_sample_steps,
                    "overrun_after_capture": env is not None
                    and getattr(env, "attempt_id", attempt_id) != attempt_id,
                    "tail_error": error if error.startswith("sampling_error:") else None,
                    "tail_error_traceback": error_traceback,
                },
            },
        )

    for _ in range(0, max_sample_steps, chunk):
        result = collector.take(attempt_id)
        if result is not None:
            return transported(result)
        try:
            runner.sample(num_timesteps=chunk, force_reset=False, explore=explore)
        except Exception as exc:
            import traceback

            error = f"sampling_error:{type(exc).__name__}:{exc}"
            error_traceback = traceback.format_exc()
            break
    result = collector.take(attempt_id)
    if result is not None:
        return transported(result)
    from common.env_utils import get_base_env

    pending_seed = None
    if getattr(runner, "env", None) is not None:
        env = get_base_env(runner.env)
        pending_seed = getattr(env, "_next_attempt_seed", None)
        if getattr(env, "attempt_id", None) == attempt_id and getattr(
            env, "_has_reset_once", False
        ):
            capture_environment(env, complete=False, error=error, error_traceback=error_traceback)
            captured = collector.take(attempt_id)
            assert captured is not None
            return captured
    live = OutcomeSnapshot((), (), (), (), False)
    return CompletedScenarioResult(
        attempt_id,
        classify_outcome(live, None, threshold_s=60, policy_limit_reached=True),
        live,
        None,
        (),
        0.0,
        {"complete": False, "error": error, "seed": pending_seed},
    )
