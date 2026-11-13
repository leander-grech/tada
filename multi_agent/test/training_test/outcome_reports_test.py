"""Observe new outcome reducers in two actual owned Algorithm reports."""

import json
import os
import subprocess
import sys
from pathlib import Path

from common.runtime_log import runtime_print


def test_two_native_multi_runner_outcome_reports(native_run, config_path):
    data = json.loads(config_path.read_text())
    data["topology"]["env_runners"] = 2
    data["scenario"]["episode_steps"] = 64
    data["optimizer"].update(train_batch_size=128, minibatch_size=32)
    data["reward"].update(realized_failure_cost=7.5, success_bonus=1.25)
    config_path.write_text(json.dumps(data))
    child = subprocess.run(
        [sys.executable, __file__, str(config_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=240,
    )
    runtime_print("OutcomeReports", "two native reports", stdout=child.stdout, stderr=child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr


def _observe_runner(runner):
    env = runner.env.envs[0].unwrapped
    records = getattr(runner, "_task11_completed", [])
    runner._task11_completed = []
    return {
        "records": records,
        "worker": runner.worker_index,
        "reward": dict(env.reward_settings),
        "fields": list(env.consumed_config_fields),
    }


def _instrument_runner(runner):
    from atc_env.episode_metrics import summarize_actions, summarize_outcome
    from callbacks.metrics_callback import ATCEpisodeMetricsCallback
    from common.env_utils import get_base_env

    runner._task11_completed = []
    owned_sample = runner.sample

    def controlled_complete_workload(**kwargs):
        # Controlled episode-budget intervention: real owned complete native samples,
        # unchanged product callback/reducer and all non-budget sampling arguments.
        kwargs.pop("num_timesteps", None)
        kwargs["num_episodes"] = runner.worker_index
        return owned_sample(**kwargs)

    runner.sample = controlled_complete_workload
    env = runner.env.envs[0].unwrapped
    owned_step = env.step
    inject_first_failure = runner.worker_index == 1

    def first_retained_native_hazard(actions):
        nonlocal inject_first_failure
        # Native retained-event fixture, not a claim of policy-caused failure.
        if inject_first_failure:
            _inject_native_failure(env)
            inject_first_failure = False
        return owned_step(actions)

    env.step = first_retained_native_hazard
    for callback in runner._callbacks:
        if isinstance(callback, ATCEpisodeMetricsCallback):
            original = callback.on_episode_end

            def observed(*, env, episode, **kwargs):
                base = get_base_env(env)
                if base.last_outcome is not None:
                    assert episode.is_done
                    record = summarize_outcome(
                        base.outcome_ledger.snapshot(), base.last_outcome, base.last_verification
                    )
                    record["seed"] = base.scenario_seed
                    record["policy_steps"] = base.step_counter
                    record["actions"] = summarize_actions(base._action_history)
                    record["reward_components"] = dict(base.episode_reward_components)
                    runner._task11_completed.append(record)
                return original(env=env, episode=episode, **kwargs)

            callback.on_episode_end = observed


def _inject_native_failure(env):
    from flight_simulator import World

    data = json.loads(env.simulator.world.json())
    callsign = env.arrival_targets[0].callsign
    data["infringements"] = [
        {"AirspaceExit": {"timestamp": 0, "aircraft": callsign}},
        {
            "LossOfSeparation": {
                "timestamp": 0,
                "aircraft_pair": [callsign, env.arrival_targets[1].callsign],
                "horizontal_separation": 1,
                "vertical_separation": 1,
            }
        },
    ]
    env.simulator.world = World.from_json(json.dumps(data))
    env.outcome_state.observe_live(env.simulator.world)


def _probe_failure(runner):
    env = runner.env.envs[0].unwrapped
    env.reset()
    _inject_native_failure(env)
    _, rewards, terminated, _, _ = env.step({})
    expected = -7.5 * len(env.arrival_targets) / env.reward_settings["count_baseline"]
    assert terminated["__all__"] and env.last_reward.bonus == expected
    assert abs(sum(rewards.values()) - env.last_reward.total()) < 1e-9
    assert abs(sum(env.episode_reward_components.values()) - env.cumulative_reward) < 1e-9
    env.step({})
    assert env.last_reward.bonus == 0
    return {
        "worker": runner.worker_index,
        "penalty": expected,
        "components": env.episode_reward_components,
    }


def _run(path):
    import ray

    from callbacks.callback_metric_helpers import get_custom_metric_value
    from training.ppo_config import build_algorithm
    from training.run_config import load_run_config

    run = load_run_config(path)
    ray.init(num_cpus=5, include_dashboard=False)
    algorithm = None
    try:
        algorithm = build_algorithm(run)
        assert algorithm.env_runner_group is not None
        algorithm.env_runner_group.foreach_env_runner(_instrument_runner, local_env_runner=False)
        reports = []
        seen = set()
        for report_index in range(2):
            result = algorithm.train()
            evidence = algorithm.env_runner_group.foreach_env_runner(
                _observe_runner, local_env_runner=False
            )
            sizes = [len(worker["records"]) for worker in evidence]
            assert len(set(sizes)) > 1, "each report needs unequal complete native contributions"
            metrics = result["env_runners"]
            runtime_print(
                "OutcomeReports", "actual driver report", metrics=metrics, workers=evidence
            )
            count = get_custom_metric_value(result, "outcome_scenario_count")
            assert count is not None and count > 0, "missing true-terminal outcome capture"
            assert count == get_custom_metric_value(result, "episodes_end_total_count")
            assert count == sum(sizes)
            for field in (
                "live_unique_pairs",
                "live_unique_exits",
                "live_unsafe_scenario_count",
                "reason_live_airspace_exit_count",
                "reason_live_loss_of_separation_count",
            ):
                assert get_custom_metric_value(result, "outcome_" + field) == (
                    1 if report_index == 0 else 0
                ), field
            records = [record for worker in evidence for record in worker["records"]]
            seeds = {row["seed"] for row in records}
            assert len(seeds) == len(records) and not seen & seeds
            seen.update(seeds)
            assert get_custom_metric_value(result, "learned_policy_steps") == sum(
                row["policy_steps"] for row in records
            )
            for field in (
                "learned_command_count",
                "preprocessing_command_count",
                "learned_speed_count",
                "learned_vector_count",
                "learned_trombone_count",
                "learned_skip_count",
            ):
                assert get_custom_metric_value(result, "actions_" + field) == sum(
                    row["actions"][field] for row in records
                )
            for field in ("deviation", "conflict", "action", "bonus"):
                actual = get_custom_metric_value(result, "reward_component_" + field)
                assert (
                    actual is not None
                    and abs(actual - sum(row["reward_components"][field] for row in records)) < 1e-7
                )
            assert (
                get_custom_metric_value(result, "effective_learning_rate")
                == run.optimizer["learning_rate"]
            )
            assert (
                abs(
                    get_custom_metric_value(result, "effective_entropy_coeff")
                    - run.optimizer["entropy_coeff"]
                )
                < 1e-7
            )
            for field in (
                "scenario_count",
                "live_target_count",
                "live_completed_count",
                "live_exact_count",
                "live_interval_count",
                "live_unknown_count",
                "live_unique_pairs",
                "live_unique_exits",
                "verified_target_count",
                "verified_exact_count",
                "verified_interval_count",
                "verified_unknown_count",
                "verification_elapsed_simulated_seconds",
            ):
                assert get_custom_metric_value(result, "outcome_" + field) == sum(
                    row[field] for row in records
                ), field
            assert (
                sum(
                    get_custom_metric_value(result, f"outcome_{kind}_count") or 0
                    for kind in (
                        "physical_success",
                        "verified_noop_success",
                        "safe_delayed",
                        "infringement",
                        "unresolved",
                    )
                )
                == count
            )
            for scope in ("live", "verified"):
                assert sum(
                    get_custom_metric_value(result, f"outcome_{scope}_{kind}_count") or 0
                    for kind in ("exact", "interval", "unknown")
                ) == get_custom_metric_value(result, f"outcome_{scope}_target_count")
            for worker in evidence:
                assert worker["reward"]["realized_failure_cost"] == 7.5
                assert "reward.realized_failure_cost" in worker["fields"]
            reports.append(count)
        failures = algorithm.env_runner_group.foreach_env_runner(
            _probe_failure, local_env_runner=False
        )
        runtime_print(
            "OutcomeReports", "actual worker independent terminal penalties", evidence=failures
        )
        runtime_print("OutcomeReports", "consecutive scenario denominators", counts=reports)
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    _run(Path(sys.argv[1]))
