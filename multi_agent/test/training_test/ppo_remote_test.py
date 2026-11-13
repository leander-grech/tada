"""Real remote runner/learner construction, activation, and instance consumption."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import ray

from common.runtime_log import runtime_print
from training.run_config import load_run_config


def _make_observer():
    def _observe_runner(runner):
        import numpy as np

        from models.observations import make_agent_observations_from_simulator

        env = runner.env.envs[0].unwrapped
        initial = env.get_training_stream_state()
        observations, _ = env.reset()
        callsigns = [a.call_sign for a in env.simulator.world.aircraft]
        proxies = {name: 80.0 for name in callsigns}
        context = env._build_global_context(
            forecast_infringements={}, deviation_proxies=proxies, current_time_s=0
        )
        encoded = make_agent_observations_from_simulator(
            env.simulator,
            callsigns,
            {},
            proxies,
            env._route_trombone_store,
            global_context=context,
            settings=env.feature_settings,
            scales=env.settings.core.observation,
            action_settings=env.settings.core.action,
        )
        assert observations
        assert all(
            np.isfinite(o["observations"]["aircraft_scalar"]).all() for o in encoded.values()
        )
        return {
            "initial": initial,
            "state": env.get_training_stream_state(),
            "seed": env.scenario_seed,
            "aircraft_count": len(env.simulator.world.aircraft),
            "threshold": env.success_time_dev_s,
            "semantic_digest": env.consumed_semantic_digest,
            "encoding": env.feature_settings.deviation_encoding,
            "neighbors": env.feature_settings.neighbor_selection,
            "actor_value": next(iter(encoded.values()))["observations"]["aircraft_scalar"][
                5
            ].item(),
            "journal": str(env._scenario_artifact_path),
        }

    return _observe_runner


def _check_two_configs(native_run, config_path):
    from config.config import Config
    from training.ppo_config import build_algorithm
    from training.provenance import write_run_manifest
    from training.runtime_gate import verify_algorithm_selection

    before = (Config.SATISFACTORY_TIME_DEV, Config.SEED, Config.GRU_HIDDEN_SIZE)
    ray.init(
        address="local",
        num_cpus=8,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
    )
    results = []
    try:
        for index in range(2):
            run = native_run
            if index:
                data = json.loads(config_path.read_text())
                data["run_id"] = "fixture-second"
                data["features"].update(
                    deviation_encoding="linear", linear_scale_s=200.0
                )
                data["features"]["dimensions"]["max_neighbors"] = 3
                data["scenario"]["aircraft_count"] = 3
                data["thresholds"]["training_success_s"] = 45.0
                data["reward"]["success_bonus"] = 7.0
                data["optimizer"].update(learning_rate=0.0003, entropy_coeff=0.023)
                config_path.write_text(json.dumps(data))
                run = load_run_config(config_path)
            algorithm = build_algorithm(run)
            try:
                assert algorithm.env_runner_group is not None
                assert algorithm.eval_env_runner_group is not None
                proof = verify_algorithm_selection(algorithm, run)
                assert proof["selection_verified"]
                assert any(a["role"] == "learner" for a in proof["actors"])
                for actor in proof["actors"]:
                    assert (
                        actor["pid"] != __import__("os").getpid() or actor.get("worker_index") == 0
                    )
                train = algorithm.env_runner_group.foreach_env_runner(
                    _make_observer(), local_env_runner=False
                )[0]
                evaluation = algorithm.eval_env_runner_group.foreach_env_runner(
                    _make_observer(), local_env_runner=False
                )[0]
                assert train["threshold"] == run.thresholds["training_success_s"]
                assert evaluation["threshold"] == 60
                assert (
                    train["semantic_digest"] == evaluation["semantic_digest"] == run.semantic_digest
                )
                assert (
                    train["aircraft_count"]
                    == evaluation["aircraft_count"]
                    == run.scenario["aircraft_count"]
                )
                assert (
                    train["initial"]["stream"]["cursor"]
                    == evaluation["initial"]["stream"]["cursor"]
                    == 0
                )
                train_state = algorithm.env_runner_group.foreach_env_runner(
                    lambda r: r.env.envs[0].unwrapped.get_training_stream_state(),
                    local_env_runner=False,
                )[0]
                assert train_state == train["state"], (
                    "operational evaluation advanced training cursor"
                )
                assert train["state"]["stream"]["domain"] == "training"
                assert evaluation["state"]["stream"]["domain"] == "operational_evaluation"
                manifest = Path(run.execution["output_root"]) / f"manifest-{index}.json"
                write_run_manifest(manifest, run, {}, algorithm=algorithm)
                assert json.loads(manifest.read_text())["runtime"]["owned_learning"][
                    "selection_verified"
                ]
                results.append((train, evaluation, proof))
            finally:
                algorithm.stop()
        assert results[0][0]["semantic_digest"] != results[1][0]["semantic_digest"]
        assert results[0][0]["actor_value"] != results[1][0]["actor_value"]
        reserved = json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
        seeds = {
            r["seed"] for role in ("validation", "heldout", "replacements") for r in reserved[role]
        }
        assert len(seeds) == 500
        events = []
        for train, evaluation, _ in results:
            for report in (train, evaluation):
                journal = [
                    json.loads(line) for line in Path(report["journal"]).read_text().splitlines()
                ]
                assert {e["status"] for e in journal} == {"drawn", "generated", "consumed"}
                assert all(e["seed"] not in seeds for e in journal)
                assert all(
                    "world_sha256" in e and "planning_world_sha256" in e
                    for e in journal
                    if e["status"] == "consumed"
                )
                events.extend(journal)
        runtime_print("OwnedPPO", "remote instance evidence", results=results, events=events)
    finally:
        ray.shutdown()
    assert before == (Config.SATISFACTORY_TIME_DEV, Config.SEED, Config.GRU_HIDDEN_SIZE)


def _check_wrong_stack(native_run, wrong_component):
    from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
    from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

    from training.ppo_config import build_ppo_config
    from training.runtime_gate import activate_algorithm

    ray.init(
        address="local",
        num_cpus=8,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
    )
    algorithm = None
    try:
        cfg = build_ppo_config(native_run)
        if wrong_component == "runner":

            class UnownedRunner(MultiAgentEnvRunner):
                # Ray instruments the selected method's signature in place.
                # Own this test method so the stock base guard stays untouched.
                def _try_env_step(self, actions):
                    return super()._try_env_step(actions)

            cfg.env_runners(env_runner_cls=UnownedRunner)
        elif wrong_component == "learner":
            cfg.learners(learner_class=PPOTorchLearner)
        algorithm = cfg.build_algo()
        assert algorithm.learner_group is not None
        if wrong_component == "gae":

            def replace_gae(learner, *_args):
                from ray.rllib.connectors.learner import GeneralAdvantageEstimation

                learner._learner_connector.connectors[-1] = GeneralAdvantageEstimation(
                    gamma=learner.config.gamma, lambda_=learner.config.lambda_
                )

            for result in algorithm.learner_group.foreach_learner(replace_gae):
                assert result.ok, result.get()
        with pytest.raises(Exception, match="actual.*class.*owned|actual learner connector stack"):
            activate_algorithm(algorithm, native_run)
        for group in (algorithm.env_runner_group, algorithm.eval_env_runner_group):
            assert group is not None
            assert group.foreach_env_runner(
                lambda r: r.env.envs[0].unwrapped.simulator.world is None,
                local_env_runner=False,
            ) == [True]
    finally:
        try:
            if algorithm is not None:
                algorithm.stop()
        finally:
            ray.shutdown()


def _check_actor_contracts(native_run, mutation):
    from training.ppo_config import build_ppo_config
    from training.provenance import write_run_manifest
    from training.runtime_gate import activate_algorithm

    ray.init(
        address="local",
        num_cpus=8,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
    )
    algorithm = None
    try:
        cfg = build_ppo_config(native_run)
        if mutation == "evaluation-mode":
            assert isinstance(cfg.evaluation_config, dict)
            cfg.evaluation_config["env_config"]["evaluation_mode"] = False
        algorithm = cfg.build_algo()
        if mutation == "live-manifest":
            manifest = Path(native_run.execution["output_root"]) / "manifest.json"
            write_run_manifest(manifest, native_run, {}, algorithm=algorithm)
            assert json.loads(manifest.read_text())["runtime"]["owned_learning"][
                "selection_verified"
            ]
        else:
            if mutation != "evaluation-mode":

                def corrupt(runner):
                    env = runner.env.envs[0].unwrapped
                    stream = env._training_stream
                    if mutation == "duplicate":
                        if runner.worker_index == 2:
                            runner.worker_index = 1
                            env._worker_index = 1
                            stream.stream_id = (stream.stream_id[0], 1, stream.stream_id[2])
                    elif mutation == "threshold":
                        env.success_time_dev_s = 59
                    elif mutation == "domain":
                        stream.domain = "training"
                    elif mutation == "root":
                        stream.root += 1
                    elif mutation == "reservation":
                        stream.reserved_manifest_hash = "0" * 64
                    else:
                        position = {"phase": 0, "runner-index": 1, "vector": 2}[mutation]
                        ids = list(stream.stream_id)
                        ids[position] += 1
                        stream.stream_id = tuple(ids)

                group = (
                    algorithm.env_runner_group
                    if mutation == "duplicate"
                    else algorithm.eval_env_runner_group
                )
                assert group is not None
                group.foreach_env_runner(corrupt, local_env_runner=False)
            with pytest.raises(
                Exception, match="actual.*(role|threshold|stream|source|identity|topology)"
            ):
                activate_algorithm(algorithm, native_run)
        for group in (algorithm.env_runner_group, algorithm.eval_env_runner_group):

            def inspect_guard(runner):
                env = runner.env.envs[0].unwrapped
                return (
                    env.simulator.world is None,
                    env.get_training_stream_state()["stream"]["cursor"],
                    env._runtime_activated,
                    env._scenario_artifact_path.exists(),
                )

            assert group is not None
            guards = group.foreach_env_runner(
                inspect_guard, local_env_runner=False, healthy_only=False
            )
            assert guards == [(True, 0, False, False)] * (
                native_run.topology["env_runners"] if group is algorithm.env_runner_group else 1
            )
            runtime_print("AuditC", "guard observation", mutation=mutation, guards=guards)
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


@pytest.mark.parametrize(
    "mutation",
    [
        "evaluation-mode",
        "threshold",
        "domain",
        "root",
        "phase",
        "runner-index",
        "vector",
        "reservation",
        "live-manifest",
        "duplicate",
    ],
)
def test_actual_actor_contract_rejects_before_generation(native_run, config_path, mutation):
    if mutation == "duplicate":
        data = json.loads(config_path.read_text())
        data["topology"]["env_runners"] = 2
        config_path.write_text(json.dumps(data))
    _run_cpu_diagnostic(config_path, "contract:" + mutation)


def _check_recurrent_minibatch(native_run):
    from training.ppo_config import build_algorithm

    ray.init(
        address="local",
        num_cpus=8,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
    )
    algorithm = None
    try:
        algorithm = build_algorithm(native_run)
        algorithm.train()
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


def test_recurrent_minibatch_tiny_remote_configuration(native_run, config_path):
    data = json.loads(config_path.read_text())
    data["optimizer"].update(train_batch_size=8, minibatch_size=1, epochs=2)
    data["features"]["architecture"]["max_seq_len"] = 3
    config_path.write_text(json.dumps(data))
    try:
        load_run_config(config_path)
    except ValueError as error:
        assert "minibatch" in str(error)
        return
    _run_cpu_diagnostic(config_path, "recurrent-minibatch")


def _run_cpu_diagnostic(config_path, mode):
    result = subprocess.run(
        [sys.executable, __file__, str(config_path), mode],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=180,
    )
    runtime_print(
        "OwnedPPO",
        "CPU-only child diagnostic",
        mode=mode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_instance_configs_in_real_remote_actors(native_run, config_path):
    _run_cpu_diagnostic(config_path, "two-configs")


@pytest.mark.parametrize("wrong_component", ["runner", "learner", "gae"])
def test_wrong_actual_stack_cannot_release_generation(native_run, config_path, wrong_component):
    _run_cpu_diagnostic(config_path, wrong_component)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    run = load_run_config(path)
    if sys.argv[2].startswith("contract:"):
        _check_actor_contracts(run, sys.argv[2].split(":", 1)[1])
    elif sys.argv[2] == "recurrent-minibatch":
        _check_recurrent_minibatch(run)
    elif sys.argv[2] == "two-configs":
        _check_two_configs(run, path)
    else:
        _check_wrong_stack(run, sys.argv[2])
