"""Authenticated weights-only reader; never a training resume or phase bridge."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from evaluation.seeds import SeedAuthorizer
from training.checkpoint_manifest import checkpoint_digest
from training.checkpoints import inspect_checkpoint, read_checkpoint_payload
from training.env import TrainingATCEnv
from training.env_runner import ATCEnvRunner
from training.ppo_config import build_ppo_config, shared_policy
from training.run_config import ResolvedRunConfig
from training.use_case import use_case_of, with_use_case


def inspect_inference_checkpoint(checkpoint) -> ResolvedRunConfig:
    """Weights-only: same version tokens and intact files; the saved config is the run."""
    from training.inference_bundle import resolve_scenario

    return resolve_scenario(inspect_checkpoint(Path(checkpoint).resolve(), scope="weights").run)


def bind_use_case(run: ResolvedRunConfig, name: str) -> ResolvedRunConfig:
    """The saved run, rebound to another use case only for diagnostic benchmark worlds."""
    return run if use_case_of(run.scenario) == name else with_use_case(run, name)


class InferenceCheckpointEnvMixin:
    """Accept an inference checkpoint only when its saved env contract matches."""

    evaluation_mode: bool
    _reserved_seeds: Callable[[], frozenset[int]]
    allows_use_case_rebind = False

    def _verify_inference_checkpoint(self, checkpoint, payload) -> None:
        run = inspect_inference_checkpoint(checkpoint)
        if self.allows_use_case_rebind:
            run = bind_use_case(run, use_case_of(payload["scenario"]))
        if not self.evaluation_mode or payload != run.env_payload():
            raise ValueError("inference saved environment contract mismatch")

    def _development_gate(self):
        """Operational inference has no development access to reserved identities."""
        return None

    def _authorize_evaluation_seeds(self, seeds) -> None:
        SeedAuthorizer(self._reserved_seeds(), self._development_gate()).authorize(seeds)


class InferenceATCEnv(InferenceCheckpointEnvMixin, TrainingATCEnv):
    """Bound environment reader for operational evaluation of a saved checkpoint."""


def build_inference_algorithm(
    checkpoint,
    *,
    eval_env_config,
    callbacks,
    timestep_budget,
    seed=None,
    explore=False,
    num_env_runners=0,
    benchmark_manifest=None,
    inference_device="cpu",
    gpus_per_runner=0.0,
    env_runner_cls=ATCEnvRunner,
):
    """Preflight full original contract, then copy only mandatory module state.

    ``env_runner_cls`` may only specialise the owned runner (e.g. the rollout-search runner).
    """
    if not issubclass(env_runner_cls, ATCEnvRunner):
        raise ValueError("inference runners must specialise the owned ATCEnvRunner")
    from ray.rllib.algorithms.ppo import PPOConfig

    from evaluation.inference_resources import (
        InferenceResources,
        observe_placement,
        validate_placement,
    )
    from evaluation.loading import sync_eval_runner_weights

    resources = InferenceResources(inference_device, num_env_runners, gpus_per_runner)
    resources.preflight()
    if inference_device == "cuda":
        import ray

        total = ray.cluster_resources().get("GPU", 0)
        available = ray.available_resources().get("GPU", 0)
        if resources.ray_gpus > total or available < total:
            raise ValueError("CUDA inference requires sufficient unallocated Ray GPUs")
    run = trained = inspect_inference_checkpoint(checkpoint)
    environment_class = InferenceATCEnv
    evaluation_count = run.scenario["aircraft_count"]
    if benchmark_manifest is not None:
        from evaluation.benchmark.worlds import BenchmarkEnv, validate_development_manifest

        validate_development_manifest(benchmark_manifest)
        contract = benchmark_manifest["contract"]["compatibility"]
        from evaluation.benchmark.binding import benchmark_aircraft_count
        from evaluation.benchmark.use_cases import selection

        run = bind_use_case(trained, selection(contract)[0])

        evaluation_count = benchmark_aircraft_count(run.env_payload(), contract)
        environment_class = BenchmarkEnv
    if type(num_env_runners) is not int or num_env_runners < 0:
        raise ValueError("inference runner count must be a nonnegative integer")
    options = dict(eval_env_config)
    permitted = {
        "evaluation_mode",
        "max_aircraft_count",
        "max_steps_per_episode",
        "policy_time_limit_steps",
        "airport",
        "simulation_config_path",
        "success_time_dev_s",
        "evaluation_seeds",
        "evaluation_seed_shards",
        "evaluation_randomize_scenario_count",
    }
    if set(options) - permitted:
        raise ValueError("incompatible inference feature/semantic override")
    for name, saved in (
        ("max_aircraft_count", evaluation_count),
        ("max_steps_per_episode", run.scenario["episode_steps"]),
        ("airport", run.scenario["airport"]),
        ("success_time_dev_s", 60.0),
    ):
        if name in options and options[name] is not None and options[name] != saved:
            raise ValueError(f"incompatible inference override: {name}; saved value is {saved}")
        options.pop(name, None)
    if "simulation_config_path" in options:
        from training.run_config import scenario_config_digest

        if (
            scenario_config_digest(Path(options.pop("simulation_config_path")))
            != run.scenario["config_sha256"]
        ):
            raise ValueError("incompatible inference world configuration")
    if options.get("evaluation_randomize_scenario_count", 0):
        raise ValueError("authenticated inference requires an explicit complete seed manifest")
    original = build_ppo_config(run)
    env = {
        **original.env_config,
        **options,
        "evaluation_mode": True,
        "inference_checkpoint": str(Path(checkpoint).resolve()),
        "inference_topology": {"env_runners": num_env_runners, "envs_per_runner": 1},
        "scenario_artifact_dir": str(Path(checkpoint).parent / "inference-scenarios"),
    }
    if benchmark_manifest is not None:
        env["benchmark_manifest"] = benchmark_manifest
    cfg = (
        PPOConfig()
        .framework("torch")
        .environment(environment_class, env_config=env, disable_env_checking=True)
        .env_runners(
            env_runner_cls=env_runner_cls,
            num_env_runners=0,
            num_envs_per_env_runner=1,
            rollout_fragment_length=run.scenario["episode_steps"],
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .multi_agent(policies={"shared_policy"}, policy_mapping_fn=shared_policy)
        .rl_module(rl_module_spec=original.rl_module_spec)
        .callbacks(callbacks)
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=num_env_runners,
            evaluation_duration=timestep_budget or 1,
            evaluation_duration_unit="timesteps" if timestep_budget else "episodes",
            evaluation_config={
                "env_config": {"evaluation_mode": True},
                "explore": explore,
                "num_gpus_per_env_runner": gpus_per_runner,
            },
        )
        .fault_tolerance(
            restart_failed_env_runners=False,
            ignore_env_runner_failures=False,
            max_num_env_runner_restarts=0,
        )
    )
    if seed is not None:
        cfg = cfg.debugging(seed=seed)
    cfg.runtime_env = {}
    algorithm = cast(Any, cfg.build_algo())
    try:
        state, _ = read_checkpoint_payload(checkpoint)
        weights = state["learner_group"]["learner"]["rl_module"]
        algorithm.learner_group.set_state({"learner": {"rl_module": weights}})
        algorithm.env_runner_group.sync_weights(
            from_worker_or_learner_group=algorithm.learner_group, inference_only=True
        )
        sync_eval_runner_weights(algorithm)
        evaluation_group = algorithm.eval_env_runner_group
        placements = evaluation_group.foreach_env_runner(
            observe_placement, local_env_runner=num_env_runners == 0
        )
        validate_placement(resources, placements)
        for group in (algorithm.env_runner_group, algorithm.eval_env_runner_group):
            group.foreach_env_runner(
                lambda runner: runner.activate_environments(run.semantic_digest)
            )
        algorithm.inference_provenance = {
            "checkpoint_sha256": checkpoint_digest(checkpoint),
            "training_config": trained.to_dict(),
            "runtime_difference": {
                "inference_device": inference_device,
                "gpus_per_runner": gpus_per_runner,
                "observed_evaluation_placement": placements,
                "installed_backend_preserved": run.resources["backend"],
            },
            "adapter": "authenticated_weights_only_v1",
            "training_semantic_digest": trained.semantic_digest,
            "operational_threshold_s": 60,
            "training_threshold_s": run.thresholds["training_success_s"],
            "inference_topology": {"evaluation_runners": num_env_runners, "local_learner": True},
            **(
                {}
                if env_runner_cls is ATCEnvRunner
                else {"env_runner": f"{env_runner_cls.__module__}.{env_runner_cls.__qualname__}"}
            ),
            "training_resume": False,
            **(
                {
                    "generator_variant": benchmark_manifest["contract"]["compatibility"][
                        "generator_variant"
                    ]
                }
                if benchmark_manifest is not None
                and "generator_variant" in benchmark_manifest["contract"]["compatibility"]
                else {}
            ),
            "use_case": {
                "saved_training": use_case_of(trained.scenario),
                "evaluation": use_case_of(run.scenario),
                "diagnostic_override": run is not trained,
            },
            "aircraft_count": {
                "saved_training": run.scenario["aircraft_count"],
                "evaluation": evaluation_count,
                "diagnostic_override": evaluation_count != run.scenario["aircraft_count"],
            },
        }
        return algorithm
    except BaseException:
        algorithm.stop()
        raise
