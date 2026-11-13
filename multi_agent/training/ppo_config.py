"""Construct fixed, instance-bound PPO and activate its verified owned actors."""

from pathlib import Path

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from callbacks import build_training_callbacks
from network.rlm import ATCRLM
from training.algorithm import ATCPPO
from training.env import TrainingATCEnv
from training.env_runner import ATCEnvRunner
from training.learner_checkpoint import ATCPPOLearner
from training.run_config import ResolvedRunConfig

POLICY_ID = "shared_policy"


def shared_policy(_agent_id, _episode, **_kwargs):
    """One shared policy independent of process-global experiment settings."""
    return POLICY_ID


def build_ppo_config(
    run: ResolvedRunConfig, *, scenario_artifact_dir: Path | None = None
) -> PPOConfig:
    """Keep construction lazy; only build_algorithm releases native generation."""
    if run.resources["num_learners"] != 1 or run.topology["envs_per_runner"] != 1:
        raise ValueError("owned PPO supports one learner and one environment per runner")
    if run.topology["env_runners"] < 1:
        raise ValueError("owned PPO requires at least one remote environment runner")
    artifacts = scenario_artifact_dir or (
        Path(run.execution["output_root"]) / run.run_id / run.phase_id / "scenarios"
    )
    env_config = {
        "resolved_env": run.env_payload(),
        "resolved_metrics": run.metrics_payload(),
        "expected_semantic_digest": run.semantic_digest,
        "scenario_artifact_dir": str(artifacts),
        "require_runtime_activation": True,
        "evaluation_mode": False,
    }
    optimizer = run.optimizer
    return (
        PPOConfig(algo_class=ATCPPO)
        .callbacks(build_training_callbacks())
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .framework("torch")
        .debugging(seed=run.root_seed)
        .environment(TrainingATCEnv, env_config=env_config, disable_env_checking=True)
        .env_runners(
            env_runner_cls=ATCEnvRunner,
            create_env_on_local_worker=False,
            num_env_runners=run.topology["env_runners"],
            num_envs_per_env_runner=run.topology["envs_per_runner"],
            num_cpus_per_env_runner=run.resources["cpus_per_runner"],
            num_gpus_per_env_runner=run.resources.get("gpus_per_runner", 0),
            rollout_fragment_length=run.scenario["episode_steps"],
            batch_mode="complete_episodes",
            sample_timeout_s=None,
        )
        .resources(num_cpus_for_main_process=1)
        .fault_tolerance(restart_failed_env_runners=False, max_num_env_runner_restarts=0)
        .learners(
            learner_class=ATCPPOLearner,
            num_learners=run.resources["num_learners"],
            num_cpus_per_learner=run.resources["cpus_per_learner"],
            num_gpus_per_learner=run.resources["gpus_per_learner"],
        )
        .multi_agent(
            policies={POLICY_ID},
            policies_to_train=[POLICY_ID],
            policy_mapping_fn=shared_policy,
            count_steps_by="env_steps",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    POLICY_ID: RLModuleSpec(
                        module_class=ATCRLM,
                        model_config={
                            "resolved_model": run.model_payload(),
                            "expected_semantic_digest": run.semantic_digest,
                            "max_seq_len": run.features["architecture"]["max_seq_len"],
                        },
                    ),
                }
            )
        )
        .training(
            lr=optimizer["learning_rate"],
            entropy_coeff=optimizer["entropy_coeff"],
            use_kl_loss=False,
            gamma=optimizer["gamma"],
            lambda_=optimizer["gae_lambda"],
            clip_param=optimizer["clip_param"],
            vf_clip_param=optimizer["vf_clip_param"],
            vf_loss_coeff=optimizer["vf_loss_coeff"],
            grad_clip=optimizer["grad_clip"],
            train_batch_size_per_learner=optimizer["train_batch_size"],
            minibatch_size=optimizer["minibatch_size"],
            num_epochs=optimizer["epochs"],
        )
        .evaluation(
            evaluation_interval=run.cadences["evaluation_iterations"],
            evaluation_num_env_runners=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
            evaluation_config={"explore": False, "env_config": {"evaluation_mode": True}},
        )
    )


def build_algorithm(
    run: ResolvedRunConfig,
    *,
    scenario_artifact_dir: Path | None = None,
    defer_activation: bool = False,
) -> Algorithm:
    """Verify the selected native artifact, build lazy actors, then prove and release."""
    from training.resource_profile import resolve_resource_profile, runtime_inventory
    from training.runtime_gate import (
        activate_algorithm,
        verify_algorithm_selection,
        verify_native_conditions,
    )

    profile = resolve_resource_profile(run, runtime_inventory())
    verify_native_conditions(run)
    algorithm = build_ppo_config(run, scenario_artifact_dir=scenario_artifact_dir).build_algo()
    try:
        setattr(algorithm, "_requested_resource_profile", profile)
        if defer_activation:
            verify_algorithm_selection(algorithm, run)
        else:
            activate_algorithm(algorithm, run)
        return algorithm
    except BaseException:
        algorithm.stop()
        raise
