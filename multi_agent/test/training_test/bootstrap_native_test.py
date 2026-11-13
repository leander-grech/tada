"""Selected native simulator and ATCRLM through the actual owned PPO boundaries."""

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from flight_simulator import World, simulate_n_next_states
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.evaluation.postprocessing import Postprocessing
from support.bootstrap import shared_mapping

from atc_env.env_config import observation_settings_from_config
from atc_env.environment import ATCEnv
from common.runtime_log import runtime_print
from config.config import Config
from models.observation_space import build_observation_space
from network.rlm import ATCRLM
from training.env_runner import ATCEnvRunner
from training.learner_checkpoint import ATCPPOLearner
from training.run_config import load_run_config

pytestmark = pytest.mark.usefixtures("ray_env_registry")


class NativeBootstrapEnv(ATCEnv):
    """Inject a disjoint native fixture using the existing evaluation-world API."""

    def __init__(self, config):
        super().__init__(config)
        world = World.from_json(config["bootstrap_world"])
        if config.get("calibrate_prepared_arrival"):
            # Imported worlds undergo route exclusion in the real reset path.
            # Calibrate this fixture's target against that actual native route.
            world = self.simulator.prepare_world_for_episode(world)
            finished = simulate_n_next_states(10000, 1000, world, [], None)
            assert finished.aircraft[0].mission.completed
            data = json.loads(world.json())
            data["arrival_targets"][0]["eta_s"] = finished.age_in_seconds
            world = World.from_json(json.dumps(data))
        self.set_next_world(world)

    def step(self, action_dict):
        # A no-op fixture trajectory, with real native advance/reward/done code.
        result = super().step({aid: 0 for aid in action_dict})
        self.raw_bootstrap_flags = (dict(result[2]), dict(result[3]))
        return result


@pytest.mark.parametrize(
    "mode", ["policy_cap", "airborne_verified", "physical_completion", "physical_failure"]
)
def test_native_outcome_flags_reach_actual_atcrlm_targets(mode, native_world, config_path):
    world = native_world(error=0, start=20)
    if mode == "physical_completion":
        end = simulate_n_next_states(10000, 1000, world, [], None)
        remaining = int(end.age_in_seconds - world.age_in_seconds) - 1
        world = simulate_n_next_states(remaining, 1000, world, [], None)
        assert not world.aircraft[0].mission.completed
    elif mode == "physical_failure":
        data = json.loads(world.json())
        data["aircraft"][0]["mission"]["failed"] = True
        world = World.from_json(json.dumps(data))
    resolved = load_run_config(config_path)
    cfg = (
        PPOConfig()
        .environment(
            NativeBootstrapEnv,
            env_config={
                "evaluation_mode": True,
                "evaluation_seeds": [4000000011],
                "max_aircraft_count": 1,
                "scenario_start_time_offset_max_s": 0,
                "max_steps_per_episode": 1,
                "noop_verification_max_seconds": 1 if mode == "policy_cap" else 10000,
                "bootstrap_world": world.json(),
                "calibrate_prepared_arrival": mode == "airborne_verified",
            },
            disable_env_checking=True,
            observation_space=build_observation_space(observation_settings_from_config()),
            action_space=gym.spaces.Discrete(Config.NUM_ACTIONS),
        )
        .env_runners(env_runner_cls=ATCEnvRunner, num_env_runners=0, num_gpus_per_env_runner=0)
        .multi_agent(policies={"p"}, policy_mapping_fn=shared_mapping)
        .learners(learner_class=ATCPPOLearner, num_learners=0, num_gpus_per_learner=0)
        .training(gamma=0.9, lambda_=0.5)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=ATCRLM,
                model_config={
                    "resolved_model": resolved.model_payload(),
                    "expected_semantic_digest": resolved.semantic_digest,
                    "max_seq_len": resolved.features["architecture"]["max_seq_len"],
                },
            )
        )
    )
    runner = cfg.env_runner_cls(cfg)
    try:
        episodes = runner.sample(num_timesteps=1, random_actions=True)
        assert len(episodes) == 1
        env = runner.env.envs[0].unwrapped
        physical = mode.startswith("physical")
        assert env.raw_bootstrap_flags[0]["__all__"] == physical
        assert env.raw_bootstrap_flags[1]["__all__"] != physical
        if mode == "airborne_verified":
            assert env.last_outcome.kind == "verified_noop_success"
            assert not env.outcome_state.physically_complete
        elif mode == "policy_cap":
            assert env.last_end_reason == "timeout"
        elif mode == "physical_completion":
            assert env.outcome_state.physically_complete
        else:
            assert env.failed_target_callsigns
        for sa in episodes[0].agent_episodes.values():
            assert sa.is_terminated == physical
        learner = cfg.build_learner()
        module = learner.module["p"]
        original = module.compute_values
        observed_values = []

        def record_values(batch, embeddings=None):
            values = original(batch, embeddings=embeddings)
            observed_values.append(values.detach().cpu().numpy().copy())
            return values

        module.compute_values = record_values
        batch = learner._learner_connector(
            rl_module=learner.module, batch={}, episodes=episodes, shared_data={}
        )["p"]
        assert len(observed_values) == 1
        mask = batch[Columns.LOSS_MASK].numpy()
        values = observed_values[0]
        rewards = batch[Columns.REWARDS].numpy()
        targets = batch[Postprocessing.VALUE_TARGETS].numpy()
        expected = rewards[:, 0] + (0 if physical else 0.9 * values[:, 1])
        np.testing.assert_allclose(targets[mask], expected, rtol=1e-5, atol=1e-6)
        assert np.isfinite(targets).all() and np.all(targets[~mask] == 0)
        runtime_print(
            "Bootstrap",
            mode,
            raw_flags=env.raw_bootstrap_flags,
            values=values.tolist(),
            rewards=rewards.tolist(),
            mask=mask.tolist(),
            targets=targets.tolist(),
            outcome=env.last_outcome.kind,
        )
        with torch.no_grad():
            output = module.forward_train(batch)
        assert output[Columns.ACTION_DIST_INPUTS].shape[-1] == Config.NUM_ACTIONS
    finally:
        runner.stop()
