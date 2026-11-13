"""Small observation-valued modules and episodes for actual PPO boundary tests."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from training.learner_checkpoint import ATCPPOLearner

OBS_SPACE = gym.spaces.Box(-10000, 10000, (1,), np.float32)
ACT_SPACE = gym.spaces.Discrete(2)


class ObservationValueModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.value_calls = 0
        self.observed_values = []

    def get_initial_state(self):
        return {"h": np.zeros(1, np.float32)} if self.model_config.get("stateful") else {}

    def compute_values(self, batch, embeddings=None):
        self.value_calls += 1
        values = batch[Columns.OBS][..., 0] + self.offset
        self.observed_values.append(values.detach().cpu().numpy().copy())
        return values

    def _forward(self, batch, **kwargs):
        return {Columns.ACTION_DIST_INPUTS: torch.zeros((*batch[Columns.OBS].shape[:-1], 2))}


def mapping(agent_id, *args, **kwargs):
    return "p" if agent_id == "a" else "q"


def shared_mapping(*args, **kwargs):
    return "p"


def learner_config(*, stateful=False, shared=True, lambda_=0.5):
    return (
        PPOConfig()
        .environment(observation_space=OBS_SPACE, action_space=ACT_SPACE)
        .multi_agent(
            policies={"p"} if shared else {"p", "q"},
            policy_mapping_fn=shared_mapping if shared else mapping,
        )
        .learners(learner_class=ATCPPOLearner, num_learners=0, num_gpus_per_learner=0)
        .training(gamma=0.9, lambda_=lambda_)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=ObservationValueModule,
                model_config={"stateful": stateful, "max_seq_len": 3},
            )
        )
    )


def episode(
    values=(3, 7), rewards=(2,), *, mode="truncated", agents=("a",), shared=True, lookback=False
):
    ep = MultiAgentEpisode(
        observation_space=gym.spaces.Dict({a: OBS_SPACE for a in agents}),
        action_space=gym.spaces.Dict({a: ACT_SPACE for a in agents}),
        agent_to_module_mapping_fn=shared_mapping if shared else mapping,
    )
    ep.add_env_reset(observations={a: np.array([values[0]], np.float32) for a in agents}, infos={})
    if lookback:
        ep.add_env_step(
            observations={a: np.array([103], np.float32) for a in agents},
            actions={a: 0 for a in agents},
            rewards={a: 100 for a in agents},
            infos={},
        )
        ep = ep.cut(len_lookback_buffer=1)
    for i, reward in enumerate(rewards):
        last = i == len(rewards) - 1
        ep.add_env_step(
            observations={a: np.array([values[i + 1]], np.float32) for a in agents},
            actions={a: 0 for a in agents},
            rewards={a: reward for a in agents},
            infos={},
            terminateds={"__all__": last and mode == "terminal"},
            truncateds={"__all__": last and mode == "truncated"},
        )
    return ep.to_numpy()


def run_pipeline(learner, episodes, *, shared_data=None):
    return learner._learner_connector(
        rl_module=learner.module,
        batch={},
        episodes=episodes,
        shared_data={} if shared_data is None else shared_data,
    )


@pytest.fixture
def ray_env_registry():
    """Own Ray's fixed test env registration without leaking/replacing prior entries."""
    registry = gym.envs.registration.registry
    name = "rllib-multi-agent-env-v0"
    previous = registry.pop(name, None)
    try:
        yield
    finally:
        registry.pop(name, None)
        if previous is not None:
            registry[name] = previous
