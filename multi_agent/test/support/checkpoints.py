"""Small real runner/module fixtures shared by checkpoint and evaluation tests."""

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from network.value_normalizer import RunningMeanStd
from support.bootstrap import ACT_SPACE, OBS_SPACE, ObservationValueModule, learner_config


class BoundaryProbeEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.possible_agents = self.agents = ["a"]
        self.observation_spaces = {"a": OBS_SPACE}
        self.action_spaces = {"a": ACT_SPACE}
        self.length = config["length"]
        self.resets = self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.steps = 0
        return {"a": np.array([3.0], np.float32)}, {}

    def step(self, action_dict):
        self.steps += 1
        done = self.steps >= self.length
        return (
            {"a": np.array([7.0], np.float32)},
            {"a": 1.0},
            {"a": False, "__all__": False},
            {"a": done, "__all__": done},
            {},
        )

    def capture_boundary_state(self):
        if self.steps != self.length:
            raise ValueError("scenario unfinished")
        return {"resets": self.resets}


class ProbePPOModule(ObservationValueModule):
    def setup(self):
        super().setup()
        self.value_normalizer = RunningMeanStd()

    def _forward(self, batch, **kwargs):
        result = super()._forward(batch, **kwargs)
        if Columns.STATE_IN in batch:
            result[Columns.STATE_OUT] = batch[Columns.STATE_IN]
        return result


def make_runner(length, *, stateful=False, worker_index=0):
    from training.env_runner import ATCEnvRunner

    cfg = (
        learner_config(stateful=stateful)
        .training(use_kl_loss=False)
        .environment(BoundaryProbeEnv, env_config={"length": length}, disable_env_checking=True)
        .env_runners(
            env_runner_cls=ATCEnvRunner,
            num_env_runners=0,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            sample_timeout_s=None,
        )
    )
    spec = cfg.rl_module_spec
    spec.module_class = ProbePPOModule
    cfg.rl_module(rl_module_spec=spec)
    return cfg, ATCEnvRunner(cfg, worker_index=worker_index)
