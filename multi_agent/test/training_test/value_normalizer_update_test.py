"""Actual recurrent PPO update and learner checkpoint preserve valid-row statistics."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tree
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import Postprocessing

from atc_env.env_config import observation_settings_from_config
from common.runtime_log import runtime_print
from config.config import Config
from models.observation_space import build_observation_space
from network.learner_connectors import ATCGeneralAdvantageEstimation
from network.rlm import ATCRLM
from network.value_normalizer import RunningMeanStd
from training.learner_checkpoint import ATCPPOLearner
from training.run_config import load_run_config


def test_recurrent_update_and_normalizer_checkpoint_round_trip(config_path):
    # Torch's selected CUDA build cannot enter cuda.device(cpu). Hide CUDA only
    # in this child before imports; keep the runtime profile and global GPUs intact.
    result = subprocess.run(
        [sys.executable, __file__, str(config_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    runtime_print(
        "Normalizer", "recurrent update evidence", output=result.stdout, stderr=result.stderr
    )


def _run_update(config_path):
    torch.manual_seed(17)
    data = json.loads(config_path.read_text())
    data["features"] = {"architecture": {"max_seq_len": 3, "gru_hidden": 24}}
    config_path.write_text(json.dumps(data))
    resolved = load_run_config(config_path)
    obs_space = build_observation_space(observation_settings_from_config())
    action_space = gym.spaces.Discrete(Config.NUM_ACTIONS)
    config = (
        PPOConfig()
        .environment(observation_space=obs_space, action_space=action_space)
        .multi_agent(policies={"p"}, policy_mapping_fn=lambda *args, **kwargs: "p")
        .learners(learner_class=ATCPPOLearner, num_learners=0, num_gpus_per_learner=0)
        .training(gamma=0.9, lambda_=0.5, lr=0.001, use_kl_loss=False)
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
    learner = config.build_learner()
    assert isinstance(learner._learner_connector.connectors[-1], ATCGeneralAdvantageEstimation)
    module = learner.module["p"]
    assert module.is_stateful()
    obs = tree.map_structure(np.zeros_like, obs_space.sample())
    obs["action_mask"][:] = 1
    episodes = []
    # Lengths 1 and 4 produce both artificial bootstrap rows and recurrent padding.
    for length in (1, 4):
        ep = MultiAgentEpisode(
            observation_space=gym.spaces.Dict({"a": obs_space}),
            action_space=gym.spaces.Dict({"a": action_space}),
            agent_to_module_mapping_fn=lambda *args, **kwargs: "p",
        )
        ep.add_env_reset(observations={"a": copy.deepcopy(obs)}, infos={})
        for step in range(length):
            ep.add_env_step(
                observations={"a": copy.deepcopy(obs)},
                actions={"a": 0},
                rewards={"a": float(step + 1)},
                infos={},
                truncateds={"__all__": step == length - 1},
                extra_model_outputs={
                    "a": {
                        Columns.STATE_OUT: module.get_initial_state(),
                        Columns.ACTION_DIST_INPUTS: np.zeros(Config.NUM_ACTIONS, np.float32),
                        Columns.ACTION_LOGP: np.float32(-np.log(Config.NUM_ACTIONS)),
                    }
                },
            )
        episodes.append(ep.to_numpy())

    expected = RunningMeanStd()
    observed = {}
    original_update_stats = learner._update_value_normalizers_from_batches

    def record_full_batch(batches):
        batch = batches[0].policy_batches["p"]
        mask = batch[Columns.LOSS_MASK]
        targets = batch[Postprocessing.VALUE_TARGETS]
        assert mask.dtype == torch.bool and mask.shape == targets.shape == (3, 3)
        assert mask.tolist() == [[True, False, False], [True, True, True], [True, False, False]]
        assert batch[Columns.SEQ_LENS].tolist() == [2, 3, 2]
        expected.update(targets[mask])
        observed.update(shape=list(mask.shape), valid=int(mask.sum()), total=mask.numel())
        original_update_stats(batches)

    learner._update_value_normalizers_from_batches = record_full_batch
    original_loss = learner.compute_loss_for_module
    losses = []

    def record_loss(**kwargs):
        assert module.value_normalizer.count.item() == 1
        loss = original_loss(**kwargs)
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
        return loss

    learner.compute_loss_for_module = record_loss
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    learner.update(episodes=episodes, num_epochs=2, minibatch_size=3)
    assert len(losses) > 1
    assert any(not torch.equal(before[name], value) for name, value in module.named_parameters())
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(module.value_normalizer.state_dict()[name], value)
    assert all(torch.isfinite(value).all() for value in module.state_dict().values())

    checkpoint = learner.save_to_path(str(config_path.parent / "learner-checkpoint"))
    restored = config.build_learner()
    restored.restore_from_path(checkpoint)
    optimizer_state = learner._get_optimizer_state()
    restored_optimizer_state = restored._get_optimizer_state()
    tree.assert_same_structure(optimizer_state, restored_optimizer_state)
    for actual, expected_value in zip(
        tree.flatten(restored_optimizer_state), tree.flatten(optimizer_state), strict=True
    ):
        # Ray reconstructs optimizer scalar hyperparameters as float32 tensors.
        if isinstance(expected_value, (float, np.floating)):
            np.testing.assert_allclose(actual, expected_value, rtol=1e-7, atol=1e-12)
        else:
            np.testing.assert_equal(actual, expected_value)
    for name, value in module.state_dict().items():
        torch.testing.assert_close(restored.module["p"].state_dict()[name], value, rtol=0, atol=0)
    probe = torch.tensor([-3.0, 2.0, 11.0])
    torch.testing.assert_close(
        restored.module["p"].value_normalizer.normalize(probe),
        module.value_normalizer.normalize(probe),
    )
    runtime_print(
        "Normalizer",
        "actual recurrent update and learner round-trip",
        mask=observed,
        losses=losses,
        statistics_updates=int(module.value_normalizer.count),
        normalizer_mean=float(module.value_normalizer.mean),
        state_round_trip=True,
    )


if __name__ == "__main__":
    _run_update(Path(sys.argv[1]))
