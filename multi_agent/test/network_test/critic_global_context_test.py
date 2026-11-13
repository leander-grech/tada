"""Tests for critic-only global context handling in ATCRLM."""

import gymnasium as gym
import pytest
import torch
from ray.rllib.core.columns import Columns

from atc_env.env_config import observation_settings_from_config
from config.config import Config
from config.settings import settings_from_config
from models.feature_transforms import encode_deviation
from models.observation_space import build_observation_space
from network.rlm import ATCRLM, unbound_model_config


def _build_obs_batch(global_context: torch.Tensor) -> dict:
    return {
        "action_mask": torch.ones(2, int(Config.NUM_ACTIONS)),
        "observations": {
            "aircraft_scalar": torch.zeros(2, int(Config.AIRCRAFT_SCALAR_LEN)),
            "global_context": global_context,
            "flight_plans_rel": torch.zeros(
                2,
                int(Config.MAX_WAYPOINTS),
                int(Config.FLIGHT_PLAN_WAYPOINT_DIM),
            ),
            "flight_plan_mask": torch.zeros(2, int(Config.MAX_WAYPOINTS)),
            "neighbor_features": torch.zeros(
                2,
                int(Config.MAX_NEIGHBORS),
                int(Config.NEIGHBOR_FEATURE_DIM),
            ),
            "neighbor_mask": torch.zeros(2, int(Config.MAX_NEIGHBORS)),
        },
    }


def _encoded_context(mode: str) -> torch.Tensor:
    context = torch.zeros(2, int(Config.GLOBAL_CONTEXT_DIM))
    for column, seconds in ((5, 50), (6, 80), (8, 30)):
        context[:, column] = encode_deviation(
            seconds, mode=mode, linear_scale_s=200, knee_s=20, cap_s=180
        )
    context[:, 11] = 2 / 3
    return context


@pytest.mark.parametrize("mode", ["linear", "signed_log"])
def test_actor_path_ignores_critic_global_context(mode) -> None:
    module = ATCRLM(
        observation_space=build_observation_space(observation_settings_from_config()),
        action_space=gym.spaces.Discrete(int(Config.NUM_ACTIONS)),
        model_config=unbound_model_config(settings_from_config().architecture),
    )
    zero_batch = {Columns.OBS: _build_obs_batch(torch.zeros(2, int(Config.GLOBAL_CONTEXT_DIM)))}
    one_batch = {Columns.OBS: _build_obs_batch(_encoded_context(mode))}

    module.eval()
    zero_out = module._forward_train(zero_batch)
    one_out = module._forward_train(one_batch)

    assert torch.allclose(
        zero_out[Columns.ACTION_DIST_INPUTS],
        one_out[Columns.ACTION_DIST_INPUTS],
    )


@pytest.mark.parametrize("mode", ["linear", "signed_log"])
def test_critic_path_uses_global_context(mode) -> None:
    module = ATCRLM(
        observation_space=build_observation_space(observation_settings_from_config()),
        action_space=gym.spaces.Discrete(int(Config.NUM_ACTIONS)),
        model_config=unbound_model_config(settings_from_config().architecture),
    )
    zero_batch = {Columns.OBS: _build_obs_batch(torch.zeros(2, int(Config.GLOBAL_CONTEXT_DIM)))}
    one_batch = {Columns.OBS: _build_obs_batch(_encoded_context(mode))}

    forward_out = module._forward_train(zero_batch)
    embeddings = forward_out[Columns.EMBEDDINGS]
    zero_values = module.compute_values(zero_batch, embeddings=embeddings)
    one_values = module.compute_values(one_batch, embeddings=embeddings)

    assert not torch.allclose(zero_values, one_values)
