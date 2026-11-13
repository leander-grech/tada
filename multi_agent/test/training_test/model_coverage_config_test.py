"""Model consumes bound role payload without mutating shared architecture globals."""

import json

import gymnasium as gym
import pytest

from atc_env.env_config import observation_settings_from_config
from config.config import Config
from models.observation_settings import ObservationSettings
from models.observation_space import build_observation_space
from network.rlm import ATCRLM
from training.run_config import load_run_config


def test_model_uses_instance_architecture_and_checks_role_digest(config_path):
    data = json.loads(config_path.read_text())
    data["features"] = {"architecture": {"gru_hidden": 24, "aircraft_scalar_hidden": 32}}
    config_path.write_text(json.dumps(data))
    resolved = load_run_config(config_path)
    before = Config.GRU_HIDDEN_SIZE
    module = ATCRLM(
        observation_space=build_observation_space(
            ObservationSettings.from_features(resolved.features)
        ),
        action_space=gym.spaces.Discrete(Config.NUM_ACTIONS),
        model_config={
            "resolved_model": resolved.model_payload(),
            "expected_semantic_digest": resolved.semantic_digest,
        },
    )
    assert module.get_initial_state()["gru_h"].shape == (24,)
    assert module.aircraft_scalar_encoder[0].out_features == 32
    assert module.consumed_semantic_digest == resolved.semantic_digest
    assert Config.GRU_HIDDEN_SIZE == before
    with pytest.raises(ValueError, match="digest"):
        ATCRLM(
            observation_space=build_observation_space(observation_settings_from_config()),
            action_space=gym.spaces.Discrete(Config.NUM_ACTIONS),
            model_config={
                "resolved_model": resolved.model_payload(),
                "expected_semantic_digest": "0" * 64,
            },
        )
