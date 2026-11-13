"""Each action-selection mode survives a real reset and several real steps."""

import numpy as np
import pytest

from actions.actions import Action
from atc_env.environment_helpers import target_agents_per_step
from config.config import Config
from config.settings import settings_from_config


@pytest.mark.parametrize("mode", ["single", "sector_based"])
def test_reset_and_steps_in_selection_mode(mode, monkeypatch, make_env):
    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", mode)
    env = make_env()

    obs, _info = env.reset()
    for _ in range(3):
        assert obs, "the env must offer at least one acting aircraft"
        assert len(obs) <= target_agents_per_step(settings_from_config().selection)
        for agent_id, agent_obs in obs.items():
            assert agent_id in env.possible_agents
            mask = np.asarray(agent_obs["action_mask"])
            assert mask.shape == (int(Config.NUM_ACTIONS),)
            assert mask[int(Action.DO_NOTHING)] == 1
        obs, rewards, terminated, truncated, _info = env.step(
            {agent_id: int(Action.DO_NOTHING) for agent_id in obs}
        )
        assert all(np.isfinite(r) for r in rewards.values())
        if terminated.get("__all__") or truncated.get("__all__"):
            break
