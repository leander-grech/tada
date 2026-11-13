"""A small real ATCRLM and observation batches for behavior tests."""

import gymnasium as gym
import torch

from atc_env.env_config import observation_settings_from_config
from config.config import Config
from config.settings import settings_from_config
from models.observation_space import build_observation_space
from network.rlm import unbound_model_config


def build_module(seed: int = 0):
    from network.rlm import ATCRLM

    torch.manual_seed(seed)
    module = ATCRLM(
        observation_space=build_observation_space(observation_settings_from_config()),
        action_space=gym.spaces.Discrete(int(Config.NUM_ACTIONS)),
        model_config=unbound_model_config(settings_from_config().architecture),
    )
    module.eval()
    return module


def obs_batch(
    batch: int, steps: int | None = None, *, seed: int = 1, valid_waypoints: int | None = None
) -> dict:
    """Random observations. `steps=None` gives [B, ...] tensors, else [B, T, ...].

    `valid_waypoints` limits the flight-plan mask; padded rows hold garbage so that any
    leak through the mask shows up as a difference.
    """
    generator = torch.Generator().manual_seed(seed)
    lead = (batch,) if steps is None else (batch, steps)

    def uniform(*shape):
        return torch.rand(*lead, *shape, generator=generator) * 2.0 - 1.0

    waypoints = int(Config.MAX_WAYPOINTS)
    valid = waypoints if valid_waypoints is None else valid_waypoints
    flight_plan_mask = torch.zeros(*lead, waypoints)
    flight_plan_mask[..., :valid] = 1.0
    neighbors = int(Config.MAX_NEIGHBORS)
    neighbor_mask = torch.zeros(*lead, neighbors)
    neighbor_mask[..., :2] = 1.0
    return {
        "action_mask": torch.ones(*lead, int(Config.NUM_ACTIONS)),
        "observations": {
            "aircraft_scalar": uniform(int(Config.AIRCRAFT_SCALAR_LEN)),
            "global_context": uniform(int(Config.GLOBAL_CONTEXT_DIM)).abs(),
            "flight_plans_rel": uniform(waypoints, int(Config.FLIGHT_PLAN_WAYPOINT_DIM)),
            "flight_plan_mask": flight_plan_mask,
            "neighbor_features": uniform(neighbors, int(Config.NEIGHBOR_FEATURE_DIM)),
            "neighbor_mask": neighbor_mask,
        },
    }


def initial_state(module, batch: int) -> dict:
    hidden = int(module.architecture["gru_hidden"])
    return {"gru_h": torch.zeros(batch, hidden)}


def step_slice(obs: dict, t: int) -> dict:
    """Time step `t` of a [B, T, ...] observation batch as a [B, ...] batch."""
    return {
        "action_mask": obs["action_mask"][:, t],
        "observations": {k: v[:, t] for k, v in obs["observations"].items()},
    }
