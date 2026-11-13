"""Canonical observation-space definition for the ATC environment.

Every consumer (environment, model, tests, tooling) should import from here
rather than re-creating the space, so changes propagate automatically.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from actions.action_table import NUM_ACTIONS
from config.observation_fields import AIRCRAFT_SCALAR_FIELDS, GLOBAL_CONTEXT_FIELDS
from models.observation_settings import ObservationSettings


def build_observation_space(settings: ObservationSettings) -> gym.spaces.Dict:
    """Build the full observation space including action mask."""
    observation_fields: dict[str, gym.Space[Any]] = {
        "aircraft_scalar": gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(AIRCRAFT_SCALAR_FIELDS),),
            dtype=np.float32,
        ),
        "global_context": gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(GLOBAL_CONTEXT_FIELDS),),
            dtype=np.float32,
        ),
        "flight_plans_rel": gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(settings.max_waypoints, settings.waypoint_dim),
            dtype=np.float32,
        ),
        "flight_plan_mask": gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(settings.max_waypoints,),
            dtype=np.float32,
        ),
        "neighbor_features": gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(settings.max_neighbors), int(settings.neighbor_dim)),
            dtype=np.float32,
        ),
        "neighbor_mask": gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(int(settings.max_neighbors),),
            dtype=np.float32,
        ),
    }
    observations = gym.spaces.Dict(observation_fields)

    root_fields: dict[str, gym.Space[Any]] = {
        "observations": observations,
        "action_mask": gym.spaces.Box(
            low=0,
            high=1,
            shape=(NUM_ACTIONS,),
            dtype=np.uint8,
        ),
    }
    return gym.spaces.Dict(root_fields)
