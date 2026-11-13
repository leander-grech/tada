"""Inspect actual observation bytes from the selected native simulator."""

import copy
import json
import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from flight_simulator import World, simulate_n_next_states
from ray.rllib.core.columns import Columns

from atc_env.env_config import observation_settings_from_config
from atc_env.environment import ATCEnv
from common.runtime_log import runtime_print
from config.config import Config
from config.settings import settings_from_config
from models.observation_space import build_observation_space
from models.observations import make_agent_observations_from_simulator
from network.rlm import ATCRLM, unbound_model_config


@pytest.fixture
def native_observer(native_base_json):
    env = ATCEnv(
        {
            "evaluation_mode": True,
            "evaluation_seeds": [4000000011],
            "max_aircraft_count": 1,
            "scenario_start_time_offset_max_s": 0,
        }
    )

    def observe(*, angle=0.0, eta=600.0, minimum=None, maximum=6000.0):
        data = json.loads(native_base_json)
        data["age_in_seconds"] = 45.0
        data["age_in_steps"] = 45
        data["arrival_targets"][0]["eta_s"] = eta
        ac = data["aircraft"][0]
        ac["under_control"] = True
        ac["state"]["position_m"] = [0, 0, 9000 * 0.3048]
        ac["state"]["velocity_m_s"] = [100 * math.sin(angle), 100 * math.cos(angle), 0]
        ac["route"][0].update(
            position_m=[1852, 3704],
            waypoint_name="FIXTURE",
            min_altitude=None if minimum is None else minimum * 0.3048,
            max_altitude=None if maximum is None else maximum * 0.3048,
        )
        neighbor = copy.deepcopy(ac)
        neighbor["call_sign"] = "NEIGHBOR"
        neighbor["state"]["position_m"] = [
            18520 * math.sin(angle),
            18520 * math.cos(angle),
            9000 * 0.3048,
        ]
        neighbor["state"]["velocity_m_s"] = [50 * math.sin(angle), 50 * math.cos(angle), 0]
        data["aircraft"].append(neighbor)
        data["arrival_targets"].append({"ac_callsign": "NEIGHBOR", "eta_s": eta})
        env.simulator.world = World.from_json(json.dumps(data))
        world = env.simulator.world
        # Forecast age must not become the live age used by the observation.
        forecast = simulate_n_next_states(10, 1000, world, [], None)
        assert forecast.age_in_seconds > world.age_in_seconds == 45
        obs = make_agent_observations_from_simulator(
            env.simulator,
            [ac["call_sign"]],
            {},
            {},
            env._route_trombone_store,
            settings=env.feature_settings,
            scales=env.settings.core.observation,
            action_settings=env.settings.core.action,
        )[ac["call_sign"]]
        assert build_observation_space(observation_settings_from_config()).contains(obs)
        return obs

    yield observe
    env.close()


@pytest.mark.parametrize("eta,remaining", [(600, 555), (30, -15), (45, 0)])
def test_actual_live_signed_time_observation(native_observer, eta, remaining):
    obs = native_observer(eta=eta)
    scalar = obs["observations"]["aircraft_scalar"]
    expected = 2 / math.pi * math.atan(1.5 * remaining / Config.TIME_TO_TARGET_MAX)
    assert scalar[4] == pytest.approx(expected)
    assert scalar[5] == 0 and scalar[8] == 0  # Unknown forecast timing stays unavailable.
    runtime_print("Geometry", "live time", eta=eta, live_age=45, encoded=float(scalar[4]))


@pytest.mark.parametrize("angle", [0, math.pi / 2, 0.731])
def test_native_neighbor_ego_channels_and_model_input(native_observer, angle):
    obs = native_observer(angle=angle)
    features = obs["observations"]["neighbor_features"][0]
    assert obs["observations"]["neighbor_mask"][0] == 1
    assert features[0] == pytest.approx(10 / Config.REL_POS_XY_MAX, abs=1e-6)
    assert features[1] == pytest.approx(0, abs=1e-6)
    assert features[3] == pytest.approx((-50 / 1852 * 3600) / Config.REL_VEL_RANGE_KTS, abs=1e-6)
    assert features[4] == pytest.approx(0, abs=1e-6)
    assert features[8:10] == pytest.approx([0, 1], abs=1e-6)
    baseline = native_observer()["observations"]["neighbor_features"]
    np.testing.assert_allclose(obs["observations"]["neighbor_features"], baseline, atol=1e-6)
    # Route XY remains world-relative, so it is deliberately independent of ego heading.
    assert obs["observations"]["flight_plans_rel"][0, :2] == pytest.approx(
        [1 / Config.REL_POS_XY_MAX, 2 / Config.REL_POS_XY_MAX], abs=1e-6
    )
    module = ATCRLM(
        observation_space=build_observation_space(observation_settings_from_config()),
        action_space=gym.spaces.Discrete(Config.NUM_ACTIONS),
        model_config=unbound_model_config(settings_from_config().architecture),
    )
    batch = {
        "observations": {
            key: torch.from_numpy(value).unsqueeze(0) for key, value in obs["observations"].items()
        },
        "action_mask": torch.from_numpy(obs["action_mask"]).unsqueeze(0),
    }
    with torch.no_grad():
        output = module.forward_train({Columns.OBS: batch})
    assert torch.isfinite(output[Columns.ACTION_DIST_INPUTS]).all()
    runtime_print("Geometry", "native ego", angle=angle, neighbor=features.tolist())


@pytest.mark.parametrize(
    "minimum,maximum,altitude",
    [(6000, None, 6000), (None, 0, 0), (6000, 10000, 8000), (None, None, 9000)],
)
def test_actual_route_constraint_altitude(native_observer, minimum, maximum, altitude):
    obs = native_observer(minimum=minimum, maximum=maximum)
    expected = np.clip((altitude - 9000) / Config.REL_POS_Z_RANGE_FT, -1, 1)
    assert obs["observations"]["flight_plans_rel"][0, 2] == pytest.approx(expected, abs=1e-6)


def test_native_inverted_altitude_rejected(native_observer):
    with pytest.raises(ValueError, match="altitude"):
        native_observer(minimum=9000, maximum=6000)
