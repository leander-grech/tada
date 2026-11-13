"""Physical route feasibility cannot depend on packed observation capacity."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from flight_simulator import World

from actions.action_mask import compute_action_mask
from actions.actions import Action, _get_command_for_action
from atc_env.env_config import observation_settings_from_config
from config.config import Config
from config.settings import settings_from_config
from models.observation_features import (
    _collect_waypoints_and_rel_features,
    _compute_flight_plans_rel,
)
from simulator.design_parser import AircraftDesignData
from simulator.route_section_store import RouteSectionStore


@pytest.fixture
def route_world(native_base_json, monkeypatch):
    # Explicit geometry: K=6 would check indices 2,3; true four-point tail starts at 4.
    monkeypatch.setattr(Config, "WAYPOINTS_TO_LEAVE_AT_MINIMUM", 4)
    monkeypatch.setattr(Config, "SKIP_MAX_INTERCEPT_ANGLE_DEG", 80)
    data = json.loads(native_base_json)
    aircraft = data["aircraft"][0]
    template = aircraft["route"][0]
    route = []
    names = ["ME632", "REMOVED", "P0", "P1", "P2", "P3", "TIXUM", "P5", "P6", "P7"]
    for index, name in enumerate(names):
        point = copy.deepcopy(template)
        point.update(waypoint_name=name, position_m=[1852 * (index + 1), 0])
        route.append(point)
    aircraft["route"] = route
    aircraft["state"]["position_m"] = [0, 0, 3000]
    # Names resolve via inline positions, including known section markers.
    data["waypoints"] = {}
    return data


@pytest.mark.parametrize("tail_allowed", [True, False])
def test_native_full_route_mask_matches_emitted_tail_at_small_and_large_k(
    route_world, tail_allowed
):
    data = route_world
    # Store a real excluded section, then advance past its entry marker.
    original = World.from_json(json.dumps(data))
    store = RouteSectionStore()
    store.cut_out_section("TEST001", original.aircraft[0].route)
    data["aircraft"][0]["route"] = data["aircraft"][0]["route"][2:]
    route_data = data["aircraft"][0]["route"]
    # Intermediate segment points opposite the true tail's feasibility.
    for first_index, allowed in [(2, not tail_allowed), (4, tail_allowed)]:
        route_data[first_index]["position_m"] = [18520, 0]
        route_data[first_index + 1]["position_m"] = [37040 if allowed else 0, 0]
    world = World.from_json(json.dumps(data))
    ac = world.aircraft[0]
    assert len(ac.route) == 8
    assert store.has_reached_section(ac.call_sign, ac.route)
    assert store.is_before_upwind_start(ac.call_sign, ac.route)
    masks = []
    for capacity in [3, 6, 10]:
        settings = replace(observation_settings_from_config(), max_waypoints=capacity)
        positions, relative = _collect_waypoints_and_rel_features(
            ac, SimpleNamespace(world=world), ac.state.position_nm_nm_ft, settings
        )
        assert len(positions) == 8
        assert len(relative) == min(capacity, 8)
        packed, packed_mask = _compute_flight_plans_rel(
            relative, settings, settings_from_config().observation
        )
        assert packed.shape == (capacity, 8)
        assert packed_mask.sum() == min(capacity, 8)
        mask = compute_action_mask(
            ac.call_sign,
            ac.state.position_nm_nm_ft,
            200,
            ac.state.velocity_knots,
            positions,
            ac.route,
            AircraftDesignData("A320", "fixture", 40000, 100, 400, []),
            None,
            store,
            settings=settings_from_config().action,
        )
        assert bool(mask[int(Action.VECTOR_TO_ILS)]) is tail_allowed
        masks.append(mask)
    assert all(np.array_equal(mask, masks[0]) for mask in masks)
    commands, post = _get_command_for_action(
        Action.VECTOR_TO_ILS,
        ac.call_sign,
        200,
        0,
        ac.route,
        (0, 0),
        store,
        world.waypoints,
        settings=settings_from_config().action,
    )
    assert post["route_after"] == ["TIXUM", "P5", "P6", "P7"]
    emitted = cast(Any, commands[0].command)
    assert [p.waypoint_name for p in emitted.route] == post["route_after"]
    assert emitted.route[0].position_nm.x == pytest.approx(10)
    assert emitted.route[1].position_nm.x == pytest.approx(20 if tail_allowed else 0)


def test_unresolved_required_route_position_is_diagnosed_even_beyond_observation_k(route_world):
    route_world["aircraft"][0]["route"][7]["position_m"] = None
    world = World.from_json(json.dumps(route_world))
    ac = world.aircraft[0]
    settings = replace(observation_settings_from_config(), max_waypoints=3)
    with pytest.raises(ValueError, match="unsupported route.*TEST001.*7.*P5"):
        _collect_waypoints_and_rel_features(
            ac, SimpleNamespace(world=world), ac.state.position_nm_nm_ft, settings
        )
