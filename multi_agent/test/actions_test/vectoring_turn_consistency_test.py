"""Tests for vectoring turn consistency between execution and action masking."""

from typing import Any, cast

import numpy as np
import pytest
from flight_simulator import RoutePoint, Vec3
from support.geometry import route_point

from actions.action_mask import _mask_vectoring_actions
from actions.actions import Action, _get_command_for_action
from config.config import Config
from config.settings import settings_from_config
from simulator.commands import TimedCommand
from simulator.route_section_store import RouteSectionStore


def _route_names(route: list[RoutePoint]) -> list[str]:
    return [rp.waypoint_name for rp in route]


def _route_positions(route: list[RoutePoint]) -> list[tuple[float, float]]:
    return [
        (
            float(cast(Any, rp).position_nm.x),
            float(cast(Any, rp).position_nm.y),
        )
        for rp in route
    ]


def _command_route(commands: list[TimedCommand]) -> list[RoutePoint]:
    assert commands
    command = commands[0].command
    assert command is not None
    return cast(list[RoutePoint], cast(Any, command).route)


def test_chained_turn_stacks_after_existing_virtual_turn() -> None:
    current_route = [
        route_point("VEC_TURN_10NM", 5.0, 0.0),
        route_point("WP1", 10.0, 0.0),
        route_point("WP2", 20.0, 0.0),
        route_point("WP3", 30.0, 0.0),
        route_point("WP4", 40.0, 0.0),
    ]

    commands, _ = _get_command_for_action(
        action_id=Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
        call_sign="TEST",
        target_vel=250.0,
        target_heading=0.0,
        current_route=current_route,
        current_pos_xy=(0.0, 0.0),
        route_trombone_store=RouteSectionStore(),
        settings=settings_from_config().action,
    )
    new_route = _command_route(commands)
    names = _route_names(new_route)
    positions = _route_positions(new_route)

    assert names[0].startswith("VEC_TURN_")
    assert names[1].startswith("VEC_TURN_")
    assert names[2] == "WP1"
    assert positions[0] == (5.0, 0.0)
    assert positions[1] != (5.0, 0.0)


def test_chained_turn_with_existing_virtual_prefix_appends_at_tail() -> None:
    current_route = [
        route_point("VEC_TURN_10NM", 5.0, 0.0),
        route_point("VEC_TURN_10NM", 8.830222129821777, 3.213937997817993),
        route_point("WP1", 10.0, 0.0),
        route_point("WP2", 20.0, 0.0),
        route_point("WP3", 30.0, 0.0),
        route_point("WP4", 40.0, 0.0),
    ]

    commands, _ = _get_command_for_action(
        action_id=Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
        call_sign="TEST",
        target_vel=250.0,
        target_heading=0.0,
        current_route=current_route,
        current_pos_xy=(0.0, 0.0),
        route_trombone_store=RouteSectionStore(),
        settings=settings_from_config().action,
    )
    new_route = _command_route(commands)
    names = _route_names(new_route)
    positions = _route_positions(new_route)

    assert names[0].startswith("VEC_TURN_")
    assert names[1].startswith("VEC_TURN_")
    assert names[2].startswith("VEC_TURN_")
    assert names[3] == "WP1"
    assert positions[0] == (5.0, 0.0)
    assert positions[1] == pytest.approx((8.830222129821777, 3.213937997817993), abs=1e-6)
    assert positions[2] != positions[1]


def test_action_mask_turn_direction_matches_execution_turn_sign() -> None:
    # Deterministic geometry where +adjustment and -adjustment differ.
    wps_list = [
        (-15.132957406745596, 5.420683861316094, 0.0),
        (-33.046424312449276, 10.407484588617198, 0.0),
        (-53.47197848884524, 21.95204932124676, 0.0),
        (-68.35631837080684, 19.78814758766903, 0.0),
        (-82.0, 17.0, 0.0),
    ]
    route = [route_point(f"WP{i}", float(wp[0]), float(wp[1])) for i, wp in enumerate(wps_list)]
    current_pos = Vec3(0.0, 0.0, 0.0)
    current_vel = Vec3(-0.9589588801605488, -0.2835494336491727, 0.0)

    mask = np.zeros(Config.NUM_ACTIONS, dtype=bool)
    mask = _mask_vectoring_actions(
        mask=mask,
        current_pos=current_pos,
        current_vel=current_vel,
        target_heading_deg=None,
        wps_list=wps_list,
        route=route,
        settings=settings_from_config().action,
    )

    assert not bool(mask[int(Action.TURN_LEFT_AND_RETURN_TO_ROUTE)])
    assert bool(mask[int(Action.TURN_RIGHT_AND_RETURN_TO_ROUTE)])


def test_action_mask_uses_downstream_waypoints_when_virtual_turn_is_pending() -> None:
    # Leading virtual turn should be treated as the anchor, not as next return waypoint.
    # Waypoints spaced wide enough that a 10 nm vector from the anchor can return.
    route = [
        route_point("VEC_TURN_10NM", 10.0, 0.0),
        route_point("WP1", 40.0, 0.0),
        route_point("WP2", 60.0, 0.0),
        route_point("WP3", 80.0, 0.0),
        route_point("WP4", 100.0, 0.0),
        route_point("WP5", 120.0, 0.0),
    ]
    wps_list = [
        (10.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
        (60.0, 0.0, 0.0),
        (80.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (120.0, 0.0, 0.0),
    ]

    mask = np.zeros(Config.NUM_ACTIONS, dtype=bool)
    mask = _mask_vectoring_actions(
        mask=mask,
        current_pos=Vec3(0.0, 0.0, 0.0),
        current_vel=Vec3(1.0, 0.0, 0.0),
        target_heading_deg=0.0,
        wps_list=wps_list,
        route=route,
        settings=settings_from_config().action,
    )

    assert bool(mask[int(Action.TURN_LEFT_AND_RETURN_TO_ROUTE)])
    assert bool(mask[int(Action.TURN_RIGHT_AND_RETURN_TO_ROUTE)])
