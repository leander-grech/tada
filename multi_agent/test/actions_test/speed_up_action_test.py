"""Behavior tests for medium speed-up action support and custom mask safety."""

from types import SimpleNamespace

import pytest
from flight_simulator import RoutePoint, Vec3
from support.geometry import design_data, route_point

from actions.action_mask import compute_action_mask, detect_airport_from_route
from actions.actions import Action, _get_command_for_action
from config.config import Config
from config.settings import settings_from_config
from simulator.route_section_store import RouteSectionStore


def _route_and_wps() -> tuple[list[RoutePoint], list[tuple[float, float, float]]]:
    route = [route_point(f"WP{i}", float(i * 20), 0.0) for i in range(8)]
    wps = [(float(i * 20), 0.0, 0.0) for i in range(8)]
    return route, wps


def _route_store_stub() -> SimpleNamespace:
    # Speed masks depend on one addable and one retractable section level.
    # A real empty RouteSectionStore reports zero for both.
    return SimpleNamespace(
        num_levels_can_add_to_route=lambda *_args, **_kwargs: 1,
        num_levels_can_retract=lambda *_args, **_kwargs: 1,
        preview_retract_section_from_route=lambda _ac_name, route: route,
        has_excluded_section=lambda *_args, **_kwargs: False,
        has_reached_section=lambda *_args, **_kwargs: False,
        is_before_upwind_start=lambda *_args, **_kwargs: False,
    )


def _base_mask_kwargs() -> dict:
    route, wps = _route_and_wps()
    return {
        "ac_name": "TEST",
        "current_pos": Vec3(0.0, 0.0, 0.0),
        "target_vel": 250.0,
        "current_vel": Vec3(1.0, 0.0, 0.0),
        "wps_list": wps,
        "route": route,
        "design_data": design_data(),
        "target_heading_deg": 0.0,
        "route_trombone_store": _route_store_stub(),
    }


def test_speed_up_medium_mask_tracks_remaining_speed_headroom() -> None:
    kwargs = _base_mask_kwargs()
    effective_max = kwargs["design_data"].max_speed - Config.SPEED_LIMIT_BUFFER_KT
    kwargs_with_headroom = {
        **kwargs,
        "target_vel": effective_max - Config.ACTION_SPEED_DELTA_MEDIUM,
    }
    kwargs_at_limit = {**kwargs, "target_vel": effective_max}

    mask_with_headroom = compute_action_mask(
        **kwargs_with_headroom,
        ego_infringements=None,
        settings=settings_from_config().action,
    )
    mask_at_limit = compute_action_mask(
        **kwargs_at_limit,
        ego_infringements=None,
        settings=settings_from_config().action,
    )

    assert bool(mask_with_headroom[int(Action.SPEED_UP_MEDIUM)])
    assert not bool(mask_at_limit[int(Action.SPEED_UP_MEDIUM)])


def test_compute_action_mask_rejects_wrong_length_custom_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "CUSTOM_ACTION_MASK_BY_AIRPORT", {"TEST": [1]})

    with pytest.raises(ValueError, match="CUSTOM_ACTION_MASK"):
        compute_action_mask(
            **_base_mask_kwargs(),
            ego_infringements=None,
            airport="TEST",
            settings=settings_from_config().action,
        )


def test_compute_action_mask_rejects_custom_mask_without_do_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_mask = [1] * Config.NUM_ACTIONS
    bad_mask[int(Action.DO_NOTHING)] = 0
    monkeypatch.setattr(Config, "CUSTOM_ACTION_MASK_BY_AIRPORT", {"TEST": bad_mask})

    with pytest.raises(ValueError, match="DO_NOTHING"):
        compute_action_mask(
            **_base_mask_kwargs(),
            ego_infringements=None,
            airport="TEST",
            settings=settings_from_config().action,
        )


def test_detect_airport_from_route_matches_waypoint_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "CUSTOM_ACTION_MASK_BY_AIRPORT", {"MXP": None, "BGY": None})
    route_mxp = [route_point("WP0", 0, 0), route_point("MXP_35R", 10, 0)]
    route_bgy = [route_point("WP0", 0, 0), route_point("BGY_28", 10, 0)]
    route_unknown = [route_point("WP0", 0, 0), route_point("WP1", 10, 0)]

    assert (
        detect_airport_from_route(
            route_mxp, airports=settings_from_config().action.custom_action_mask_by_airport
        )
        == "MXP"
    )
    assert (
        detect_airport_from_route(
            route_bgy, airports=settings_from_config().action.custom_action_mask_by_airport
        )
        == "BGY"
    )
    assert (
        detect_airport_from_route(
            route_unknown, airports=settings_from_config().action.custom_action_mask_by_airport
        )
        is None
    )


def test_custom_mask_applied_via_detected_airport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """detect_airport_from_route result feeds into compute_action_mask correctly."""
    mask_list = [1] * Config.NUM_ACTIONS
    mask_list[int(Action.SLOW_DOWN_SMALL)] = 0
    monkeypatch.setattr(Config, "CUSTOM_ACTION_MASK_BY_AIRPORT", {"MXP": mask_list})

    kwargs = _base_mask_kwargs()
    kwargs["route"].append(route_point("MXP_35R", 200.0, 0.0))
    kwargs["wps_list"].append((200.0, 0.0, 0.0))

    airport = detect_airport_from_route(
        kwargs["route"], airports=settings_from_config().action.custom_action_mask_by_airport
    )
    mask = compute_action_mask(
        **kwargs, ego_infringements=None, airport=airport, settings=settings_from_config().action
    )
    assert not bool(mask[int(Action.SLOW_DOWN_SMALL)])


def test_speed_up_masked_when_next_waypoint_has_speed_constraint() -> None:
    wp_speed_limit = 220.0
    kwargs = _base_mask_kwargs()

    # Replace first route point with one that has a speed constraint
    kwargs["route"][0] = route_point("WP0", 0.0, 0.0, target_speed_knots=wp_speed_limit)

    # At limit: 220 + 20 > 220 → masked
    kwargs["target_vel"] = wp_speed_limit
    mask_at_limit = compute_action_mask(
        **kwargs, ego_infringements=None, settings=settings_from_config().action
    )
    assert not bool(mask_at_limit[int(Action.SPEED_UP_MEDIUM)])

    # Below limit with headroom: 200 + 20 = 220 → allowed
    kwargs["target_vel"] = wp_speed_limit - Config.ACTION_SPEED_DELTA_MEDIUM
    mask_with_headroom = compute_action_mask(
        **kwargs, ego_infringements=None, settings=settings_from_config().action
    )
    assert bool(mask_with_headroom[int(Action.SPEED_UP_MEDIUM)])


def test_speed_up_ignores_constraint_on_second_waypoint() -> None:
    kwargs = _base_mask_kwargs()

    # Only second waypoint has a speed constraint; first has none
    kwargs["route"][1] = route_point("WP1", 20.0, 0.0, target_speed_knots=180.0)
    kwargs["target_vel"] = 250.0

    mask = compute_action_mask(
        **kwargs, ego_infringements=None, settings=settings_from_config().action
    )
    assert bool(mask[int(Action.SPEED_UP_MEDIUM)])


def test_speed_up_medium_matches_medium_slow_down_delta() -> None:
    target_speed = 250.0

    _, slow_state = _get_command_for_action(
        action_id=Action.SLOW_DOWN_MEDIUM,
        call_sign="TEST",
        target_vel=target_speed,
        target_heading=0.0,
        current_route=[],
        current_pos_xy=(0.0, 0.0),
        route_trombone_store=RouteSectionStore(),
        settings=settings_from_config().action,
    )
    _, speed_up_state = _get_command_for_action(
        action_id=Action.SPEED_UP_MEDIUM,
        call_sign="TEST",
        target_vel=target_speed,
        target_heading=0.0,
        current_route=[],
        current_pos_xy=(0.0, 0.0),
        route_trombone_store=RouteSectionStore(),
        settings=settings_from_config().action,
    )

    assert (target_speed - slow_state["speed_after"]) == pytest.approx(
        speed_up_state["speed_after"] - target_speed
    )
