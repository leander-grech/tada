"""Regression tests for trombone action planning and masking state."""

from flight_simulator import RoutePoint, Vec3
from support.geometry import design_data, route_point

from actions.action_mask import compute_action_mask
from actions.actions import Action, _get_command_for_action
from config.settings import settings_from_config
from simulator.route_section_store import RouteSectionStore


def _mxp_trombone_route() -> list[RoutePoint]:
    names = [
        "START0",
        "START1",
        "MC761",
        "MC756",
        "ESITE",
        "MC746",
        "MC741",
        "MC435",
        "MC430",
        "MC425",
        "MC420",
        "INLER",
        "END0",
        "END1",
        "END2",
    ]
    return [route_point(name, float(index), 0.0) for index, name in enumerate(names)]


def _route_positions(route: list[RoutePoint]) -> list[tuple[float, float, float]]:
    return [(float(index), 0.0, 0.0) for index, _ in enumerate(route)]


def test_shorten_command_planning_does_not_mutate_store_before_apply() -> None:
    store = RouteSectionStore()
    trimmed_route = store.cut_out_section("AC1", _mxp_trombone_route())
    current_route = store.add_section_to_route("AC1", trimmed_route, 4)

    commands, post_action_state = _get_command_for_action(
        action_id=Action.SHORTEN_TROMBONE,
        call_sign="AC1",
        target_vel=250.0,
        target_heading=0.0,
        current_route=current_route,
        current_pos_xy=(0.0, 0.0),
        route_trombone_store=store,
        settings=settings_from_config().action,
    )

    assert store.num_levels_can_retract("AC1") == 4
    assert post_action_state["route_after"] == [
        "START0",
        "START1",
        "MC761",
        "MC756",
        "ESITE",
        "MC746",
        "MC430",
        "MC425",
        "MC420",
        "INLER",
        "END0",
        "END1",
        "END2",
    ]

    commands[0].apply_side_effects()

    assert store.num_levels_can_retract("AC1") == 3


def test_shorten_mask_requires_latest_level_to_be_retractable_from_live_route() -> None:
    store = RouteSectionStore()
    trimmed_route = store.cut_out_section("AC1", _mxp_trombone_route())
    full_route = store.add_section_to_route("AC1", trimmed_route, 4)
    live_route = full_route[6:]

    mask = compute_action_mask(
        ac_name="AC1",
        current_pos=Vec3(0.0, 0.0, 0.0),
        target_vel=250.0,
        current_vel=Vec3(1.0, 0.0, 0.0),
        wps_list=_route_positions(live_route),
        route=live_route,
        design_data=design_data(),
        target_heading_deg=0.0,
        route_trombone_store=store,
        ego_infringements=None,
        settings=settings_from_config().action,
    )

    assert not bool(mask[int(Action.SHORTEN_TROMBONE)])
    assert store.num_levels_can_retract("AC1") == 4
