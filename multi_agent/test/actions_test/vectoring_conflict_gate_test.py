"""Tests that vectoring actions are gated on conflict (infringement) presence."""

from types import SimpleNamespace

import numpy as np
from flight_simulator import RoutePoint, Vec3
from support.geometry import design_data, route_point

from actions.action_mask import compute_action_mask
from actions.actions import Action
from config.settings import settings_from_config
from simulator.route_section_store import RouteSectionStore


def _straight_route_and_wps(
    n: int = 8,
) -> tuple[list[RoutePoint], list[tuple[float, float, float]]]:
    """Create a long straight route that makes vectoring geometrically feasible."""
    route = [route_point(f"WP{i}", float(i * 20), 0.0) for i in range(n)]
    wps = [(float(i * 20), 0.0, 0.0) for i in range(n)]
    return route, wps


def _base_mask_kwargs() -> dict:
    """Shared keyword arguments for compute_action_mask calls."""
    route, wps = _straight_route_and_wps()
    return {
        "ac_name": "TEST",
        "current_pos": Vec3(0.0, 0.0, 0.0),
        "target_vel": 250.0,
        "current_vel": Vec3(1.0, 0.0, 0.0),
        "wps_list": wps,
        "route": route,
        "design_data": design_data(),
        "target_heading_deg": 0.0,
        "route_trombone_store": RouteSectionStore(),
    }


def _vectoring_is_enabled(mask: np.ndarray) -> bool:
    """True if at least one vectoring action is enabled."""
    left = bool(mask[int(Action.TURN_LEFT_AND_RETURN_TO_ROUTE)])
    right = bool(mask[int(Action.TURN_RIGHT_AND_RETURN_TO_ROUTE)])
    return left or right


def test_vectoring_disabled_without_conflicts() -> None:
    """Empty infringement list -> vectoring must be disabled."""
    mask = compute_action_mask(
        **_base_mask_kwargs(), ego_infringements=[], settings=settings_from_config().action
    )
    assert not _vectoring_is_enabled(mask)


def test_vectoring_enabled_with_conflicts() -> None:
    """Non-empty infringement list -> vectoring should be available."""
    infr = SimpleNamespace(value=0.5, timestamp=0.0, counterpart="OTHER")
    mask = compute_action_mask(
        **_base_mask_kwargs(), ego_infringements=[infr], settings=settings_from_config().action
    )
    assert _vectoring_is_enabled(mask)


def test_vectoring_disabled_with_none_infringements() -> None:
    """None infringements (default) -> vectoring must be disabled."""
    mask = compute_action_mask(
        **_base_mask_kwargs(), ego_infringements=None, settings=settings_from_config().action
    )
    assert not _vectoring_is_enabled(mask)


def test_non_vectoring_actions_unaffected_by_conflict_gate() -> None:
    """Speed, skip, and DO_NOTHING actions should not depend on infringements."""
    kwargs = _base_mask_kwargs()
    mask_no_conflict = compute_action_mask(
        **kwargs, ego_infringements=[], settings=settings_from_config().action
    )
    infr = SimpleNamespace(value=0.5, timestamp=0.0, counterpart="OTHER")
    mask_with_conflict = compute_action_mask(
        **kwargs, ego_infringements=[infr], settings=settings_from_config().action
    )

    non_vectoring = [
        Action.DO_NOTHING,
        Action.SLOW_DOWN_SMALL,
        Action.SLOW_DOWN_MEDIUM,
        Action.SLOW_DOWN_LARGE,
        Action.SKIP_1_WAYPOINT,
    ]
    for action in non_vectoring:
        assert bool(mask_no_conflict[int(action)]) == bool(mask_with_conflict[int(action)]), (
            f"{action.name} should not be affected by conflict gating"
        )
