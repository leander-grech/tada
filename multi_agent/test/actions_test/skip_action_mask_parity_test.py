"""Regression tests for skip-action masking parity with execution constraints."""

import numpy as np
from flight_simulator import Vec3

from actions.action_mask import _mask_skip_waypoints_actions
from actions.actions import Action
from config.config import Config
from config.settings import settings_from_config

_ORIGIN = Vec3(0.0, 0.0, 0.0)


def _straight_wps(count: int) -> list[tuple[float, float, float]]:
    return [(float(i + 1), 0.0, 0.0) for i in range(count)]


def _wps_with_turn_at(count: int, turn_index: int) -> list[tuple[float, float, float]]:
    """Waypoints on x-axis except *turn_index* which is offset laterally."""
    wps = [(float(i + 1), 0.0, 0.0) for i in range(count)]
    x = wps[turn_index][0]
    wps[turn_index] = (x, 1.0, 0.0)
    return wps


def _compute_skip_mask(current_pos: Vec3, wps: list[tuple[float, float, float]]) -> np.ndarray:
    mask = np.zeros(Config.NUM_ACTIONS, dtype=bool)
    return _mask_skip_waypoints_actions(
        mask=mask, current_pos=current_pos, wps_list=wps, settings=settings_from_config().action
    )


def test_all_skips_blocked_on_straight_geometry() -> None:
    """Collinear waypoints produce no route deviation — all skips are no-ops."""
    n = Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM
    all_skip_actions = [
        (Action.SKIP_1_WAYPOINT, n + 1),
        (Action.SKIP_2_WAYPOINTS_NOT_NEXT, n + 3),
        (Action.SKIP_3_WAYPOINTS_NOT_NEXT, n + 4),
        (Action.SKIP_4_WAYPOINTS_NOT_NEXT, n + 5),
    ]
    for action, n_wps in all_skip_actions:
        mask = _compute_skip_mask(_ORIGIN, _straight_wps(n_wps))
        assert not bool(mask[int(action)]), f"{action.name} should be blocked on straight geometry"


def test_skip_1_allowed_when_waypoint_has_meaningful_deviation() -> None:
    """A clear turn at the skipped waypoint makes skip-1 worthwhile."""
    n = Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM
    mask = _compute_skip_mask(_ORIGIN, _wps_with_turn_at(n + 1, turn_index=0))
    assert bool(mask[int(Action.SKIP_1_WAYPOINT)])


def test_skip_multi_allowed_when_skipped_segment_has_turn() -> None:
    """Skip-2/3/4 are allowed when the skipped segment contains a meaningful turn."""
    n = Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM
    skip_actions_with_turn = [
        (Action.SKIP_2_WAYPOINTS_NOT_NEXT, n + 3, 1),
        (Action.SKIP_3_WAYPOINTS_NOT_NEXT, n + 4, 2),
        (Action.SKIP_4_WAYPOINTS_NOT_NEXT, n + 5, 3),
    ]
    for action, n_wps, turn_idx in skip_actions_with_turn:
        mask = _compute_skip_mask(_ORIGIN, _wps_with_turn_at(n_wps, turn_index=turn_idx))
        assert bool(mask[int(action)]), (
            f"{action.name} should be allowed with turn at index {turn_idx}"
        )


def test_skip_action_mask_blocks_when_downstream_segment_missing() -> None:
    mask = _compute_skip_mask(_ORIGIN, _straight_wps(5))
    assert not bool(mask[int(Action.SKIP_4_WAYPOINTS_NOT_NEXT)])
