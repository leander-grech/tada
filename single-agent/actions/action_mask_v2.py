"""Functions for calculating valid action masks for aircraft."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from config.config import Config
from flight_simulator import Vec3, RoutePoint
from simulator.design_parser import AircraftDesignData

from .actions_v2 import Action
from simulator.route_section_store import RouteSectionStore

_EPS = 1e-9




def compute_action_mask(
    ac_name: str,
    current_pos: Vec3,
    target_vel: float,
    current_vel: Vec3,
    wps_list: List[Tuple[float, float, float]],
    route_name: str,
    route: list[RoutePoint],
    design_data: AircraftDesignData,
    *,
    wp_vel_target: Optional[float] = None,
    route_waypoint_names: Optional[Sequence[str]] = None,
    target_heading_deg: Optional[float] = None,
    predicted_infringement: Optional[float] = None,
    route_trombone_store: RouteSectionStore,
) -> np.ndarray:
    """Return boolean mask for valid actions (False / zero = invalid action)"""

    mask = np.zeros(Config.NUM_ACTIONS, dtype=bool)
    mask[int(Action.DO_NOTHING)] = True

    mask = _mask_speed_actions(mask, target_vel, design_data, wp_vel_target)

    if len(wps_list) > Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM:
        mask = _mask_vectoring_actions(mask, current_pos, current_vel, target_heading_deg, wps_list)
        mask = _mask_skip_waypoints_actions(mask, current_pos, wps_list)
        mask = _mask_lengthen_trombone_actions(
            ac_name=ac_name,
            mask=mask,
            route=route,
            route_trombone_store=route_trombone_store,
        )
        mask = _mask_shorten_trombone_actions(
            ac_name=ac_name,
            mask=mask,
            route=route,
            route_trombone_store=route_trombone_store,
        )
        mask = _mask_vector_to_ils_action(
            ac_name=ac_name,
            mask=mask,
            current_pos=current_pos,
            wps_list=wps_list,
            route=route,
            route_trombone_store=route_trombone_store,
        )

    mask = _mask_risk_aware_no_op_action(mask=mask, predicted_infringement=predicted_infringement)

    if Config.CUSTOM_ACTION_MASK is not None:
        for i, val in enumerate(Config.CUSTOM_ACTION_MASK):
            if i < len(mask) and val > 0:
                mask[i] = True

    assert mask.any(), "action_mask ended up empty: this must never happen"
    return mask


def _mask_speed_actions(
    mask: np.ndarray,
    target_vel: float,
    design_data: AircraftDesignData,
    wp_vel_target: Optional[float],
) -> np.ndarray:
    """Mask speed actions.

    The set provides three slowing levels and two speed-up levels.
    """
    config = Config
    min_vel = float(design_data.min_speed)
    max_vel = float(design_data.max_speed)

    slow_levels = (
        (Action.SLOW_DOWN_SMALL, config.ACTION_SPEED_ADJUSTMENT_SMALL),
        (Action.SLOW_DOWN_MEDIUM, config.ACTION_SPEED_ADJUSTMENT_MEDIUM),
        (Action.SLOW_DOWN_LARGE, config.ACTION_SPEED_ADJUSTMENT_LARGE),
    )
    speed_levels = (
        (Action.SPEED_UP_SMALL, config.ACTION_SPEED_ADJUSTMENT_SMALL),
        (Action.SPEED_UP_MEDIUM, config.ACTION_SPEED_ADJUSTMENT_MEDIUM),
    )

    for action, delta in slow_levels:
        mask[int(action)] = target_vel >= (min_vel + delta)

    for action, delta in speed_levels:
        allowed = target_vel <= (max_vel - delta)
        # Never accelerate past the speed the next waypoint asks for.
        if allowed and wp_vel_target is not None:
            allowed = (target_vel + delta) <= wp_vel_target
        mask[int(action)] = allowed

    return mask


def _mask_skip_waypoints_actions(
    mask: np.ndarray,
    current_pos: Vec3,
    wps_list: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Mask skip waypoints actions"""

    skip_actions = [
        (Action.SKIP_1_WAYPOINT, 1, 0),
        (Action.SKIP_2_WAYPOINTS_NOT_NEXT, 2, 1),
        (Action.SKIP_3_WAYPOINTS_NOT_NEXT, 3, 1),
        (Action.SKIP_4_WAYPOINTS_NOT_NEXT, 4, 1),
    ]

    flight_plan_length = len(wps_list)

    for action, skip_amount, offset in skip_actions:
        total_skip = offset + skip_amount
        if flight_plan_length - total_skip - 1 < Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM:
            continue

        if _is_intercept_angle_ok(
            start_pos=(current_pos.x, current_pos.y),
            next_wp=wps_list[total_skip][:2],
            downstream_wp=wps_list[total_skip + 1][:2],
        ):
            mask[action] = True

    return mask


def _mask_lengthen_trombone_actions(
        ac_name: str,
    mask: np.ndarray,
    route: list[RoutePoint],
    route_trombone_store: RouteSectionStore,
) -> np.ndarray:

    # One action, one level. Cumulative range comes from stacking it, not from separate
    # actions per level, so it is legal whenever the store has anything left to give.
    if route_trombone_store.num_levels_can_add_to_route(ac_name, route) > 0:
        mask[int(Action.LENGTHEN_TROMBONE)] = True

    return mask


def _mask_shorten_trombone_actions(
    ac_name: str,
    mask: np.ndarray,
    route: list[RoutePoint],
    route_trombone_store: RouteSectionStore,
) -> np.ndarray:
    """Legal whenever an inserted trombone level is still ahead of the aircraft.

    Replaces a v1 helper of the same name that was dead code: it referenced
    Action.SHORTEN_TROMBONE_1/_4, which no enum ever defined, and its body was waypoint-skip
    logic that had been copy-pasted and renamed. It was never called, and would have raised
    AttributeError if it had been.
    """
    if route_trombone_store.num_levels_can_remove_from_route(ac_name, route) > 0:
        mask[int(Action.SHORTEN_TROMBONE)] = True

    return mask


def _mask_vectoring_actions(
    mask: np.ndarray,
    current_pos: Vec3,
    current_vel: Vec3,
    target_heading_deg: Optional[float],
    wps_list: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Mask vectoring actions"""

    current_x = current_pos.x
    current_y = current_pos.y

    heading_rad = (
        math.radians(target_heading_deg)
        if target_heading_deg is not None
        else math.atan2(current_vel.y, current_vel.x)
    )

    # Match the heading convention used by the turn commands.
    left = 1 if Config.TURN_MASK_FIX else -1
    turn_actions: Dict[int, Dict[str, int]] = {
        Action.TURN_LEFT_AND_RETURN_TO_ROUTE: {"dist": 10, "dirn": left, "skips": 0},
        Action.TURN_RIGHT_AND_RETURN_TO_ROUTE: {"dist": 10, "dirn": -left, "skips": 0},
    }

    max_return_angle_rad = Config.MAX_TURN_ANGLE_AFTER_HEADING_ADJUSTMENT * math.pi / 180.0
    heading_adjustment_rad = Config.ACTION_HEADING_ADJUSTMENT * math.pi / 180.0

    wps_len = len(wps_list)

    for action_idx, params in turn_actions.items():
        skip = params["skips"]
        if (wps_len - skip) < Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM:
            continue

        if evaluate_turn_feasibility(
            current_pos=(current_x, current_y),
            heading_rad=heading_rad,
            wps_list=wps_list,
            turn_distance=float(params["dist"]),
            turn_direction_multiplier=int(params["dirn"]),
            heading_adjustment_rad=heading_adjustment_rad,
            max_return_angle_rad=max_return_angle_rad,
            skip_count=skip,
        ):
            mask[action_idx] = True

    return mask


def evaluate_turn_feasibility(
    current_pos: Tuple[float, float],
    heading_rad: float,
    wps_list: List[Tuple[float, float, float]],
    turn_distance: float,
    turn_direction_multiplier: int,
    heading_adjustment_rad: float,
    max_return_angle_rad: float,
    skip_count: int,
) -> bool:
    """Evaluate turn feasibility by checking intercept angle"""

    next_idx = skip_count
    down_idx = 1 + skip_count
    if down_idx >= len(wps_list):
        return False

    current_x, current_y = current_pos
    next_wp_x, next_wp_y = float(wps_list[next_idx][0]), float(wps_list[next_idx][1])
    downstream_wp_x, downstream_wp_y = float(wps_list[down_idx][0]), float(wps_list[down_idx][1])

    new_heading_rad = heading_rad + turn_direction_multiplier * heading_adjustment_rad

    new_x = current_x + turn_distance * math.cos(new_heading_rad)
    new_y = current_y + turn_distance * math.sin(new_heading_rad)

    wp_heading_rad = math.atan2(next_wp_y - new_y, next_wp_x - new_x)

    angle_diff = abs(wp_heading_rad - new_heading_rad)
    if angle_diff > math.pi:
        angle_diff = 2.0 * math.pi - angle_diff
    if angle_diff > max_return_angle_rad:
        return False

    return _is_intercept_angle_ok(
        start_pos=(new_x, new_y),
        next_wp=(next_wp_x, next_wp_y),
        downstream_wp=(downstream_wp_x, downstream_wp_y),
    )


def _mask_vector_to_ils_action(
    ac_name: str,
    mask: np.ndarray,
    current_pos: Vec3,
    wps_list: List[Tuple[float, float, float]],
    route: list[RoutePoint],
    route_trombone_store: RouteSectionStore,
) -> np.ndarray:
    """Mask vector to ILS action"""

    # Do not allow vector to ILS if not yet on trombone,
    # or if already on upwind side (already in ILS direction)
    if not route_trombone_store.has_reached_section(
            ac_name, route
    ) or not route_trombone_store.is_before_upwind_start(ac_name, route):
        return mask

    current_x = float(current_pos.x)
    current_y = float(current_pos.y)

    tail_len = Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM
    base_idx = len(wps_list) - tail_len
    if base_idx < 0 or (base_idx + 1) >= len(wps_list):
        return mask

    nx, ny, _ = wps_list[base_idx]
    dx, dy, _ = wps_list[base_idx + 1]

    if _is_intercept_angle_ok(
        start_pos=(current_x, current_y),
        next_wp=(float(nx), float(ny)),
        downstream_wp=(float(dx), float(dy)),
    ):
        mask[int(Action.VECTOR_TO_ILS)] = True

    return mask


def _mask_risk_aware_no_op_action(
    mask: np.ndarray,
    predicted_infringement: Optional[float],
) -> np.ndarray:
    """Risk-aware gating for no-op action."""

    if predicted_infringement is None:
        return mask

    high_infringement_risk = predicted_infringement is not None and predicted_infringement >= 1.0

    if high_infringement_risk:
        mask[int(Action.DO_NOTHING)] = False

    return mask


def _is_intercept_angle_ok(
    start_pos: Tuple[float, float],
    next_wp: Tuple[float, float],
    downstream_wp: Tuple[float, float],
) -> bool:
    """True if angle between (start->next) and (next->downstream) <= configured max."""

    sx, sy = start_pos
    nx, ny = next_wp
    dx, dy = downstream_wp

    v1x, v1y = nx - sx, ny - sy
    v2x, v2y = dx - nx, dy - ny

    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < _EPS or n2 < _EPS:
        return False

    cos_angle = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    cos_thresh = math.cos(math.radians(Config.SKIP_MAX_INTERCEPT_ANGLE_DEG))
    return cos_angle >= cos_thresh
