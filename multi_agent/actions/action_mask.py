"""Functions for calculating valid action masks for aircraft."""

import math
from collections.abc import Mapping

import numpy as np
from flight_simulator import RoutePoint, Vec3

from actions.action_table import (
    NUM_ACTIONS,
    SKIP_SPECS,
    SPEED_DELTAS,
    TURN_ACTION_SPECS,
    Action,
    speed_delta,
)
from actions.actions import turn_direction_multiplier
from config.settings import ActionSettings
from geometry.vectoring import (
    find_vector_intercept_index,
    get_pending_virtual_turn_anchor_xy,
    point_in_polygon,
    split_leading_virtual_turns,
)
from simulator.design_parser import AircraftDesignData
from simulator.route_section_store import RouteSectionStore

_EPS = 1e-9
_VALID_CUSTOM_ACTION_MASK_VALUES = frozenset({0, 1})


def detect_airport_from_route(
    route: list[RoutePoint], airports: Mapping[str, object]
) -> str | None:
    """Detect airport code from route waypoint names.

    Scans waypoints from the end (where airport-specific names appear)
    and matches against the *airports* keys (``custom_action_mask_by_airport``).
    """
    if not airports:
        return None
    for point in reversed(route):
        name = point.waypoint_name
        for code in airports:
            if name == code or name.startswith(f"{code}_"):
                return code
    return None


def compute_action_mask(  # pylint: disable=too-many-positional-arguments
    ac_name: str,
    current_pos: Vec3,
    target_vel: float,
    current_vel: Vec3,
    wps_list: list[tuple[float, float, float]],
    route: list[RoutePoint],
    design_data: AircraftDesignData,
    target_heading_deg: float | None,
    route_trombone_store: RouteSectionStore,
    ego_infringements: list | None = None,
    airport: str | None = None,
    *,
    settings: ActionSettings,
) -> np.ndarray:
    """Return valid actions using complete, position-aligned physical route geometry.

    ``wps_list`` must resolve every ``route`` entry, independently of observation K.
    """

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[int(Action.DO_NOTHING)] = True

    mask = _mask_speed_actions(mask, target_vel, design_data, route, settings)

    if len(wps_list) > settings.waypoints_to_leave_at_minimum:
        if ego_infringements:
            exclusion_polygon = route_trombone_store.get_exclusion_polygon(ac_name)
            mask = _mask_vectoring_actions(
                mask,
                current_pos,
                current_vel,
                target_heading_deg,
                wps_list,
                route,
                exclusion_polygon=exclusion_polygon,
                settings=settings,
            )
        mask = _mask_skip_waypoints_actions(mask, current_pos, wps_list, settings)
        mask = _mask_trombone_actions(
            ac_name=ac_name,
            mask=mask,
            route=route,
            route_trombone_store=route_trombone_store,
            min_waypoints=settings.waypoints_to_leave_at_minimum,
        )
        mask = _mask_vector_to_ils_action(
            ac_name=ac_name,
            mask=mask,
            current_pos=current_pos,
            wps_list=wps_list,
            route=route,
            route_trombone_store=route_trombone_store,
            settings=settings,
        )

    custom_action_mask = _validated_custom_action_mask(
        settings.custom_action_mask_by_airport.get(airport) if airport else None
    )
    if custom_action_mask is not None:
        mask &= custom_action_mask

    assert mask.any(), "action_mask ended up empty: this must never happen"
    return mask


def _mask_speed_actions(
    mask: np.ndarray,
    target_vel: float,
    design_data: AircraftDesignData,
    route: list[RoutePoint],
    settings: ActionSettings,
) -> np.ndarray:
    """Mask speed actions: block actions that violate bounds or produce no change."""

    r = settings.speed_resolution
    rounded_target = round(target_vel / r) * r

    buf = settings.speed_limit_buffer_kt
    min_vel = float(design_data.min_speed) + buf
    design_max = float(design_data.max_speed) - buf
    next_wp_speed = route[0].target_speed if route else None
    max_vel = min(next_wp_speed, design_max) if next_wp_speed is not None else design_max

    def _allowed(delta: float) -> bool:
        new = round((target_vel + delta) / r) * r
        return min_vel <= new <= max_vel and new != rounded_target

    for action in SPEED_DELTAS:
        mask[int(action)] = _allowed(speed_delta(action, settings))

    return mask


def _validated_custom_action_mask(custom_action_mask) -> np.ndarray | None:
    if custom_action_mask is None:
        return None

    if len(custom_action_mask) != NUM_ACTIONS:
        raise ValueError(
            "CUSTOM_ACTION_MASK length must match NUM_ACTIONS: "
            f"{len(custom_action_mask)} != {NUM_ACTIONS}"
        )

    if any(value not in _VALID_CUSTOM_ACTION_MASK_VALUES for value in custom_action_mask):
        raise ValueError("CUSTOM_ACTION_MASK values must be 0/1 booleans")

    if int(custom_action_mask[int(Action.DO_NOTHING)]) != 1:
        raise ValueError("CUSTOM_ACTION_MASK must keep DO_NOTHING enabled")

    return np.asarray(custom_action_mask, dtype=bool)


def _skip_has_meaningful_deviation(
    start_xy: tuple[float, float],
    wps_list: list[tuple[float, float, float]],
    offset: int,
    total_skip: int,
    min_deviation_cos: float,
) -> bool:
    """True if any intermediate waypoint in the skip region bends enough to matter.

    Builds the point sequence [start, wps[offset], ..., wps[total_skip]] and
    checks each interior vertex angle against the threshold.
    """
    points: list[tuple[float, float]] = [start_xy]
    for i in range(offset, total_skip + 1):
        points.append(wps_list[i][:2])
    for i in range(1, len(points) - 1):
        cos = _vertex_angle_cos(points[i - 1], points[i], points[i + 1])
        if cos is not None and cos <= min_deviation_cos:
            return True
    return False


def _mask_skip_waypoints_actions(
    mask: np.ndarray,
    current_pos: Vec3,
    wps_list: list[tuple[float, float, float]],
    settings: ActionSettings,
) -> np.ndarray:
    """Mask skip waypoints actions"""

    flight_plan_length = len(wps_list)

    # The last N waypoints are protected and must never be skipped.
    protected_start = flight_plan_length - settings.waypoints_to_leave_at_minimum
    min_dev_cos = math.cos(math.radians(settings.skip_min_deviation_deg))

    for action, (skip_amount, offset) in SKIP_SPECS.items():
        total_skip = offset + skip_amount
        # The skip region [offset, offset+skip_amount) must end before the protected tail.
        if total_skip > protected_start:
            continue

        # Intercept-angle check needs both target and downstream waypoints.
        if (total_skip + 1) >= flight_plan_length:
            continue

        start_xy = (current_pos.x, current_pos.y) if offset == 0 else wps_list[0][:2]

        # Require at least one meaningful turn in the skipped segment;
        # if the route is nearly straight, skipping has no effect.
        if not _skip_has_meaningful_deviation(start_xy, wps_list, offset, total_skip, min_dev_cos):
            continue

        if _is_intercept_angle_ok(
            start_pos=start_xy,
            next_wp=wps_list[total_skip][:2],
            downstream_wp=wps_list[total_skip + 1][:2],
            max_intercept_angle_deg=settings.skip_max_intercept_angle_deg,
        ):
            mask[action] = True

    return mask


def _mask_trombone_actions(
    ac_name: str,
    mask: np.ndarray,
    route: list[RoutePoint],
    route_trombone_store: RouteSectionStore,
    min_waypoints: int,
) -> np.ndarray:
    if route_trombone_store.num_levels_can_add_to_route(ac_name, route) > 0:
        mask[int(Action.LENGTHEN_TROMBONE)] = True
    preview_retracted_route = route_trombone_store.preview_retract_section_from_route(
        ac_name, route
    )
    if preview_retracted_route is not None and len(preview_retracted_route) >= min_waypoints:
        mask[int(Action.SHORTEN_TROMBONE)] = True
    return mask


def _mask_vectoring_actions(
    mask: np.ndarray,
    current_pos: Vec3,
    current_vel: Vec3,
    target_heading_deg: float | None,
    wps_list: list[tuple[float, float, float]],
    route: list[RoutePoint],
    exclusion_polygon: list[list[float]] | None = None,
    *,
    settings: ActionSettings,
) -> np.ndarray:
    """Mask vectoring actions.

    Uses a single shrunk exclusion polygon (centre corridor of the trombone):
      - Aircraft inside the polygon → block ALL vectoring (would cross traffic)
      - Turn point inside the polygon → block that specific vector direction
    Aircraft on trombone legs (near polygon edges, outside shrunk area) can
    vector to the side — matching real Milano ATC vectoring patterns.
    """
    heading_rad = (
        math.radians(target_heading_deg)
        if target_heading_deg is not None
        else math.atan2(current_vel.y, current_vel.x)
    )

    max_return_angle_rad = settings.max_turn_angle_after_heading_adjustment * math.pi / 180.0
    heading_adjustment_rad = settings.action_heading_adjustment * math.pi / 180.0

    pending_virtual_prefix, downstream_route = split_leading_virtual_turns(route)
    virtual_prefix_len = len(pending_virtual_prefix)
    downstream_wps = wps_list[virtual_prefix_len:] if virtual_prefix_len > 0 else wps_list

    pending_virtual_anchor = get_pending_virtual_turn_anchor_xy(pending_virtual_prefix)
    if pending_virtual_anchor is not None:
        current_x, current_y = pending_virtual_anchor
    else:
        current_x = float(current_pos.x)
        current_y = float(current_pos.y)

    if min(len(downstream_route), len(downstream_wps)) < settings.waypoints_to_leave_at_minimum:
        return mask

    # Block all vectoring when the aircraft is inside the exclusion polygon
    # (centre corridor of the trombone). Vectoring from here would cross
    # opposite-direction trombone leg traffic.
    if exclusion_polygon is not None and point_in_polygon(
        (current_x, current_y), exclusion_polygon
    ):
        return mask

    for action_idx, turn_spec in TURN_ACTION_SPECS.items():
        if evaluate_turn_feasibility(
            current_pos=(current_x, current_y),
            heading_rad=heading_rad,
            wps_list=downstream_wps,
            turn_distance=float(turn_spec.distance_in_nm),
            turn_direction_sign=turn_direction_multiplier(turn_spec.turn_direction),
            heading_adjustment_rad=heading_adjustment_rad,
            max_return_angle_rad=max_return_angle_rad,
            exclusion_polygon=exclusion_polygon,
            max_intercept_angle_deg=settings.skip_max_intercept_angle_deg,
            min_remaining_waypoints=settings.waypoints_to_leave_at_minimum,
        ):
            mask[action_idx] = True

    return mask


def evaluate_turn_feasibility(
    current_pos: tuple[float, float],
    heading_rad: float,
    wps_list: list[tuple[float, float, float]],
    turn_distance: float,
    turn_direction_sign: int,
    heading_adjustment_rad: float,
    max_return_angle_rad: float,
    exclusion_polygon: list[list[float]] | None = None,
    *,
    max_intercept_angle_deg: float,
    min_remaining_waypoints: int,
) -> bool:
    """Evaluate turn feasibility by scanning downstream waypoints for a valid re-entry.

    Instead of checking only the immediate next waypoint, scans up to
    MAX_VECTOR_INTERCEPT_SKIP waypoints ahead. This makes vectoring feasible
    on trombone legs where the next waypoint is a 180° turn-back.

    If an exclusion polygon is provided (convex hull of trombone waypoints),
    vectors whose turn point would land inside it are rejected — prevents
    the aircraft from vectoring through other aircraft's trombone legs.
    """
    if len(wps_list) < 2:
        return False

    current_x, current_y = current_pos
    new_heading_rad = heading_rad + turn_direction_sign * heading_adjustment_rad
    new_x = current_x + turn_distance * math.cos(new_heading_rad)
    new_y = current_y + turn_distance * math.sin(new_heading_rad)

    # Reject vectors that would place the turn point inside the trombone area
    if exclusion_polygon is not None and point_in_polygon((new_x, new_y), exclusion_polygon):
        return False

    downstream_positions = [(float(w[0]), float(w[1])) for w in wps_list]

    return (
        find_vector_intercept_index(
            turn_point_xy=(new_x, new_y),
            new_heading_rad=new_heading_rad,
            downstream_positions=downstream_positions,
            max_return_angle_rad=max_return_angle_rad,
            max_intercept_angle_deg=max_intercept_angle_deg,
            min_remaining_waypoints=min_remaining_waypoints,
        )
        >= 0
    )


def _mask_vector_to_ils_action(
    ac_name: str,
    mask: np.ndarray,
    current_pos: Vec3,
    wps_list: list[tuple[float, float, float]],
    route: list[RoutePoint],
    route_trombone_store: RouteSectionStore,
    *,
    settings: ActionSettings,
) -> np.ndarray:
    """Mask vector to ILS action"""

    # Do not allow vector to ILS if aircraft has no excluded section,
    # if not yet on trombone, or if already on upwind side (already in ILS direction)
    if not route_trombone_store.has_excluded_section(ac_name):
        return mask
    if not route_trombone_store.has_reached_section(
        ac_name, route
    ) or not route_trombone_store.is_before_upwind_start(ac_name, route):
        return mask

    current_x = float(current_pos.x)
    current_y = float(current_pos.y)

    tail_len = settings.waypoints_to_leave_at_minimum
    base_idx = len(wps_list) - tail_len

    nx, ny, _ = wps_list[base_idx]
    dx, dy, _ = wps_list[base_idx + 1]

    if _is_intercept_angle_ok(
        start_pos=(current_x, current_y),
        next_wp=(float(nx), float(ny)),
        downstream_wp=(float(dx), float(dy)),
        max_intercept_angle_deg=settings.skip_max_intercept_angle_deg,
    ):
        mask[int(Action.VECTOR_TO_ILS)] = True

    return mask


def _vertex_angle_cos(
    start: tuple[float, float],
    vertex: tuple[float, float],
    end: tuple[float, float],
) -> float | None:
    """Cosine of the turn angle at *vertex* on path start→vertex→end.

    Returns None if either segment is degenerate (near-zero length).
    cos=1 means straight, cos=0 means 90° turn, cos=-1 means U-turn.
    """
    v1x, v1y = vertex[0] - start[0], vertex[1] - start[1]
    v2x, v2y = end[0] - vertex[0], end[1] - vertex[1]

    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < _EPS or n2 < _EPS:
        return None

    cos_angle = (v1x * v2x + v1y * v2y) / (n1 * n2)
    return max(-1.0, min(1.0, cos_angle))


def _is_intercept_angle_ok(
    start_pos: tuple[float, float],
    next_wp: tuple[float, float],
    downstream_wp: tuple[float, float],
    max_intercept_angle_deg: float,
) -> bool:
    """True if angle between (start->next) and (next->downstream) <= *max_intercept_angle_deg*."""
    cos = _vertex_angle_cos(start_pos, next_wp, downstream_wp)
    if cos is None:
        return False
    return cos >= math.cos(math.radians(max_intercept_angle_deg))
