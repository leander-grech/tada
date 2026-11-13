"""Shared helpers for vectoring turn actions and masking."""

import math
from collections.abc import Sequence
from typing import Any

VIRTUAL_TURN_WAYPOINT_PREFIX = "VEC_TURN_"

# Maximum number of downstream waypoints to skip when searching for a
# feasible vector re-entry point.  Keeps the single-action effect bounded
# and predictable for RL while still allowing vectors on trombone legs
# (which need to skip past at least one 180° turn-back).
MAX_VECTOR_INTERCEPT_SKIP = 4


def find_vector_intercept_index(
    turn_point_xy: tuple[float, float],
    new_heading_rad: float,
    downstream_positions: Sequence[tuple[float, float]],
    max_return_angle_rad: float,
    max_intercept_angle_deg: float,
    min_remaining_waypoints: int,
) -> int:
    """Find the first downstream waypoint where the aircraft can rejoin the route.

    Scans up to MAX_VECTOR_INTERCEPT_SKIP waypoints ahead, checking:
      1. Return angle from turn point to candidate waypoint (≤ max_return_angle_rad)
      2. Intercept angle from turn point → candidate → next waypoint (≤ max_intercept_angle_deg)
      3. Enough waypoints remain after the candidate

    Returns the 0-based index into downstream_positions, or -1 if none feasible.
    """
    tx, ty = turn_point_xy
    cos_max_intercept = math.cos(math.radians(max_intercept_angle_deg))
    scan_limit = min(len(downstream_positions) - 1, MAX_VECTOR_INTERCEPT_SKIP)

    for i in range(scan_limit):
        remaining = len(downstream_positions) - i
        if remaining < min_remaining_waypoints:
            break

        wx, wy = downstream_positions[i]
        dwx, dwy = downstream_positions[i + 1]

        # 1. Return angle: can the aircraft fly from turn point to this waypoint?
        #    Modulo normalises to [0, 2π) so the wrap to [0, π] is always valid
        #    (atan2 ∈ [-π,π] and new_heading_rad may exceed that range).
        wp_heading = math.atan2(wy - ty, wx - tx)
        angle_diff = (wp_heading - new_heading_rad) % (2.0 * math.pi)
        if angle_diff > math.pi:
            angle_diff = 2.0 * math.pi - angle_diff
        if angle_diff > max_return_angle_rad:
            continue

        # 2. Intercept angle: is the path turn_point → candidate → downstream smooth?
        v1x, v1y = wx - tx, wy - ty
        v2x, v2y = dwx - wx, dwy - wy
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_angle = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        if cos_angle >= cos_max_intercept:
            return i

    return -1


PolygonXY = list[list[float]]


def convex_hull(points: Sequence[tuple[float, float]]) -> PolygonXY:
    """Andrew's monotone-chain convex hull. Returns vertices in CCW order.

    For small point sets (trombone waypoints, ~10-20 points) this is
    effectively free.  Used once per episode to build the trombone
    exclusion polygon.
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return [[p[0], p[1]] for p in pts]

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [[p[0], p[1]] for p in hull]


def shrink_polygon(polygon: PolygonXY, factor: float = 0.5) -> PolygonXY:
    """Shrink a polygon toward its centroid by the given factor (0–1).

    factor=1.0 returns the original; factor=0.5 shrinks to 50% size.
    Used to create an inner exclusion zone that blocks vectoring from the
    trombone centre while allowing it from the sides (actual trombone legs).
    """
    n = len(polygon)
    if n < 3:
        return polygon
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return [[cx + factor * (p[0] - cx), cy + factor * (p[1] - cy)] for p in polygon]


def point_in_polygon(point_xy: tuple[float, float], polygon: PolygonXY) -> bool:
    """Ray-casting point-in-polygon test (works for any simple polygon)."""
    x, y = point_xy
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def resolve_waypoint_position(
    rp: Any,
    world_waypoints: dict,
) -> tuple[float, float] | None:
    """Resolve (x, y) for a single route point using world waypoints + inline fallback."""
    wp = world_waypoints.get(str(getattr(rp, "waypoint_name", "")))
    pos_nm = getattr(wp, "position_nm", None) if wp is not None else None
    if pos_nm is not None:
        return float(pos_nm.x), float(pos_nm.y)
    rp_pos = getattr(rp, "position_nm", None)
    if rp_pos is not None:
        return float(rp_pos.x), float(rp_pos.y)
    return None


def resolve_route_positions(
    route: Sequence[Any],
    world_waypoints: dict,
) -> list[tuple[float, float]]:
    """Resolve (x, y) positions for a route. Stops at the first unresolvable point."""
    positions: list[tuple[float, float]] = []
    for rp in route:
        pos = resolve_waypoint_position(rp, world_waypoints)
        if pos is None:
            break
        positions.append(pos)
    return positions


def _is_virtual_turn_route_point(route_point: Any) -> bool:
    waypoint_name = str(getattr(route_point, "waypoint_name", ""))
    position_nm = getattr(route_point, "position_nm", None)
    return waypoint_name.startswith(VIRTUAL_TURN_WAYPOINT_PREFIX) and position_nm is not None


def get_pending_virtual_turn_anchor_xy(
    pending_virtual_prefix: Sequence[Any],
) -> tuple[float, float] | None:
    """Return anchor XY from the tail of the pending virtual-turn prefix."""
    if not pending_virtual_prefix:
        return None
    # For queued turns, the next generated turn starts from the tail of the
    # existing virtual-turn prefix.
    position_nm = pending_virtual_prefix[-1].position_nm
    return float(position_nm.x), float(position_nm.y)


def leading_virtual_turn_count(route: Sequence[Any]) -> int:
    count = 0
    for route_point in route:
        if _is_virtual_turn_route_point(route_point):
            count += 1
        else:
            break
    return count


def split_leading_virtual_turns(route: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    virtual_prefix_len = leading_virtual_turn_count(route)
    return list(route[:virtual_prefix_len]), list(route[virtual_prefix_len:])
