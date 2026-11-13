"""Trombone route management for flight simulation.

This module provides classes for managing trombone (back-and-forth) sections
in aircraft routes, allowing dynamic insertion and removal of route segments.
"""

from abc import ABC, abstractmethod

from flight_simulator import RoutePoint


def _validate_positive_levels(n_levels: int) -> None:
    if n_levels <= 0:
        raise ValueError("n_levels must be positive")


def _route_contains_waypoint(route: list[RoutePoint], waypoint_name: str) -> bool:
    return any(point.waypoint_name == waypoint_name for point in route)


def _split_route_after_waypoint(
    route: list[RoutePoint], waypoint_name: str
) -> tuple[list[RoutePoint], list[RoutePoint]]:
    for index, point in enumerate(route):
        if point.waypoint_name == waypoint_name:
            return route[: index + 1], route[index + 1 :]
    raise ValueError(f"Waypoint {waypoint_name} not found in route")


class SectionInterface(ABC):
    """Interface for a section of route that can be extracted and inserted back."""

    @abstractmethod
    def extract_part(self, n_levels: int) -> tuple[list[RoutePoint], list[RoutePoint]]:
        pass

    @abstractmethod
    def peek_path_points(self, n_levels: int) -> list[RoutePoint]:
        """Return the route points that WOULD be inserted for n_levels, without consuming them."""

    @abstractmethod
    def forward_names(self) -> list[str]:
        pass

    @abstractmethod
    def backward_names(self) -> list[str]:
        pass

    @abstractmethod
    def num_levels(self) -> int:
        pass

    @abstractmethod
    def restore_level(self, level_points: list[RoutePoint]) -> None:
        """Restore one previously extracted level so it can be inserted again.

        For BackAndForthSection, *level_points* must be ordered
        ``[forward_point, backward_point]`` (i.e. the route-scan order).
        """


class OneWaySection(SectionInterface):
    """Represents a one-way section of route that can be extracted and inserted back."""

    def __init__(self, points: list[RoutePoint]):
        self._frwd_section = points

    def extract_part(self, n_levels: int) -> tuple[list[RoutePoint], list[RoutePoint]]:
        _validate_positive_levels(n_levels)
        if n_levels > len(self._frwd_section):
            raise ValueError(
                f"Requested n_levels={n_levels} exceeds available forward levels "
                f"({len(self._frwd_section)})"
            )
        new_fwd_points = self._frwd_section[:n_levels]
        self._frwd_section = self._frwd_section[n_levels:]

        return (new_fwd_points, [])

    def peek_path_points(self, n_levels: int) -> list[RoutePoint]:
        """Return the route points that WOULD be inserted for n_levels, without consuming them."""
        _validate_positive_levels(n_levels)
        if n_levels > len(self._frwd_section):
            raise ValueError(
                f"Requested n_levels={n_levels} exceeds available levels ({len(self._frwd_section)})"
            )
        return self._frwd_section[:n_levels]

    def forward_names(self) -> list[str]:
        return [wp.waypoint_name for wp in self._frwd_section]

    def backward_names(self) -> list[str]:
        return []

    def num_levels(self) -> int:
        return len(self._frwd_section)

    def restore_level(self, level_points: list[RoutePoint]) -> None:
        if len(level_points) != 1:
            raise ValueError(
                f"OneWaySection.restore_level expects 1 point, got {len(level_points)}"
            )
        self._frwd_section = level_points + self._frwd_section


class BackAndForthSection(SectionInterface):
    """Represents a back-and-forth section of a route split into forward and backward parts."""

    def __init__(self, points: list[RoutePoint]):
        num = len(points)
        if num % 2 != 0:
            raise ValueError(f"BackAndForthSection requires an even number of points, got {num}")
        half = num // 2
        self._frwd_section = points[:half]
        self._bckw_section = points[half:]

    def extract_part(self, n_levels: int) -> tuple[list[RoutePoint], list[RoutePoint]]:
        _validate_positive_levels(n_levels)
        if n_levels > len(self._frwd_section) or n_levels > len(self._bckw_section):
            raise ValueError(
                f"Requested n_levels={n_levels} exceeds available section levels "
                f"(forward={len(self._frwd_section)}, backward={len(self._bckw_section)})"
            )
        new_fwd_points = self._frwd_section[:n_levels]
        new_bck_points = self._bckw_section[-n_levels:]
        self._frwd_section = self._frwd_section[n_levels:]
        self._bckw_section = self._bckw_section[:-n_levels]

        return (new_fwd_points, new_bck_points)

    def peek_path_points(self, n_levels: int) -> list[RoutePoint]:
        """Return the route points that WOULD be inserted for n_levels, without consuming them."""
        _validate_positive_levels(n_levels)
        if n_levels > len(self._frwd_section):
            raise ValueError(
                f"Requested n_levels={n_levels} exceeds available levels ({len(self._frwd_section)})"
            )
        return self._frwd_section[:n_levels] + self._bckw_section[-n_levels:]

    def forward_names(self) -> list[str]:
        return [wp.waypoint_name for wp in self._frwd_section]

    def backward_names(self) -> list[str]:
        return [wp.waypoint_name for wp in self._bckw_section]

    def num_levels(self) -> int:
        if len(self._frwd_section) != len(self._bckw_section):
            raise RuntimeError("BackAndForthSection internal state is inconsistent")
        return len(self._frwd_section)

    def restore_level(self, level_points: list[RoutePoint]) -> None:
        if len(level_points) != 2:
            raise ValueError(
                f"BackAndForthSection.restore_level expects 2 points, got {len(level_points)}"
            )
        self._frwd_section = [level_points[0]] + self._frwd_section
        self._bckw_section = self._bckw_section + [level_points[1]]


class RouteSectionExtender:
    """Stores section excluded from a route and its position on route,
    allowing part of of section to be inserted back into the route."""

    def __init__(
        self,
        excluded: SectionInterface,
        last_point_before_excluded_points: str,
        first_point_after_excluded_points: str,
    ) -> None:
        self._excluded = excluded
        self._last_point_before_excluded = last_point_before_excluded_points
        self._original_last_point_before_excluded = last_point_before_excluded_points
        self._first_point_after_excluded = first_point_after_excluded_points
        self._retraction_log: list[frozenset[str]] = []

    def insert_section_to_route(self, n_levels: int, route: list[RoutePoint]) -> list[RoutePoint]:
        _validate_positive_levels(n_levels)

        # Split old route where new section will be inserted
        new_route_start, new_route_end = self._cut_route_from_named_point(
            route, self._last_point_before_excluded
        )

        # Before extraction: snapshot names
        frwd_before = self._excluded.forward_names()
        bckw_before = self._excluded.backward_names()

        new_frwd_points, new_bckw_points = self._excluded.extract_part(n_levels)

        # After extraction: compute per-level name sets
        for i in range(n_levels):
            level_names = {frwd_before[i]}
            if bckw_before and i < len(bckw_before):
                level_names.add(bckw_before[-(i + 1)])
            self._retraction_log.append(frozenset(level_names))

        self._last_point_before_excluded = new_frwd_points[-1].waypoint_name
        # If there was backward part, point after still excluded part changes
        if len(new_bckw_points) > 0:
            self._first_point_after_excluded = new_bckw_points[0].waypoint_name

        new_route = new_route_start + new_frwd_points + new_bckw_points + new_route_end
        return new_route

    def preview_section_to_route(self, n_levels: int, route: list[RoutePoint]) -> list[RoutePoint]:
        _validate_positive_levels(n_levels)

        new_route_start, new_route_end = self._cut_route_from_named_point(
            route, self._last_point_before_excluded
        )
        preview_points = self._excluded.peek_path_points(n_levels)
        return new_route_start + preview_points + new_route_end

    def num_levels_can_add_to_route(self, route: list[RoutePoint]) -> int:
        # If past cut point, cannot add trombone section anymore
        if not self.is_before_current_section_cut_point(route):
            return 0
        return self._excluded.num_levels()

    def is_before_current_section_cut_point(self, route: list[RoutePoint]) -> bool:
        """Has route not yet reached cut point to currently excluded section (in trombone, turning point to current route base edge)"""
        return _route_contains_waypoint(route, self._last_point_before_excluded)

    def is_before_upwind_start(self, route: list[RoutePoint]) -> bool:
        """Has route not yet reached upwind start point"""
        return _route_contains_waypoint(route, self._first_point_after_excluded)

    def has_reached_section(self, route: list[RoutePoint]) -> bool:
        """Has route reached the trombone (past original cut point)"""
        return not _route_contains_waypoint(route, self._original_last_point_before_excluded)

    def peek_all_excluded_points(self) -> list[RoutePoint]:
        """Return all excluded waypoints across all remaining levels."""
        num_levels = self._excluded.num_levels()
        if num_levels == 0:
            return []
        return self._excluded.peek_path_points(num_levels)

    @property
    def boundary_names(self) -> tuple[str, str]:
        """Return (before_excluded, after_excluded) boundary waypoint names."""
        return self._last_point_before_excluded, self._first_point_after_excluded

    def retract_one_level_from_route(self, route: list[RoutePoint]) -> list[RoutePoint] | None:
        retraction = self._preview_retraction(route)
        if retraction is None:
            return None

        new_route, removed_points, first_removed_idx, last_removed_idx = retraction
        # Points are collected in route order so forward precedes backward —
        # restore_level relies on this ordering for BackAndForthSection.
        self._excluded.restore_level(removed_points)
        self._last_point_before_excluded = route[first_removed_idx - 1].waypoint_name
        self._first_point_after_excluded = route[last_removed_idx + 1].waypoint_name
        self._retraction_log.pop()

        return new_route

    def preview_retract_one_level_from_route(
        self, route: list[RoutePoint]
    ) -> list[RoutePoint] | None:
        retraction = self._preview_retraction(route)
        if retraction is None:
            return None
        return retraction[0]

    def _preview_retraction(
        self, route: list[RoutePoint]
    ) -> tuple[list[RoutePoint], list[RoutePoint], int, int] | None:
        if not self._retraction_log:
            return None
        names_to_remove = self._retraction_log[-1]

        # Single pass: partition route into kept and removed points while
        # recording indices for boundary validation.
        new_route: list[RoutePoint] = []
        removed_points: list[RoutePoint] = []
        removed_indices: list[int] = []
        for index, wp in enumerate(route):
            if wp.waypoint_name in names_to_remove:
                removed_points.append(wp)
                removed_indices.append(index)
            else:
                new_route.append(wp)

        if len(removed_indices) != len(names_to_remove):
            return None

        first_removed_idx = removed_indices[0]
        last_removed_idx = removed_indices[-1]
        if first_removed_idx == 0 or last_removed_idx == len(route) - 1:
            return None

        return new_route, removed_points, first_removed_idx, last_removed_idx

    @property
    def num_retractable_levels(self) -> int:
        return len(self._retraction_log)

    def _cut_route_from_named_point(
        self, route: list[RoutePoint], cut_point_name: str
    ) -> tuple[list[RoutePoint], list[RoutePoint]]:
        return _split_route_after_waypoint(route, cut_point_name)


MXP_SECTION_START_MARKERS = ["MC761", "MC561"]
MXP_SECTION_END_MARKER = "INLER"

BGY_SECTION_START_MARKERS = ["ME632", "ME628"]
BGY_SECTION_END_MARKER = "TIXUM"

SECTION_START_MARKERS = MXP_SECTION_START_MARKERS + BGY_SECTION_START_MARKERS
SECTION_END_MARKERS = [MXP_SECTION_END_MARKER, BGY_SECTION_END_MARKER]
_SECTION_END_NAMES = frozenset(SECTION_END_MARKERS)


def is_past_section_end(route: list[RoutePoint]) -> bool:
    """True when no section-end marker (INLER/TIXUM) remains in the route."""
    return not any(rp.waypoint_name in _SECTION_END_NAMES for rp in route)


def cut_out_section_from_route(
    route: list[RoutePoint],
) -> tuple[list[RoutePoint], list[RoutePoint], RoutePoint, RoutePoint] | None:
    skipping = False
    remaining_part: list[RoutePoint] = []
    excluded_part: list[RoutePoint] = []
    before_exclusion_point: RoutePoint | None = None
    after_exclusion_point: RoutePoint | None = None
    for wp in route:
        if wp.waypoint_name in SECTION_END_MARKERS:
            skipping = False
            after_exclusion_point = wp

        if skipping:
            excluded_part.append(wp)
        else:
            remaining_part.append(wp)

        if skipping is False and (wp.waypoint_name in SECTION_START_MARKERS):
            skipping = True
            before_exclusion_point = wp

    if before_exclusion_point is None or after_exclusion_point is None:
        return None
    return remaining_part, excluded_part, before_exclusion_point, after_exclusion_point


class RouteSectionStore:
    """Stores and manages trombone sections for multiple aircraft routes."""

    def __init__(self) -> None:
        self._route_parts: dict[str, RouteSectionExtender] = {}
        # Shrunk convex hull of ALL excluded trombone waypoints (all families).
        # Covers the centre corridor between trombone legs — aircraft on the
        # legs (near edges) are outside this polygon and can vector to the side.
        # Computed once per episode from actual waypoint positions.
        self._trombone_exclusion_polygon: list[list[float]] | None = None

    def clear(self) -> None:
        """Clear all stored route section parts."""
        self._route_parts.clear()
        self._trombone_exclusion_polygon = None

    def cut_out_section(self, ac_name: str, route: list[RoutePoint]) -> list[RoutePoint]:
        result = cut_out_section_from_route(route)
        if result is None:
            # Some routes (e.g. certain externally provided worlds) may not contain
            # section markers. Keep route unchanged and mark as non-section-capable.
            self._route_parts.pop(ac_name, None)
            return route

        remaining_part, excluded_part, before_exclusion_point, after_exclusion_point = result

        # Store excluded part and cut point for possible later re-addition
        # Check if the removed part was trombone (back-and-forth) or point-merge (one-way) section
        excluded: SectionInterface | None = None
        if before_exclusion_point.waypoint_name in ["MC761", "MC561"]:
            excluded = BackAndForthSection(excluded_part)
        else:
            excluded = OneWaySection(excluded_part)
        self._route_parts[ac_name] = RouteSectionExtender(
            excluded,
            before_exclusion_point.waypoint_name,
            after_exclusion_point.waypoint_name,
        )
        return remaining_part

    def add_section_to_route(
        self, ac_name: str, route: list[RoutePoint], n_levels: int
    ) -> list[RoutePoint]:
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return route
        return tr.insert_section_to_route(n_levels, route)

    def preview_section_to_route(
        self, ac_name: str, route: list[RoutePoint], n_levels: int
    ) -> list[RoutePoint]:
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return route
        return tr.preview_section_to_route(n_levels, route)

    def num_levels_can_add_to_route(self, ac_name: str, route: list[RoutePoint]) -> int:
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return 0
        return tr.num_levels_can_add_to_route(route)

    def retract_section_from_route(
        self, ac_name: str, route: list[RoutePoint]
    ) -> list[RoutePoint] | None:
        extender = self._route_parts.get(ac_name)
        if extender is None:
            return None
        return extender.retract_one_level_from_route(route)

    def preview_retract_section_from_route(
        self, ac_name: str, route: list[RoutePoint]
    ) -> list[RoutePoint] | None:
        extender = self._route_parts.get(ac_name)
        if extender is None:
            return None
        return extender.preview_retract_one_level_from_route(route)

    def num_levels_can_retract(self, ac_name: str) -> int:
        extender = self._route_parts.get(ac_name)
        if extender is None:
            return 0
        return extender.num_retractable_levels

    def has_excluded_section(self, ac_name: str) -> bool:
        """Whether the aircraft has an excluded section (e.g. trombone) stored."""
        return ac_name in self._route_parts

    def is_before_current_section_cut_point(self, ac_name: str, route: list[RoutePoint]) -> bool:
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return False
        return tr.is_before_current_section_cut_point(route)

    def is_before_upwind_start(self, ac_name: str, route: list[RoutePoint]) -> bool:
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return False
        return tr.is_before_upwind_start(route)

    def has_reached_section(self, ac_name: str, route: list[RoutePoint]) -> bool:
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return False
        return tr.has_reached_section(route)

    def get_all_excluded_points(self, ac_name: str) -> list[RoutePoint] | None:
        """Return all excluded route points for an aircraft, or None if no section."""
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return None
        points = tr.peek_all_excluded_points()
        return points if points else None

    def get_section_boundary_names(self, ac_name: str) -> tuple[str, str] | None:
        """Return the (before, after) boundary waypoint names for an aircraft's section."""
        tr = self._route_parts.get(ac_name)
        if tr is None:
            return None
        return tr.boundary_names

    def set_trombone_exclusion_polygon(self, polygon: list[list[float]]) -> None:
        """Store the trombone exclusion polygon (shrunk to centre corridor)."""
        self._trombone_exclusion_polygon = polygon

    def get_exclusion_polygon(self, ac_name: str) -> list[list[float]] | None:
        """Get the trombone exclusion polygon, or None if not applicable."""
        if not self.has_excluded_section(ac_name):
            return None
        return self._trombone_exclusion_polygon
