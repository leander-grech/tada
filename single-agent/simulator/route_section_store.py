"""Trombone route management for flight simulation.

This module provides classes for managing trombone (back-and-forth) sections
in aircraft routes, allowing dynamic insertion and removal of route segments.
"""

from abc import ABC, abstractmethod

from flight_simulator import RoutePoint

from config.config import Config

# BGY point merge (use case 2): the sequencing arc, in flying order, and the merge point every
# leg converges on (simulation_config/scenario_generator/aman.toml, airports.BGY).
POINT_MERGE_ARC = ("ME632", "ME628", "ME624", "ME620", "ME616", "ULVUN")
MERGE_POINT = "TIXUM"


class SectionInterface(ABC):
    """Interface for a section of route that can be extracted and inserted back."""

    @abstractmethod
    def clone(self) -> "SectionInterface":
        """A copy safe to mutate independently, SHARING the RoutePoint objects.

        ``copy.deepcopy`` cannot be used on anything holding RoutePoints: they are PyO3
        objects and raise ``TypeError: cannot pickle 'builtins.RoutePoint' object``. Sharing
        them is correct regardless — this module only ever slices lists of RoutePoints and
        reads ``.waypoint_name``, and never mutates a RoutePoint itself.
        """

    @abstractmethod
    def extract_part(self, n_levels: int) -> tuple[list[RoutePoint], list[RoutePoint]]:
        pass

    @abstractmethod
    def return_part(self, fwd: list[RoutePoint], bck: list[RoutePoint]) -> None:
        """Exact inverse of :meth:`extract_part` — put previously extracted points back.

        Required so LENGTHEN_TROMBONE and SHORTEN_TROMBONE are true opposites: the store must
        end up in the state it would have been in had the lengthen never happened, otherwise
        repeated lengthen/shorten cycles would leak levels.
        """

    @abstractmethod
    def num_levels(self) -> int:
        pass


class OneWaySection(SectionInterface):
    """Represents a one-way section of route that can be extracted and inserted back."""

    def __init__(self, points: list[RoutePoint]):
        self._frwd_section = points

    def extract_part(self, n_levels: int) -> tuple[list[RoutePoint], list[RoutePoint]]:
        assert n_levels <= len(self._frwd_section)
        new_fwd_points = self._frwd_section[:n_levels]
        self._frwd_section = self._frwd_section[n_levels:]

        return (new_fwd_points, [])

    def return_part(self, fwd: list[RoutePoint], bck: list[RoutePoint]) -> None:
        # extract_part took from the FRONT, so return to the front.
        assert not bck, "one-way sections never yield backward points"
        self._frwd_section = list(fwd) + self._frwd_section

    def num_levels(self) -> int:
        return len(self._frwd_section)

    def clone(self) -> "OneWaySection":
        new = OneWaySection.__new__(OneWaySection)
        new._frwd_section = list(self._frwd_section)
        return new


class BackAndForthSection(SectionInterface):
    """Represents a back-and-forth section of a route split into forward and backward parts."""

    def __init__(self, points: list[RoutePoint]):
        num = len(points)
        assert num % 2 == 0
        half = num // 2
        self._frwd_section = points[:half]
        self._bckw_section = points[half:]

    def extract_part(self, n_levels: int) -> tuple[list[RoutePoint], list[RoutePoint]]:
        assert n_levels <= len(self._frwd_section) and n_levels <= len(self._bckw_section)
        new_fwd_points = self._frwd_section[:n_levels]
        new_bck_points = self._bckw_section[-n_levels:]
        self._frwd_section = self._frwd_section[n_levels:]
        self._bckw_section = self._bckw_section[:-n_levels]

        return (new_fwd_points, new_bck_points)

    def return_part(self, fwd: list[RoutePoint], bck: list[RoutePoint]) -> None:
        # extract_part took the FRONT of the outbound leg and the BACK of the inbound one,
        # so each goes back where it came from.
        self._frwd_section = list(fwd) + self._frwd_section
        self._bckw_section = self._bckw_section + list(bck)

    def num_levels(self) -> int:
        assert len(self._frwd_section) == len(self._bckw_section)
        return len(self._frwd_section)

    def clone(self) -> "BackAndForthSection":
        new = BackAndForthSection.__new__(BackAndForthSection)
        new._frwd_section = list(self._frwd_section)
        new._bckw_section = list(self._bckw_section)
        return new


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
        # Undo stack, so SHORTEN can be the exact inverse of LENGTHEN. Each entry records the
        # points a single insertion added plus the cut-point names it overwrote, which is
        # everything needed to put the extender back into its prior state.
        self._inserted: list[tuple[list[RoutePoint], list[RoutePoint], str, str]] = []

    def insert_section_to_route(self, n_levels: int, route: list[RoutePoint]) -> list[RoutePoint]:
        assert n_levels > 0

        # Split old route where new section will be inserted
        new_route_start, new_route_end = self._cut_route_from_named_point(
            route, self._last_point_before_excluded
        )

        prev_last = self._last_point_before_excluded
        prev_first = self._first_point_after_excluded

        new_frwd_points, new_bckw_points = self._excluded.extract_part(n_levels)
        self._last_point_before_excluded = new_frwd_points[-1].waypoint_name
        # If there was backward part, point after still excluded part changes
        if len(new_bckw_points) > 0:
            self._first_point_after_excluded = new_bckw_points[0].waypoint_name

        self._inserted.append((new_frwd_points, new_bckw_points, prev_last, prev_first))

        new_route = new_route_start + new_frwd_points + new_bckw_points + new_route_end
        return new_route

    def remove_section_from_route(self, n_levels: int, route: list[RoutePoint]) -> list[RoutePoint]:
        """Undo the most recent ``n_levels`` insertions, returning the shortened route.

        Insertions nest — the route reads A, f1..fn, bn..b1, B — so they can only be undone in
        reverse order, hence a stack rather than an index. Each pop restores the cut-point
        names the insertion overwrote and hands its waypoints back to the excluded section, so
        the extender ends up byte-identical to its pre-insertion state and repeated
        lengthen/shorten cycles cannot leak levels.

        Only insertions the aircraft has not yet flown can be undone; see
        :meth:`num_levels_can_remove_from_route`. Requesting more than that removes what it
        can and stops, which keeps the action safe to leave unmasked.
        """
        assert n_levels > 0
        names_in_route = {wp.waypoint_name for wp in route}
        new_route = list(route)

        for _ in range(n_levels):
            if not self._inserted:
                break
            fwd, bck, prev_last, prev_first = self._inserted[-1]
            drop = {wp.waypoint_name for wp in fwd} | {wp.waypoint_name for wp in bck}
            # Already flown past part of this level: it is no longer ours to remove.
            if not drop.issubset(names_in_route):
                break

            self._inserted.pop()
            new_route = [wp for wp in new_route if wp.waypoint_name not in drop]
            names_in_route -= drop
            self._excluded.return_part(fwd, bck)
            self._last_point_before_excluded = prev_last
            self._first_point_after_excluded = prev_first

        return new_route

    def num_levels_can_remove_from_route(self, route: list[RoutePoint]) -> int:
        """How many inserted levels are still ahead of the aircraft and therefore removable."""
        names_in_route = {wp.waypoint_name for wp in route}
        count = 0
        for fwd, bck, _prev_last, _prev_first in reversed(self._inserted):
            drop = {wp.waypoint_name for wp in fwd} | {wp.waypoint_name for wp in bck}
            if not drop.issubset(names_in_route):
                break
            count += 1
        return count

    def num_levels_can_add_to_route(self, route: list[RoutePoint]) -> int:
        # If past cut point, cannot add trombone section anymore
        if not self.is_before_current_section_cut_point(route):
            return 0
        return self._excluded.num_levels()

    def is_before_current_section_cut_point(self, route: list[RoutePoint]) -> bool:
        """Has route not yet reached cut point to currently excluded section (in trombone, turning point to current route base edge)"""
        return self._last_point_before_excluded in [wp.waypoint_name for wp in route]

    def is_before_upwind_start(self, route: list[RoutePoint]) -> bool:
        """Has route not yet reached upwind start point"""
        return self._first_point_after_excluded in [wp.waypoint_name for wp in route]

    def has_reached_section(self, route: list[RoutePoint]) -> bool:
        """Has route reached the trombone (past original cut point)"""
        return self._original_last_point_before_excluded not in [wp.waypoint_name for wp in route]

    def _cut_route_from_named_point(
        self, route: list[RoutePoint], cut_point_name: str
    ) -> tuple[list[RoutePoint], list[RoutePoint]]:
        cut_index = [wp.waypoint_name for wp in route].index(cut_point_name)
        return route[: cut_index + 1], route[cut_index + 1 :]

    def clone(self) -> "RouteSectionExtender":
        new = RouteSectionExtender(
            self._excluded.clone(),
            self._last_point_before_excluded,
            self._first_point_after_excluded,
        )
        new._original_last_point_before_excluded = self._original_last_point_before_excluded
        # Copy the undo stack's list containers; the RoutePoints inside are shared.
        new._inserted = [
            (list(fwd), list(bck), pl, pf) for fwd, bck, pl, pf in self._inserted
        ]
        return new


class PointMergeLeg:
    """Canonical point merge (``TADA_ACTION_SET=v2pms``): the flight keeps its WHOLE sequencing
    leg and the controller shortcuts it with a direct-to the merge point.

    A flight follows the leg by default and the controller turns it direct to the merge point
    when spacing allows. Only one point is stored per
    flight: the arc point where its leg begins (ME632, or ME628 for the flights that join the
    arc there). The flight is ON the leg once that point is behind it.
    """

    def __init__(self, entry: str) -> None:
        self._entry = entry

    def clone(self) -> "PointMergeLeg":
        return PointMergeLeg(self._entry)

    def can_direct_to_merge(self, route: list[RoutePoint]) -> bool:
        """On the leg (entry point flown), with at least one arc point before the merge point."""
        names = [wp.waypoint_name for wp in route]
        if MERGE_POINT not in names or self._entry in names:
            return False
        ahead = names[: names.index(MERGE_POINT)]
        return len(ahead) > 0 and all(n in POINT_MERGE_ARC for n in ahead)

    def direct_to_merge_route(self, route: list[RoutePoint]) -> list[RoutePoint]:
        names = [wp.waypoint_name for wp in route]
        return list(route[names.index(MERGE_POINT):])


def cut_out_section_from_route(
    route: list[RoutePoint],
) -> tuple[list[RoutePoint], list[RoutePoint], RoutePoint, RoutePoint]:
    skipping = False
    remaining_part: list[RoutePoint] = []
    excluded_part: list[RoutePoint] = []
    before_exclusion_point: RoutePoint | None = None
    after_exclusion_point: RoutePoint | None = None
    for wp in route:
        if wp.waypoint_name in ["INLER", "TIXUM"]:
            skipping = False
            after_exclusion_point = wp

        if skipping:
            excluded_part.append(wp)
        else:
            remaining_part.append(wp)

        if skipping is False and (wp.waypoint_name in ["MC761", "MC561", "ME362", "ME628"]):
            skipping = True
            before_exclusion_point = wp

    assert before_exclusion_point is not None
    assert after_exclusion_point is not None
    return remaining_part, excluded_part, before_exclusion_point, after_exclusion_point


class RouteSectionStore:
    """Stores and manages trombone sections for multiple aircraft routes."""

    def __init__(self) -> None:
        self._route_parts: dict[str, RouteSectionExtender | PointMergeLeg] = {}
        # Point merge is procedure dependent: with the v2pms clearance set
        # a point-merge route keeps its whole leg (PointMergeLeg) instead of being cut.
        self.canonical_point_merge = str(Config.ACTION_SET).lower() == "v2pms"

    def cut_out_section(self, ac_name: str, route: list[RoutePoint]) -> list[RoutePoint]:
        if self.canonical_point_merge:
            names = [wp.waypoint_name for wp in route]
            if MERGE_POINT not in names:
                raise ValueError(
                    f"{ac_name}: TADA_ACTION_SET=v2pms is the point-merge clearance set, but this route "
                    f"has no merge point {MERGE_POINT} (a trombone scenario? use TADA_ACTION_SET=v2)")
            entry = next((n for n in names if n in POINT_MERGE_ARC), None)
            assert entry is not None, f"{ac_name}: point-merge route without a sequencing-arc point: {names}"
            self._route_parts[ac_name] = PointMergeLeg(entry)
            return list(route)
        remaining_part, excluded_part, before_exclusion_point, after_exclusion_point = (
            cut_out_section_from_route(route)
        )
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
        tr = self._route_parts[ac_name]
        return tr.insert_section_to_route(n_levels, route)

    def remove_section_from_route(
        self, ac_name: str, route: list[RoutePoint], n_levels: int
    ) -> list[RoutePoint]:
        tr = self._route_parts[ac_name]
        return tr.remove_section_from_route(n_levels, route)

    def num_levels_can_add_to_route(self, ac_name: str, route: list[RoutePoint]) -> int:
        return self._route_parts[ac_name].num_levels_can_add_to_route(route)

    def num_levels_can_remove_from_route(self, ac_name: str, route: list[RoutePoint]) -> int:
        return self._route_parts[ac_name].num_levels_can_remove_from_route(route)

    def is_before_current_section_cut_point(self, ac_name: str, route: list[RoutePoint]) -> bool:
        return self._route_parts[ac_name].is_before_current_section_cut_point(route)

    def can_direct_to_merge(self, ac_name: str, route: list[RoutePoint]) -> bool:
        part = self._route_parts.get(ac_name)
        return isinstance(part, PointMergeLeg) and part.can_direct_to_merge(route)

    def direct_to_merge_route(self, ac_name: str, route: list[RoutePoint]) -> list[RoutePoint]:
        part = self._route_parts[ac_name]
        assert isinstance(part, PointMergeLeg)
        return part.direct_to_merge_route(route)

    def is_before_upwind_start(self, ac_name: str, route: list[RoutePoint]) -> bool:
        return self._route_parts[ac_name].is_before_upwind_start(route)

    def has_reached_section(self, ac_name: str, route: list[RoutePoint]) -> bool:
        return self._route_parts[ac_name].has_reached_section(route)

    def clone(self) -> "RouteSectionStore":
        """Independently mutable copy, for read-only look-ahead over clearances that would
        otherwise alter live episode state. See SectionInterface.clone on why not deepcopy."""
        new = RouteSectionStore()
        new._route_parts = {k: v.clone() for k, v in self._route_parts.items()}
        return new
