"""Tests for route section retraction (SHORTEN_TROMBONE support)."""

import pytest
from flight_simulator import RoutePoint
from numpy.testing import assert_equal

from simulator.route_section_store import (
    RouteSectionStore,
)


def _pt(name: str) -> RoutePoint:
    return RoutePoint.from_waypoint_name(name, 0, 0)


def _names(route: list[RoutePoint]) -> list[str]:
    return [wp.waypoint_name for wp in route]


# Shared waypoints
p1, p2 = _pt("X"), _pt("Y")
s = _pt("MC761")  # section start marker (back-and-forth)
t1, t2, t3, t4, t5, t6 = _pt("A"), _pt("B"), _pt("C"), _pt("D"), _pt("E"), _pt("F")
e = _pt("INLER")  # section end marker
p3, p4 = _pt("K"), _pt("L")


@pytest.fixture()
def store_with_route() -> tuple[RouteSectionStore, str, list[RoutePoint]]:
    """Build a RouteSectionStore from a full route with a back-and-forth section."""
    ac_name = "AC1"
    full_route: list[RoutePoint] = [p1, p2, s, t1, t2, t3, t4, e, p3, p4]
    store = RouteSectionStore()
    trimmed_route = store.cut_out_section(ac_name, full_route)
    return store, ac_name, trimmed_route


def test_retract_after_single_level_add_restores_original_length(
    store_with_route: tuple[RouteSectionStore, str, list[RoutePoint]],
) -> None:
    store, ac_name, original_route = store_with_route
    original_len = len(original_route)

    extended = store.add_section_to_route(ac_name, original_route, n_levels=1)
    assert len(extended) == original_len + 2  # one forward + one backward added

    retracted = store.retract_section_from_route(ac_name, extended)
    assert retracted is not None
    assert len(retracted) == original_len


def test_retract_without_prior_add_returns_none(
    store_with_route: tuple[RouteSectionStore, str, list[RoutePoint]],
) -> None:
    store, ac_name, original_route = store_with_route
    result = store.retract_section_from_route(ac_name, original_route)
    assert result is None


def test_retract_unknown_aircraft_returns_none() -> None:
    store = RouteSectionStore()
    result = store.retract_section_from_route("UNKNOWN", [_pt("Z")])
    assert result is None


def test_num_retractable_levels_tracks_additions(
    store_with_route: tuple[RouteSectionStore, str, list[RoutePoint]],
) -> None:
    store, ac_name, original_route = store_with_route

    assert_equal(store.num_levels_can_retract(ac_name), 0)

    extended_1 = store.add_section_to_route(ac_name, original_route, n_levels=1)
    assert_equal(store.num_levels_can_retract(ac_name), 1)

    store.add_section_to_route(ac_name, extended_1, n_levels=1)
    assert_equal(store.num_levels_can_retract(ac_name), 2)


def test_retract_decrements_retractable_count(
    store_with_route: tuple[RouteSectionStore, str, list[RoutePoint]],
) -> None:
    store, ac_name, original_route = store_with_route

    extended = store.add_section_to_route(ac_name, original_route, n_levels=1)
    assert_equal(store.num_levels_can_retract(ac_name), 1)

    store.retract_section_from_route(ac_name, extended)
    assert_equal(store.num_levels_can_retract(ac_name), 0)


def test_multi_level_add_creates_per_level_entries() -> None:
    """Adding 3 levels at once should create 3 retractable entries."""
    ac_name = "AC2"
    full_route: list[RoutePoint] = [p1, p2, s, t1, t2, t3, t4, t5, t6, e, p3, p4]
    store = RouteSectionStore()
    trimmed = store.cut_out_section(ac_name, full_route)

    store.add_section_to_route(ac_name, trimmed, n_levels=3)
    assert_equal(store.num_levels_can_retract(ac_name), 3)


def test_retract_removes_correct_waypoints_lifo_order(
    store_with_route: tuple[RouteSectionStore, str, list[RoutePoint]],
) -> None:
    """Retraction should remove the most recently added level (LIFO)."""
    store, ac_name, original_route = store_with_route
    # t1,t2 forward / t3,t4 backward — add 1 level inserts t1 (fwd) + t4 (bkwd)
    extended_1 = store.add_section_to_route(ac_name, original_route, n_levels=1)
    # add second level inserts t2 (fwd) + t3 (bkwd)
    extended_2 = store.add_section_to_route(ac_name, extended_1, n_levels=1)

    # First retract removes the second level (t2 + t3)
    after_first_retract = store.retract_section_from_route(ac_name, extended_2)
    assert after_first_retract is not None
    assert "B" not in _names(after_first_retract)
    assert "C" not in _names(after_first_retract)
    assert "A" in _names(after_first_retract)
    assert "D" in _names(after_first_retract)

    # Second retract removes the first level (t1 + t4)
    after_second_retract = store.retract_section_from_route(ac_name, after_first_retract)
    assert after_second_retract is not None
    assert "A" not in _names(after_second_retract)
    assert "D" not in _names(after_second_retract)


def test_retract_restores_last_removed_trombone_level_to_addable_section(
    store_with_route: tuple[RouteSectionStore, str, list[RoutePoint]],
) -> None:
    store, ac_name, original_route = store_with_route

    fully_extended = store.add_section_to_route(ac_name, original_route, n_levels=2)
    retracted = store.retract_section_from_route(ac_name, fully_extended)

    assert retracted is not None
    assert _names(retracted) == ["X", "Y", "MC761", "A", "D", "INLER", "K", "L"]
    assert store.num_levels_can_add_to_route(ac_name, retracted) == 1
    assert store.is_before_current_section_cut_point(ac_name, retracted) is True
    assert store.is_before_upwind_start(ac_name, retracted) is True

    restored = store.add_section_to_route(ac_name, retracted, n_levels=1)
    assert _names(restored) == _names(fully_extended)


def test_num_levels_can_retract_unknown_aircraft() -> None:
    store = RouteSectionStore()
    assert_equal(store.num_levels_can_retract("UNKNOWN"), 0)
