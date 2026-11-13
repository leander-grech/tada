"""Inserting back-and-forth and one-way route sections into routes."""

from flight_simulator import RoutePoint
from numpy.testing import assert_equal

from simulator.route_section_store import BackAndForthSection, OneWaySection, RouteSectionExtender


def point(name: str) -> RoutePoint:
    return RoutePoint.from_waypoint_name(name, 0, 0)


p1 = point("X")
p2 = point("Y")

s = point("U")
t1 = point("A")
t2 = point("B")
t3 = point("C")
t4 = point("D")
e = point("W")

p3 = point("K")
p4 = point("L")


def test_insert_part_of_back_and_forth_section_to_route() -> None:
    excluded_trombone: list[RoutePoint] = [t1, t2, t3, t4]
    extender = RouteSectionExtender(
        BackAndForthSection(excluded_trombone), s.waypoint_name, e.waypoint_name
    )

    route: list[RoutePoint] = [p1, p2, s, e, p3, p4]
    new_route = extender.insert_section_to_route(1, route)
    assert_equal(new_route, [p1, p2, s, t1, t4, e, p3, p4])


def test_insert_all_of_back_and_forth_section_to_route() -> None:
    excluded_trombone: list[RoutePoint] = [t1, t2, t3, t4]
    extender = RouteSectionExtender(
        BackAndForthSection(excluded_trombone), s.waypoint_name, e.waypoint_name
    )

    route: list[RoutePoint] = [p1, p2, s, e, p3, p4]
    new_route = extender.insert_section_to_route(2, route)
    assert_equal(new_route, [p1, p2, s, t1, t2, t3, t4, e, p3, p4])


def test_remaining_levels_when_not_yet_at_cut_point() -> None:
    excluded_trombone: list[RoutePoint] = [t1, t2, t3, t4]
    extender = RouteSectionExtender(
        BackAndForthSection(excluded_trombone), s.waypoint_name, e.waypoint_name
    )

    route: list[RoutePoint] = [p1, p2, s, e, p3, p4]
    assert_equal(extender.num_levels_can_add_to_route(route), 2)
    extended_route = extender.insert_section_to_route(2, route)
    assert_equal(extender.num_levels_can_add_to_route(extended_route), 0)


def test_has_remaining_levels_when_reached_original_cut_point_but_not_new() -> None:
    excluded_trombone: list[RoutePoint] = [t1, t2, t3, t4]
    extender = RouteSectionExtender(
        BackAndForthSection(excluded_trombone), s.waypoint_name, e.waypoint_name
    )

    route: list[RoutePoint] = [p1, p2, s, e, p3]
    # Add "A" and "D" to route
    extended_route = extender.insert_section_to_route(1, route)
    # Complete new route until and including original cut point "U" but not new cut point "A"
    part_of_extended_route = extended_route[3:]
    assert_equal(extender.num_levels_can_add_to_route(part_of_extended_route), 1)


def test_remaining_levels_0_when_reached_cut_point() -> None:
    excluded_trombone: list[RoutePoint] = [t1, t2, t3, t4]
    extender = RouteSectionExtender(
        BackAndForthSection(excluded_trombone), s.waypoint_name, e.waypoint_name
    )

    route: list[RoutePoint] = [p1, p2, s, e, p3]
    # Add "A" and "D" to route
    extended_route = extender.insert_section_to_route(1, route)
    # Complete new route until and including new cut point "A" => cut point passed
    part_of_extended_route = extended_route[4:]
    assert_equal(extender.num_levels_can_add_to_route(part_of_extended_route), 0)


def test_is_not_on_trombone_when_not_reached_original_cut_point():
    extender = RouteSectionExtender(BackAndForthSection([t1, t4]), s.waypoint_name, e.waypoint_name)
    route: list[RoutePoint] = [
        p1,
        p2,
        s,
        e,
        p3,
    ]
    assert extender.has_reached_section(route) is False
    assert extender.is_before_current_section_cut_point(route) is True
    assert extender.is_before_upwind_start(route) is True


def test_is_not_on_trombone_when_extended_route_not_reached_original_cut_point() -> None:
    extender = RouteSectionExtender(
        BackAndForthSection([t1, t2, t3, t4]), s.waypoint_name, e.waypoint_name
    )
    route: list[RoutePoint] = [p1, p2, s, e, p3]
    # Add t1 + t4 to route
    extended_route = extender.insert_section_to_route(1, route)
    # Extended route has reached point just before original cut point s
    extended_route_part = extended_route[2:]
    assert extender.has_reached_section(extended_route_part) is False
    assert extender.is_before_current_section_cut_point(extended_route_part) is True
    assert extender.is_before_upwind_start(extended_route_part) is True


def test_is_on_upwind_part_when_extended_route_not_reached_new_cut_point() -> None:
    extender = RouteSectionExtender(
        BackAndForthSection([t1, t2, t3, t4]), s.waypoint_name, e.waypoint_name
    )
    route: list[RoutePoint] = [p1, p2, s, e, p3]
    extended_route = extender.insert_section_to_route(1, route)
    # Extended route has completed original cut point s but not new cut point t1
    extended_route_part = extended_route[3:]
    assert extender.has_reached_section(extended_route_part) is True
    assert extender.is_before_current_section_cut_point(extended_route_part) is True
    assert extender.is_before_upwind_start(extended_route_part) is True


def test_is_between_downwind_and_upwind_when_not_reached_new_cutback_point() -> None:
    extender = RouteSectionExtender(
        BackAndForthSection([t1, t2, t3, t4]), s.waypoint_name, e.waypoint_name
    )
    route: list[RoutePoint] = [p1, p2, s, e, p3]
    # Add t1 + t4 to route
    extended_route = extender.insert_section_to_route(1, route)
    # Extended route has has new cut-back-point t4 remaining (on trombone end edge)
    extended_route_part = extended_route[4:]
    assert extender.has_reached_section(extended_route_part) is True
    assert extender.is_before_current_section_cut_point(extended_route_part) is False
    assert extender.is_before_upwind_start(extended_route_part) is True


def test_is_on_upwind_part_when_extended_route_reached_new_cutback_point() -> None:
    extender = RouteSectionExtender(
        BackAndForthSection([t1, t2, t3, t4]), s.waypoint_name, e.waypoint_name
    )
    route: list[RoutePoint] = [p1, p2, s, e, p3]
    # Add t1 + t4 to route
    extended_route = extender.insert_section_to_route(1, route)
    # Extended route has has cut-back-point t4 remaining (on trombone end edge)
    extended_route_part = extended_route[5:]
    assert extender.has_reached_section(extended_route_part) is True
    assert extender.is_before_current_section_cut_point(extended_route_part) is False
    assert extender.is_before_upwind_start(extended_route_part) is False


ow_x = point("X")
ow_y = point("Y")

ow_a = point("A")
ow_b = point("B")
ow_c = point("C")

ow_k = point("K")


def test_one_way_insert_part_of_section_to_route():
    excluded_route: list[RoutePoint] = [ow_a, ow_b, ow_c]
    excluded = OneWaySection(excluded_route)
    extender = RouteSectionExtender(excluded, ow_y.waypoint_name, ow_k.waypoint_name)

    route: list[RoutePoint] = [ow_x, ow_y, ow_k]
    new_route = extender.insert_section_to_route(2, route)
    assert_equal(new_route, [ow_x, ow_y, ow_a, ow_b, ow_k])


def test_one_way_remaining_levels_when_not_yet_at_cut_point():
    excluded_route: list[RoutePoint] = [ow_a, ow_b, ow_c]
    excluded = OneWaySection(excluded_route)
    extender = RouteSectionExtender(excluded, ow_y.waypoint_name, ow_k.waypoint_name)

    route: list[RoutePoint] = [ow_x, ow_y, ow_k]
    assert_equal(extender.num_levels_can_add_to_route(route), 3)
    extended_route = extender.insert_section_to_route(1, route)
    assert_equal(extender.num_levels_can_add_to_route(extended_route), 2)


def test_one_way_remaining_levels_0_when_reached_cut_point():
    excluded_route: list[RoutePoint] = [ow_a, ow_b, ow_c]
    excluded = OneWaySection(excluded_route)
    extender = RouteSectionExtender(excluded, ow_y.waypoint_name, ow_k.waypoint_name)

    route: list[RoutePoint] = [ow_x, ow_y, ow_k]
    # Add "A" and "B" to route
    extended_route = extender.insert_section_to_route(2, route)
    # Complete new route until and including new cut point "B" => cut point passed
    part_of_extended_route = extended_route[4:]
    assert_equal(extender.num_levels_can_add_to_route(part_of_extended_route), 0)
