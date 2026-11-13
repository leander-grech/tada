"""Tests for trombone cut functionality."""

from flight_simulator import RoutePoint
from numpy.testing import assert_equal

from simulator.route_section_store import cut_out_section_from_route


def name_to_point(name: str):
    return RoutePoint.from_waypoint_name(name, 0, 0)


p1 = name_to_point("X")
p2 = name_to_point("Y")

s_a = name_to_point("MC761")
s_b = name_to_point("MC561")
t1 = name_to_point("A")
t2 = name_to_point("B")
t3 = name_to_point("C")
t4 = name_to_point("D")
e = name_to_point("INLER")

p3 = name_to_point("K")
p4 = name_to_point("L")


def test_cut_trombone_from_route_through_mc761():
    route = [p1, p2, s_a, t1, t2, t3, t4, e, p3]
    result = cut_out_section_from_route(route)
    assert result is not None
    remaining_part, excluded_part, before_exclusion_point, after_exclusion_point = result
    assert_equal(remaining_part, [p1, p2, s_a, e, p3])
    assert_equal(excluded_part, [t1, t2, t3, t4])
    assert_equal(before_exclusion_point, s_a)
    assert_equal(after_exclusion_point, e)


def test_cut_trombone_from_route_through_mc561():
    route = [p2, s_b, t1, t2, t3, t4, e, p3, p4]
    result = cut_out_section_from_route(route)
    assert result is not None
    remaining_part, excluded_part, before_exclusion_point, after_exclusion_point = result
    assert_equal(before_exclusion_point, s_b)
    assert_equal(after_exclusion_point, e)
