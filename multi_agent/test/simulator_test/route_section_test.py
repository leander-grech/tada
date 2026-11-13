"""Extracting levels from one-way and back-and-forth route sections."""

import pytest
from flight_simulator import RoutePoint
from numpy.testing import assert_equal

from simulator.route_section_store import BackAndForthSection, OneWaySection


def point(name: str) -> RoutePoint:
    return RoutePoint.from_waypoint_name(name, 0, 0)


def names(points: list[RoutePoint]) -> list[str]:
    return [p.waypoint_name for p in points]


@pytest.mark.parametrize(
    "section_cls,route_names,message",
    [
        (OneWaySection, "ABC", "exceeds available forward levels"),
        (BackAndForthSection, "ABCDEF", "exceeds available section levels"),
    ],
)
def test_extracting_more_levels_than_available_is_rejected(section_cls, route_names, message):
    section = section_cls([point(n) for n in route_names])
    with pytest.raises(ValueError, match=message):
        section.extract_part(4)


def test_one_way_extract_returns_forward_part_only():
    section = OneWaySection([point(n) for n in "ABC"])
    forward, backward = section.extract_part(2)
    assert_equal(section.num_levels(), 1)
    assert_equal(names(forward), ["A", "B"])
    assert_equal(names(backward), [])


def test_back_and_forth_extract_pairs_forward_with_return_leg_and_resumes():
    section = BackAndForthSection([point(n) for n in "ABCDEF"])
    forward, backward = section.extract_part(2)
    assert_equal(names(forward), ["A", "B"])
    assert_equal(names(backward), ["E", "F"])
    forward, backward = section.extract_part(1)
    assert_equal(names(forward), ["C"])
    assert_equal(names(backward), ["D"])


def test_back_and_forth_requires_even_point_count():
    with pytest.raises(ValueError, match="requires an even number of points"):
        BackAndForthSection([point(n) for n in "ABC"])
