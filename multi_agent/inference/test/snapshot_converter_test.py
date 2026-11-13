"""Tests for SnapshotConverter."""

# pylint: disable=redefined-outer-name

import json
from pathlib import Path
from typing import Any

import pytest

from inference.server_models import (
    BGY_PREFERRED_ALTITUDE_NUM_LEVELS,
    Airport,
    SerializedAircraft,
    SnapshotAircraftState,
    SnapshotRouteTracker,
    SnapshotTargetState,
    SnapshotWaypoint,
)
from inference.snapshot_converter import (
    InvalidPreferredAltitudeError,
    SnapshotConverter,
)

TEST_DIR = Path(__file__).parent


@pytest.fixture()
def converted():
    snapshot = json.loads((TEST_DIR / "test_snapshot.json").read_text())
    expected = json.loads((TEST_DIR / "test_world.json").read_text())
    converter = SnapshotConverter()
    result_json, callsign_map = converter.convert(snapshot)
    result = json.loads(result_json)
    return result, expected, callsign_map


def test_convert_aircraft_count(converted):
    result, expected, _ = converted
    assert len(result["aircraft"]) == 2
    assert len(result["aircraft"]) == len(expected["aircraft"])


def test_convert_aircraft_fields(converted):
    result, expected, _ = converted
    result_by_cs = {ac["call_sign"]: ac for ac in result["aircraft"]}
    expected_by_cs = {ac["call_sign"]: ac for ac in expected["aircraft"]}

    for cs in expected_by_cs:
        r = result_by_cs[cs]
        e = expected_by_cs[cs]

        assert r["call_sign"] == e["call_sign"]
        assert r["start_time_s"] == pytest.approx(e["start_time_s"], rel=1e-4)
        assert r["runway_id"] == e["runway_id"]
        assert r["aircraft_type"] == e["aircraft_type"]
        assert r["following_route"] == e["following_route"]
        assert r["target_state"].keys() == e["target_state"].keys()
        for field, value in e["target_state"].items():
            if value is None:
                assert r["target_state"][field] is None
            else:
                assert r["target_state"][field] == pytest.approx(value, rel=1e-4)
        assert r["mission"] == pytest.approx(e["mission"])
        assert r["final_fix_m"] == pytest.approx(e["final_fix_m"], rel=1e-4)


def test_convert_arrival_targets(converted):
    result, expected, _ = converted
    result_arrivals = {(at["ac_callsign"], at["eta_s"]) for at in result["arrival_targets"]}
    expected_arrivals = {(at["ac_callsign"], at["eta_s"]) for at in expected["arrival_targets"]}
    assert result_arrivals == expected_arrivals


def test_callsign_mapping(converted):
    _, _, callsign_map = converted
    assert len(callsign_map) == 2
    for remapped in callsign_map.values():
        assert remapped.startswith("TEST")
        assert remapped[4:].isdigit()
        assert len(remapped[4:]) == 3


def test_convert_aircraft_routes(converted):
    result, expected, _ = converted
    result_by_cs = {ac["call_sign"]: ac for ac in result["aircraft"]}
    expected_by_cs = {ac["call_sign"]: ac for ac in expected["aircraft"]}

    for cs in expected_by_cs:
        r_route = result_by_cs[cs]["route"]
        e_route = expected_by_cs[cs]["route"]

        # Waypoint names match
        r_names = [wp["waypoint_name"] for wp in r_route]
        e_names = [wp["waypoint_name"] for wp in e_route]
        assert r_names == e_names, f"{cs}: waypoint names differ"

        for r_wp, e_wp in zip(r_route, e_route):
            name = e_wp["waypoint_name"]
            # altitude constraints
            if e_wp["min_altitude"] is not None:
                assert r_wp["min_altitude"] == pytest.approx(e_wp["min_altitude"], rel=1e-6), (
                    f"{cs}/{name}: min_altitude mismatch"
                )
            else:
                assert r_wp["min_altitude"] is None, f"{cs}/{name}: expected min_altitude=None"

            if e_wp["max_altitude"] is not None:
                assert r_wp["max_altitude"] == pytest.approx(e_wp["max_altitude"], rel=1e-6), (
                    f"{cs}/{name}: max_altitude mismatch"
                )
            else:
                assert r_wp["max_altitude"] is None, f"{cs}/{name}: expected max_altitude=None"

            # speed
            if e_wp["target_speed"] is not None:
                assert r_wp["target_speed"] == pytest.approx(e_wp["target_speed"], rel=1e-6), (
                    f"{cs}/{name}: target_speed mismatch"
                )
            else:
                assert r_wp["target_speed"] is None, f"{cs}/{name}: expected target_speed=None"


def test_finished_aircraft_excluded():
    snapshot = [
        {
            "callSign": "FINISHED1",
            "manuallyUpdatedSpeed": False,
            "state": {
                "position": [10.0, 20.0],
                "heading": 90.0,
                "speed": 250,
                "altitude": 20000,
                "status": 3,
                "controlMode": "route",
                "controlStatus": 2,
            },
            "targetState": {"heading": 90.0, "speed": 250, "altitude": 20000},
            "routeTracker": {
                "currentIndex": 0,
                "route": ["OSBUL", "BGY"],
            },
            "amanArrivalTime": 500.0,
            "amanOffsetSeconds": 0.0,
            "secondsToStart": 0,
        }
    ]
    converter = SnapshotConverter()
    result_json, callsign_map = converter.convert(snapshot)
    result = json.loads(result_json)

    assert len(result["aircraft"]) == 0
    assert len(callsign_map) == 0


def _make_aircraft(
    *,
    callsign: str,
    route: list[SnapshotWaypoint],
    current_index: int = 0,
) -> dict[str, Any]:
    """Build a valid SerializedAircraft dict via Pydantic validation."""
    return SerializedAircraft(
        callSign=callsign,
        secondsToStart=0.0,
        manuallyUpdatedSpeed=False,
        identified=True,
        state=SnapshotAircraftState(
            position=[10.0, 20.0],
            heading=90.0,
            speed=250.0,
            altitude=20000.0,
            turnDirection=0,
            status=1,
            controlMode="route",
            controlStatus=2,
            cautionState="none",
        ),
        targetState=SnapshotTargetState(heading=90.0, speed=250.0, altitude=20000.0),
        routeTracker=SnapshotRouteTracker(
            currentIndex=current_index,
            route=route,
        ),
        amanArrivalTime=500.0,
        amanOffsetSeconds=0.0,
    ).model_dump()


def test_detect_airport_from_approach_waypoint_name():
    converter = SnapshotConverter()
    snapshot = [
        _make_aircraft(
            callsign="AC1",
            route=[
                SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="MXP_35L_APP01", location=[11.0, 21.0], speed=180.0),
            ],
        )
    ]

    result_json, _ = converter.convert(snapshot)
    result = json.loads(result_json)

    assert len(result["aircraft"]) == 1
    assert result["aircraft"][0]["runway_id"] == "35L"


def test_clamp_route_index_when_snapshot_index_is_past_end():
    converter = SnapshotConverter()
    snapshot = [
        _make_aircraft(
            callsign="AC1",
            route=[
                SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="BGY", location=[11.0, 21.0], speed=180.0),
            ],
            current_index=999,
        )
    ]

    result_json, _ = converter.convert(snapshot)
    result = json.loads(result_json)

    assert len(result["aircraft"]) == 1
    assert result["aircraft"][0]["runway_id"] == "10"
    assert [wp["waypoint_name"] for wp in result["aircraft"][0]["route"]] == ["BGY"]


def test_duplicate_callsigns_are_skipped():
    converter = SnapshotConverter()
    shared_callsign = "DUP1"
    snapshot = [
        _make_aircraft(
            callsign=shared_callsign,
            route=[
                SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="BGY", location=[11.0, 21.0], speed=180.0),
            ],
        ),
        _make_aircraft(
            callsign=shared_callsign,
            route=[
                SnapshotWaypoint(name="RUVUK", location=[12.0, 22.0], speed=220.0),
                SnapshotWaypoint(name="MXP", location=[13.0, 23.0], speed=170.0),
            ],
        ),
    ]

    result_json, callsign_map = converter.convert(snapshot)
    result = json.loads(result_json)

    assert len(result["aircraft"]) == 1
    assert list(callsign_map.keys()) == [shared_callsign]


def test_convert_raises_if_all_non_finished_aircraft_are_invalid():
    converter = SnapshotConverter()
    snapshot = [
        _make_aircraft(
            callsign="AC1",
            route=[
                SnapshotWaypoint(name="UNKNOWN_1", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="UNKNOWN_2", location=[11.0, 21.0], speed=180.0),
            ],
        )
    ]

    with pytest.raises(ValueError, match="produced no valid aircraft"):
        converter.convert(snapshot)


def test_convert_split_mixed_airports():
    converter = SnapshotConverter()
    snapshot = [
        _make_aircraft(
            callsign="MXP1",
            route=[
                SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="MXP", location=[11.0, 21.0], speed=180.0),
            ],
        ),
        _make_aircraft(
            callsign="BGY1",
            route=[
                SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="BGY", location=[11.0, 21.0], speed=180.0),
            ],
        ),
        _make_aircraft(
            callsign="MXP2",
            route=[
                SnapshotWaypoint(name="RUVUK", location=[12.0, 22.0], speed=220.0),
                SnapshotWaypoint(name="MXP_35L_APP01", location=[13.0, 23.0], speed=170.0),
            ],
        ),
    ]
    result = converter.convert_split(snapshot)

    assert set(result.keys()) == {Airport.MXP, Airport.BGY}

    mxp_world = json.loads(result[Airport.MXP][0])
    bgy_world = json.loads(result[Airport.BGY][0])
    assert len(mxp_world["aircraft"]) == 2
    assert len(bgy_world["aircraft"]) == 1

    # Each airport group has independent callsign namespace
    mxp_callsigns = set(result[Airport.MXP][1].values())
    bgy_callsigns = set(result[Airport.BGY][1].values())
    assert "TEST001" in mxp_callsigns
    assert "TEST001" in bgy_callsigns


def test_convert_split_empty_snapshot():
    converter = SnapshotConverter()
    result = converter.convert_split([])
    assert result == {}


def test_convert_split_finished_aircraft_excluded():
    converter = SnapshotConverter()
    snapshot = [
        _make_aircraft(
            callsign="AC1",
            route=[
                SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="MXP", location=[11.0, 21.0], speed=180.0),
            ],
        ),
    ]
    # Mark as finished
    snapshot[0]["state"]["status"] = 3

    result = converter.convert_split(snapshot)
    assert result == {}


def _bgy_aircraft_with_preferred_level(level: int) -> dict[str, Any]:
    ac = _make_aircraft(
        callsign="BGY1",
        route=[
            SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
            SnapshotWaypoint(name="BGY", location=[11.0, 21.0], speed=180.0),
        ],
    )
    ac["preferredAltitudeLevel"] = level
    return ac


def test_preferred_altitude_defaults_to_hardcoded_num_levels():
    converter = SnapshotConverter()
    result_json, _ = converter.convert([_bgy_aircraft_with_preferred_level(1)])
    result = json.loads(result_json)
    assert result["aircraft"][0]["preferred_altitude_fraction"] == {
        "preferred_level": 1,
        "num_levels": BGY_PREFERRED_ALTITUDE_NUM_LEVELS,
    }


def test_preferred_altitude_uses_override_num_levels():
    converter = SnapshotConverter()
    result_json, _ = converter.convert(
        [_bgy_aircraft_with_preferred_level(5)], preferred_altitude_num_levels=6
    )
    result = json.loads(result_json)
    assert result["aircraft"][0]["preferred_altitude_fraction"] == {
        "preferred_level": 5,
        "num_levels": 6,
    }


def test_preferred_altitude_raises_when_level_out_of_range():
    converter = SnapshotConverter()
    with pytest.raises(InvalidPreferredAltitudeError, match="out of range"):
        converter.convert([_bgy_aircraft_with_preferred_level(4)])


def test_preferred_altitude_raises_with_override_when_level_out_of_range():
    converter = SnapshotConverter()
    with pytest.raises(InvalidPreferredAltitudeError, match="out of range"):
        converter.convert([_bgy_aircraft_with_preferred_level(2)], preferred_altitude_num_levels=2)


def test_convert_split_passes_preferred_altitude_num_levels():
    converter = SnapshotConverter()
    result = converter.convert_split(
        [_bgy_aircraft_with_preferred_level(4)], preferred_altitude_num_levels=5
    )
    bgy_world = json.loads(result[Airport.BGY][0])
    assert bgy_world["aircraft"][0]["preferred_altitude_fraction"] == {
        "preferred_level": 4,
        "num_levels": 5,
    }


def test_convert_split_undetectable_airport_skipped():
    converter = SnapshotConverter()
    snapshot = [
        _make_aircraft(
            callsign="AC_UNKNOWN",
            route=[
                SnapshotWaypoint(name="UNKNOWN_1", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="UNKNOWN_2", location=[11.0, 21.0], speed=180.0),
            ],
        ),
        _make_aircraft(
            callsign="AC_MXP",
            route=[
                SnapshotWaypoint(name="OSBUL", location=[10.0, 20.0], speed=230.0),
                SnapshotWaypoint(name="MXP", location=[11.0, 21.0], speed=180.0),
            ],
        ),
    ]
    result = converter.convert_split(snapshot)

    # Only MXP group, unknown is skipped
    assert list(result.keys()) == [Airport.MXP]
    assert len(result[Airport.MXP][1]) == 1
    assert "AC_MXP" in result[Airport.MXP][1]
