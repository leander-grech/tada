"""Unit tests for suggestion_converter."""

from actions.actions import Action
from geometry.vectoring import VIRTUAL_TURN_WAYPOINT_PREFIX
from inference.suggestion_converter import (
    _BGY_CONFIG,
    _MXP_CONFIG,
    MXP_TROMBONE_END,
    _convert_heading,
    _convert_return_to_route,
    _convert_skip,
    _convert_speed,
    _convert_trombone,
    _convert_vector_to_ils,
    _time_to_execute,
    convert_actions_to_suggestions,
)
from simulator.route_section_store import (
    BGY_SECTION_START_MARKERS,
    MXP_SECTION_START_MARKERS,
)

_DEFAULT_CALL_SIGN = "SAS123"
_EVENT_HEADING = "aircraft-heading"
_EVENT_SPEED = "aircraft-speed"
_EVENT_WAYPOINT = "aircraft-waypoint"
_EVENT_DCT_FROM_TO = "aircraft-dct-from-to"

_MXP_START = MXP_SECTION_START_MARKERS[1]
_MXP_END = _MXP_CONFIG.section_end
_BGY_START_1 = BGY_SECTION_START_MARKERS[0]
_BGY_START_2 = BGY_SECTION_START_MARKERS[1]
_BGY_END = _BGY_CONFIG.section_end


def _vec_turn_wp(n: int) -> str:
    return f"{VIRTUAL_TURN_WAYPOINT_PREFIX}{n}"


def _make_record(
    action: int,
    *,
    call_sign: str = _DEFAULT_CALL_SIGN,
    heading_after: float | None = None,
    speed_after: float | None = None,
    decision_time_s: float = 100.0,
    route_before: list[str] | None = None,
    route_after: list[str] | None = None,
) -> dict:
    record: dict = {
        "action": action,
        "call_sign": call_sign,
        "decision_time_s": decision_time_s,
    }
    if heading_after is not None:
        record["heading_after"] = heading_after
    if speed_after is not None:
        record["speed_after"] = speed_after
    if route_before is not None:
        record["route_before"] = route_before
    if route_after is not None:
        record["route_after"] = route_after
    return record


class TestConvertHeading:
    """Tests for _convert_heading (heading-only, no return-to-route)."""

    def test_turn_with_valid_route_after_returns_heading_only(self):
        record = _make_record(
            Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
            heading_after=130.0,
            route_after=[_vec_turn_wp(1), "ARFIS", _MXP_END],
            decision_time_s=100.0,
        )
        result = _convert_heading(record)
        assert result is not None
        assert result["eventType"] == _EVENT_HEADING

    def test_turn_without_route_after_returns_heading_only(self):
        record = _make_record(
            Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
            heading_after=130.0,
        )
        result = _convert_heading(record)
        assert result is not None
        assert result["eventType"] == _EVENT_HEADING

    def test_turn_with_empty_route_after_returns_heading_only(self):
        record = _make_record(
            Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
            heading_after=130.0,
            route_after=[],
        )
        result = _convert_heading(record)
        assert result is not None
        assert result["eventType"] == _EVENT_HEADING

    def test_turn_with_all_vec_turn_waypoints_returns_heading_only(self):
        record = _make_record(
            Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
            heading_after=130.0,
            route_after=[_vec_turn_wp(1), _vec_turn_wp(2)],
        )
        result = _convert_heading(record)
        assert result is not None
        assert result["eventType"] == _EVENT_HEADING

    def test_route_after_does_not_affect_heading_result(self):
        """Heading converter emits a heading event regardless of the post-turn route."""
        record = _make_record(
            Action.TURN_RIGHT_AND_RETURN_TO_ROUTE,
            heading_after=50.0,
            route_after=[_vec_turn_wp(1), _vec_turn_wp(2), "REAL_WP", "NEXT_WP"],
        )
        result = _convert_heading(record)
        assert result is not None
        assert result["eventType"] == _EVENT_HEADING

    def test_left_turn_heading_value(self):
        """Left turn: heading_after=130 (math) → aviation (-130+90)%360 = 320."""
        record = _make_record(
            Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
            heading_after=130.0,
        )
        result = _convert_heading(record)
        assert result is not None
        _, heading_val, turn_dir = result["eventArgs"]
        assert turn_dir == 0  # shortest
        assert heading_val == 320

    def test_right_turn_heading_value(self):
        """Right turn: heading_after=50 (math) → aviation (-50+90)%360 = 40."""
        record = _make_record(
            Action.TURN_RIGHT_AND_RETURN_TO_ROUTE,
            heading_after=50.0,
        )
        result = _convert_heading(record)
        assert result is not None
        _, heading_val, turn_dir = result["eventArgs"]
        assert turn_dir == 0  # shortest
        assert heading_val == 40

    def test_left_turn_wraps_around_360(self):
        """Left turn: heading_after=30 (math) → aviation (-30+90)%360 = 60."""
        record = _make_record(
            Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
            heading_after=30.0,
        )
        result = _convert_heading(record)
        assert result is not None
        _, heading_val, turn_dir = result["eventArgs"]
        assert turn_dir == 0  # shortest
        assert heading_val == 60

    def test_right_turn_wraps_around_zero(self):
        """Right turn: heading_after=330 (math) → aviation (-330+90)%360 = 120."""
        record = _make_record(
            Action.TURN_RIGHT_AND_RETURN_TO_ROUTE,
            heading_after=330.0,
        )
        result = _convert_heading(record)
        assert result is not None
        _, heading_val, turn_dir = result["eventArgs"]
        assert turn_dir == 0  # shortest
        assert heading_val == 120

    def test_all_turn_actions_use_shortest_direction(self):
        """Every turn action variant outputs turn direction 0 (shortest)."""
        for action in [
            Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
            Action.TURN_RIGHT_AND_RETURN_TO_ROUTE,
        ]:
            result = _convert_heading(_make_record(action, heading_after=90.0))
            assert result is not None, f"Failed for {action}"
            assert result["eventArgs"][2] == 0, f"Expected turn_dir=0 for {action}"

    def test_non_turn_action_returns_none(self):
        record = _make_record(Action.DO_NOTHING)
        result = _convert_heading(record)
        assert result is None


class TestConvertActionsToSuggestionsWithHeading:
    """Integration test: heading suggestions flow through the main converter."""

    def test_heading_produces_one_suggestion(self):
        actions = [
            _make_record(
                Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
                heading_after=130.0,
                route_after=[_vec_turn_wp(1), "ARFIS"],
            )
        ]
        result, _ = convert_actions_to_suggestions(actions)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_HEADING

    def test_heading_without_return_produces_one_suggestion(self):
        actions = [
            _make_record(
                Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
                heading_after=130.0,
            )
        ]
        result, _ = convert_actions_to_suggestions(actions)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_HEADING


# ── Speed tests ──────────────────────────────────────────────────────────────


class TestConvertSpeed:
    def test_speed_up_medium(self):
        record = _make_record(Action.SPEED_UP_MEDIUM, speed_after=215.0)
        result = _convert_speed(record)
        assert result is not None
        assert result["eventType"] == _EVENT_SPEED
        assert result["eventArgs"] == [_DEFAULT_CALL_SIGN, 220]

    def test_slow_down_small(self):
        record = _make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0)
        result = _convert_speed(record)
        assert result is not None
        assert result["eventType"] == _EVENT_SPEED
        assert result["eventArgs"] == [_DEFAULT_CALL_SIGN, 180]

    def test_slow_down_large(self):
        record = _make_record(Action.SLOW_DOWN_LARGE, speed_after=150.0)
        result = _convert_speed(record)
        assert result is not None
        assert result["eventType"] == _EVENT_SPEED
        assert result["eventArgs"] == [_DEFAULT_CALL_SIGN, 150]

    def test_time_to_execute_is_set(self):
        record = _make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0, decision_time_s=50.0)
        result = _convert_speed(record)
        assert result is not None
        assert result["timeToExecute"] == _time_to_execute(50.0)

    def test_snapshot_time_offset(self):
        record = _make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0, decision_time_s=50.0)
        result = _convert_speed(record, snapshot_time=10.0)
        assert result is not None
        assert result["timeToExecute"] == _time_to_execute(50.0, 10.0)

    def test_non_speed_action_returns_none(self):
        record = _make_record(Action.DO_NOTHING)
        assert _convert_speed(record) is None


# ── Skip tests ───────────────────────────────────────────────────────────────


class TestConvertSkip:
    def test_skip_1_waypoint(self):
        record = _make_record(
            Action.SKIP_1_WAYPOINT,
            route_after=["WP_B", "WP_C"],
        )
        result = _convert_skip(record)
        assert result is not None
        assert result["eventType"] == _EVENT_WAYPOINT
        assert result["eventArgs"][1] == {"name": "WP_B"}

    def test_skip_1_has_time_to_execute(self):
        record = _make_record(
            Action.SKIP_1_WAYPOINT,
            route_after=["WP_B"],
            decision_time_s=60.0,
        )
        result = _convert_skip(record)
        assert result is not None
        assert result["timeToExecute"] == _time_to_execute(60.0)

    def test_skip_1_empty_route_after_returns_none(self):
        record = _make_record(Action.SKIP_1_WAYPOINT, route_after=[])
        assert _convert_skip(record) is None

    def test_skip_2_waypoints(self):
        # route_after = route[:1] + route[3:] = ["WP_A", "WP_D", "WP_E"]
        record = _make_record(
            Action.SKIP_2_WAYPOINTS_NOT_NEXT,
            route_after=["WP_A", "WP_D", "WP_E"],
        )
        result = _convert_skip(record)
        assert result is not None
        assert result["eventType"] == _EVENT_DCT_FROM_TO
        assert result["eventArgs"][1] == {"from": "WP_A", "to": "WP_D"}

    def test_skip_3_waypoints(self):
        # route_after = route[:1] + route[4:] = ["WP_A", "WP_E"]
        record = _make_record(
            Action.SKIP_3_WAYPOINTS_NOT_NEXT,
            route_after=["WP_A", "WP_E"],
        )
        result = _convert_skip(record)
        assert result is not None
        assert result["eventArgs"][1] == {"from": "WP_A", "to": "WP_E"}

    def test_skip_4_waypoints(self):
        # route_after = route[:1] + route[5:] = ["WP_A", "WP_F"]
        record = _make_record(
            Action.SKIP_4_WAYPOINTS_NOT_NEXT,
            route_after=["WP_A", "WP_F"],
        )
        result = _convert_skip(record)
        assert result is not None
        assert result["eventArgs"][1] == {"from": "WP_A", "to": "WP_F"}

    def test_skip_2_route_too_short_returns_none(self):
        record = _make_record(
            Action.SKIP_2_WAYPOINTS_NOT_NEXT,
        )
        assert _convert_skip(record) is None

    def test_non_skip_action_returns_none(self):
        record = _make_record(Action.DO_NOTHING)
        assert _convert_skip(record) is None


# ── Vector to ILS tests ──────────────────────────────────────────────────────


class TestConvertVectorToIls:
    def test_vector_to_ils(self):
        record = _make_record(
            Action.VECTOR_TO_ILS,
            route_after=[_MXP_END],
            decision_time_s=80.0,
        )
        result = _convert_vector_to_ils(record)
        assert result is not None
        assert result["eventType"] == _EVENT_WAYPOINT
        assert result["eventArgs"] == [_DEFAULT_CALL_SIGN, {"name": _MXP_END}]
        assert result["timeToExecute"] == _time_to_execute(80.0)

    def test_non_vector_action_returns_none(self):
        record = _make_record(Action.DO_NOTHING)
        assert _convert_vector_to_ils(record) is None


# ── Return-to-route tests ────────────────────────────────────────────────────


class TestConvertReturnToRoute:
    def test_return_to_route_produces_waypoint_suggestion(self):
        record = {
            "action": Action.RETURN_TO_ROUTE,
            "call_sign": _DEFAULT_CALL_SIGN,
            "decision_time_s": 200.0,
            "return_wp": "ARFIS",
        }
        result = _convert_return_to_route(record)
        assert result is not None
        assert result["eventType"] == _EVENT_WAYPOINT
        assert result["eventArgs"] == [_DEFAULT_CALL_SIGN, {"name": "ARFIS"}]
        assert result["timeToExecute"] == _time_to_execute(200.0)

    def test_return_to_route_with_snapshot_time(self):
        record = {
            "action": Action.RETURN_TO_ROUTE,
            "call_sign": _DEFAULT_CALL_SIGN,
            "decision_time_s": 200.0,
            "return_wp": "ARFIS",
        }
        result = _convert_return_to_route(record, snapshot_time=50.0)
        assert result is not None
        assert result["timeToExecute"] == _time_to_execute(200.0, 50.0)

    def test_non_return_action_returns_none(self):
        record = _make_record(Action.DO_NOTHING)
        assert _convert_return_to_route(record) is None

    def test_missing_return_wp_returns_none(self):
        record = {
            "action": Action.RETURN_TO_ROUTE,
            "call_sign": _DEFAULT_CALL_SIGN,
            "decision_time_s": 200.0,
        }
        assert _convert_return_to_route(record) is None


# ── Trombone tests ───────────────────────────────────────────────────────────


class TestConvertTrombone:
    def test_mxp_trombone_midpoint_suggestion(self):
        record = _make_record(
            Action.LENGTHEN_TROMBONE,
            route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
        )
        result = _convert_trombone(record)
        assert result is not None
        assert result["eventType"] == _EVENT_DCT_FROM_TO
        # trombone_wps = [TB_1, TB_2, TB_3, TB_4], mid=2 → from=TB_2, to=TB_3
        assert result["eventArgs"][1] == {"from": "TB_2", "to": "TB_3"}

    def test_mxp_shorten_trombone_uses_route_after(self):
        record = _make_record(
            Action.SHORTEN_TROMBONE,
            route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
        )
        result = _convert_trombone(record)
        assert result is not None
        assert result["eventArgs"][1] == {"from": "TB_2", "to": "TB_3"}

    def test_mxp_single_waypoint_falls_back_to_section_bounds(self):
        record = _make_record(
            Action.LENGTHEN_TROMBONE,
            route_after=[_MXP_START, "TB_1", _MXP_END],
        )
        result = _convert_trombone(record)
        assert result is not None
        assert result["eventArgs"][1] == {"from": _MXP_START, "to": _MXP_END}

    def test_mxp_zero_waypoints_falls_back_to_section_bounds(self):
        record = _make_record(
            Action.SHORTEN_TROMBONE,
            route_after=[_MXP_START, _MXP_END],
        )
        result = _convert_trombone(record)
        assert result is not None
        assert result["eventArgs"][1] == {"from": _MXP_START, "to": _MXP_END}

    def test_missing_section_end_returns_none(self):
        record = _make_record(
            Action.SHORTEN_TROMBONE,
            route_after=[_MXP_START, "TB_1"],
        )
        assert _convert_trombone(record) is None

    def test_mxp_full_section_returns_none(self):
        """MXP apex waypoint present → full section restored → no suggestion."""
        record = _make_record(
            Action.LENGTHEN_TROMBONE,
            route_after=[_MXP_START, "TB_1", MXP_TROMBONE_END, "TB_3", _MXP_END],
        )
        assert _convert_trombone(record) is None


# ── Main converter integration tests ─────────────────────────────────────────


class TestConvertActionsToSuggestions:
    def test_do_nothing_is_skipped(self):
        actions = [_make_record(Action.DO_NOTHING)]
        result, _ = convert_actions_to_suggestions(actions)
        assert result == {}

    def test_speed_action_through_converter(self):
        actions = [_make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0)]
        result, _ = convert_actions_to_suggestions(actions)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_SPEED

    def test_skip_action_through_converter(self):
        actions = [_make_record(Action.SKIP_1_WAYPOINT, route_after=["WP_B"])]
        result, _ = convert_actions_to_suggestions(actions)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_WAYPOINT

    def test_vector_to_ils_through_converter(self):
        actions = [_make_record(Action.VECTOR_TO_ILS, route_after=[_MXP_END])]
        result, _ = convert_actions_to_suggestions(actions)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_WAYPOINT

    def test_only_last_trombone_per_callsign_is_converted(self):
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", _MXP_END],
            ),
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_3", "TB_4", "TB_5", "TB_6", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        # Only the second (last) trombone should produce a suggestion
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_DCT_FROM_TO

    def test_snapshot_time_applied(self):
        actions = [_make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0, decision_time_s=50.0)]
        result, _ = convert_actions_to_suggestions(actions, snapshot_time=10.0)
        assert result[_DEFAULT_CALL_SIGN][0]["timeToExecute"] == _time_to_execute(50.0, 10.0)

    def test_multiple_callsigns(self):
        actions = [
            _make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0, call_sign="AAL100"),
            _make_record(Action.SLOW_DOWN_SMALL, speed_after=190.0, call_sign="BAW200"),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        assert "AAL100" in result
        assert "BAW200" in result
        assert len(result["AAL100"]) == 1
        assert len(result["BAW200"]) == 1

    def test_trombone_fallback_for_route_without_trombone_action(self):
        actions = [_make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0)]
        all_routes = {_DEFAULT_CALL_SIGN: [_MXP_START, "WP_MID", _MXP_END]}
        result, _ = convert_actions_to_suggestions(actions, all_routes)
        suggestions = result[_DEFAULT_CALL_SIGN]
        # Should have speed + trombone fallback
        assert len(suggestions) == 2
        assert suggestions[1]["eventType"] == _EVENT_DCT_FROM_TO
        assert suggestions[1]["eventArgs"][1] == {"from": _MXP_START, "to": _MXP_END}

    def test_raw_action_attached_to_suggestion(self):
        record = _make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0)
        actions = [record]
        result, _ = convert_actions_to_suggestions(actions)
        raw = result[_DEFAULT_CALL_SIGN][0]["rawAction"]
        assert raw["action"] == Action.SLOW_DOWN_SMALL
        assert raw["actionName"] == "SLOW_DOWN_SMALL"
        assert raw["call_sign"] == _DEFAULT_CALL_SIGN

    def test_raw_action_absent_on_fallback_suggestion(self):
        actions = [_make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0)]
        all_routes = {_DEFAULT_CALL_SIGN: [_MXP_START, "WP_MID", _MXP_END]}
        result, _ = convert_actions_to_suggestions(actions, all_routes)
        fallback = result[_DEFAULT_CALL_SIGN][1]
        assert "rawAction" not in fallback

    def test_trombone_fallback_skipped_when_trombone_action_exists(self):
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", _MXP_END],
            ),
        ]
        all_routes = {_DEFAULT_CALL_SIGN: [_MXP_START, "WP_MID", _MXP_END]}
        result, _ = convert_actions_to_suggestions(actions, all_routes)
        # Only the trombone suggestion, no fallback
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_DCT_FROM_TO
        assert result[_DEFAULT_CALL_SIGN][0]["eventArgs"][1] == {
            "from": "TB_1",
            "to": "TB_2",
        }

    def test_trombone_fallback_skipped_when_skip_removed_section_start(self):
        """Skip action that removes MXP section start suppresses the fallback."""
        actions = [
            _make_record(
                Action.SKIP_1_WAYPOINT,
                route_before=[_MXP_START, "WP_MID", _MXP_END],
                route_after=["WP_MID", _MXP_END],
            ),
        ]
        all_routes = {_DEFAULT_CALL_SIGN: [_MXP_START, "WP_MID", _MXP_END]}
        result, _ = convert_actions_to_suggestions(actions, all_routes)
        suggestions = result[_DEFAULT_CALL_SIGN]
        assert len(suggestions) == 1
        assert suggestions[0]["eventType"] == _EVENT_WAYPOINT

    def test_trombone_suppressed_when_skip_removes_from_wp(self):
        """Skip after trombone removes the 'from' waypoint → no trombone suggestion."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            # Skip removes TB_1 and TB_2; TB_2 is the trombone suggestion's "from".
            _make_record(
                Action.SKIP_2_WAYPOINTS_NOT_NEXT,
                route_before=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
                route_after=[_MXP_START, "TB_3", "TB_4", _MXP_END],
            ),
        ]
        result, raw_actions = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        # Only the skip suggestion remains; trombone suggestion suppressed
        assert len(suggestions) == 1
        assert suggestions[0]["rawAction"]["actionName"] == "SKIP_2_WAYPOINTS_NOT_NEXT"
        assert _DEFAULT_CALL_SIGN in raw_actions

    def test_skip_suggestion_suppressed_when_route_unchanged(self):
        """Skip whose route_before == route_after didn't shorten the route → suppress."""
        actions = [
            _make_record(
                Action.SKIP_2_WAYPOINTS_NOT_NEXT,
                route_before=["A", "D", "E"],
                route_after=["A", "D", "E"],
                call_sign="AAL2",
            ),
        ]
        result, raw_actions = convert_actions_to_suggestions(actions)
        assert result["AAL2"] == []
        assert [ra["actionName"] for ra in raw_actions["AAL2"]] == ["SKIP_2_WAYPOINTS_NOT_NEXT"]

    def test_vector_to_ils_suggestion_suppressed_when_route_unchanged(self):
        """VECTOR_TO_ILS whose route_before == route_after didn't shorten the route → suppress."""
        actions = [
            _make_record(
                Action.VECTOR_TO_ILS,
                route_before=[_MXP_END],
                route_after=[_MXP_END],
                call_sign="AAL3",
            ),
        ]
        result, raw_actions = convert_actions_to_suggestions(actions)
        assert result["AAL3"] == []
        assert [ra["actionName"] for ra in raw_actions["AAL3"]] == ["VECTOR_TO_ILS"]

    def test_earlier_skip_suppressed_when_dropped_wps_reappear_in_later_route_before(self):
        """Earlier skip's dropped waypoints reappear in a later skip's route_before →
        the earlier action did not persist, suppress it."""
        actions = [
            _make_record(
                Action.SKIP_2_WAYPOINTS_NOT_NEXT,
                route_before=["IKREZ", "MC425", "MC420", "INLER", _MXP_END],
                route_after=["IKREZ", "INLER", _MXP_END],
                decision_time_s=1980.0,
            ),
            _make_record(
                Action.SKIP_1_WAYPOINT,
                route_before=["MC425", "MC420", "INLER", _MXP_END],
                route_after=["MC420", "INLER", _MXP_END],
                decision_time_s=2025.0,
            ),
        ]
        result, raw_actions = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        assert len(suggestions) == 1
        assert suggestions[0]["rawAction"]["actionName"] == "SKIP_1_WAYPOINT"
        assert [ra["actionName"] for ra in raw_actions[_DEFAULT_CALL_SIGN]] == [
            "SKIP_2_WAYPOINTS_NOT_NEXT"
        ]

    def test_sequential_independent_skips_both_preserved(self):
        """Earlier skip's dropped waypoints do not appear in later skip's route_before →
        both suggestions kept (aircraft progressed naturally between actions)."""
        actions = [
            _make_record(
                Action.SKIP_1_WAYPOINT,
                route_before=["A", "B", "C", "D", _MXP_END],
                route_after=["B", "C", "D", _MXP_END],
                decision_time_s=1000.0,
            ),
            _make_record(
                Action.SKIP_1_WAYPOINT,
                route_before=["C", "D", _MXP_END],
                route_after=["D", _MXP_END],
                decision_time_s=1045.0,
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        assert len(suggestions) == 2
        assert [s["rawAction"]["actionName"] for s in suggestions] == [
            "SKIP_1_WAYPOINT",
            "SKIP_1_WAYPOINT",
        ]

    def test_trombone_preserved_when_skip_removes_only_to_wp(self):
        """Skip removes only the 'to' waypoint (not 'from') → keep trombone suggestion."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            # Suggestion is {from: TB_2, to: TB_3}; skip drops TB_3 and TB_? leaving TB_2 intact.
            _make_record(
                Action.SKIP_2_WAYPOINTS_NOT_NEXT,
                route_before=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_4", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        trombone_suggestions = [s for s in suggestions if s["eventType"] == _EVENT_DCT_FROM_TO]
        assert any(
            s["eventArgs"][1] == {"from": "TB_2", "to": "TB_3"} for s in trombone_suggestions
        )

    def test_trombone_preserved_when_skip_keeps_from_wp(self):
        """Skip after trombone does NOT remove the 'from' waypoint → suggestion preserved."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            # Skip removes TB_4, but TB_2 (the "from") is still present
            _make_record(
                Action.SKIP_1_WAYPOINT,
                route_before=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        trombone_suggestions = [s for s in suggestions if s["eventType"] == _EVENT_DCT_FROM_TO]
        assert len(trombone_suggestions) == 1
        assert trombone_suggestions[0]["eventArgs"][1] == {"from": "TB_2", "to": "TB_3"}

    def test_trombone_preserved_when_from_passed_before_skip(self):
        """Aircraft passed "from" naturally before skip drops "to" → keep trombone."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            # Aircraft has flown past TB_2 (trombone "from"); route_before no longer has it.
            # SKIP_1 drops TB_3 (trombone "to"); "from" is absent because it was passed.
            _make_record(
                Action.SKIP_1_WAYPOINT,
                route_before=["TB_3", "TB_4", _MXP_END],
                route_after=["TB_4", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        trombone_suggestions = [s for s in suggestions if s["eventType"] == _EVENT_DCT_FROM_TO]
        assert len(trombone_suggestions) == 1
        assert trombone_suggestions[0]["eventArgs"][1] == {"from": "TB_2", "to": "TB_3"}

    def test_trombone_suppressed_when_later_skip_shares_from_wp(self):
        """Later SKIP_2/3/4 whose route_after[0] == trombone 'from' → suppress trombone.

        The skip does not drop the trombone's "from" waypoint from its route
        (so the existing 'dropped' rule does not fire), but the skip's own
        dct-from-to starts at the same waypoint, superseding the trombone.
        """
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            # Trombone suggestion is {from: TB_2, to: TB_3}.
            # Aircraft has advanced past TB_1; skip starts from TB_2.
            _make_record(
                Action.SKIP_2_WAYPOINTS_NOT_NEXT,
                route_before=["TB_2", "TB_3", "TB_4", _MXP_END],
                route_after=["TB_2", "TB_4", _MXP_END],
            ),
        ]
        result, raw_actions = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        # Only the skip suggestion remains; trombone suppressed.
        assert len(suggestions) == 1
        assert suggestions[0]["rawAction"]["actionName"] == "SKIP_2_WAYPOINTS_NOT_NEXT"
        assert [ra["actionName"] for ra in raw_actions[_DEFAULT_CALL_SIGN]] == ["LENGTHEN_TROMBONE"]

    def test_trombone_preserved_when_later_skip_has_different_from_wp(self):
        """Later skip's route_after[0] differs from trombone 'from' → keep both."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            # Trombone suggestion is {from: TB_2, to: TB_3}. Skip's 'from' = TB_3.
            _make_record(
                Action.SKIP_2_WAYPOINTS_NOT_NEXT,
                route_before=["TB_3", "TB_4", _MXP_END],
                route_after=["TB_3", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        assert len(suggestions) == 2
        assert [s["rawAction"]["actionName"] for s in suggestions] == [
            "LENGTHEN_TROMBONE",
            "SKIP_2_WAYPOINTS_NOT_NEXT",
        ]

    def test_trombone_preserved_when_earlier_skip_shares_from_wp(self):
        """Skip that precedes the trombone must not suppress it (later-only rule)."""
        actions = [
            _make_record(
                Action.SKIP_2_WAYPOINTS_NOT_NEXT,
                route_before=["TB_2", "X", "Y", _MXP_END],
                route_after=["TB_2", "Y", _MXP_END],
            ),
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        trombone_suggestions = [s for s in suggestions if s["eventType"] == _EVENT_DCT_FROM_TO]
        assert any(
            s["rawAction"]["actionName"] == "LENGTHEN_TROMBONE" for s in trombone_suggestions
        )

    def test_skip_1_sharing_next_waypoint_does_not_suppress_trombone(self):
        """SKIP_1 has no 'from' in its suggestion → cannot suppress by shared-from rule."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_MXP_START, _MXP_END],
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            # SKIP_1's route_after[0] == TB_2, but SKIP_1 produces aircraft-waypoint (no 'from').
            _make_record(
                Action.SKIP_1_WAYPOINT,
                route_before=["X", "TB_2", "TB_3", "TB_4", _MXP_END],
                route_after=["TB_2", "TB_3", "TB_4", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        suggestions = result[_DEFAULT_CALL_SIGN]
        trombone_suggestions = [s for s in suggestions if s["eventType"] == _EVENT_DCT_FROM_TO]
        assert any(
            s["rawAction"]["actionName"] == "LENGTHEN_TROMBONE" for s in trombone_suggestions
        )

    def test_full_trombone_produces_raw_action_not_suggestion(self):
        """Full section restored produces raw action debug data, not a suggestion."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_after=[_MXP_START, "TB_1", MXP_TROMBONE_END, "TB_3", _MXP_END],
            ),
        ]
        result, raw_actions = convert_actions_to_suggestions(actions)
        assert result[_DEFAULT_CALL_SIGN] == []
        assert _DEFAULT_CALL_SIGN in raw_actions
        assert len(raw_actions[_DEFAULT_CALL_SIGN]) == 1
        assert raw_actions[_DEFAULT_CALL_SIGN][0]["actionName"] == "LENGTHEN_TROMBONE"

    def test_shorten_trombone_as_last_produces_suggestion_from_route_after(self):
        """LENGTHEN followed by SHORTEN: suggestion built from SHORTEN's route_after."""
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            _make_record(
                Action.SHORTEN_TROMBONE,
                route_after=[_MXP_START, "TB_2", "TB_3", _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        # trombone_wps = [TB_2, TB_3], mid=1 → from=TB_2, to=TB_3
        assert result[_DEFAULT_CALL_SIGN][0]["eventArgs"][1] == {"from": "TB_2", "to": "TB_3"}
        assert result[_DEFAULT_CALL_SIGN][0]["rawAction"]["actionName"] == "SHORTEN_TROMBONE"

    def test_trailing_shorten_to_zero_emits_section_bounds_suggestion(self):
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_after=[_MXP_START, "TB_1", "TB_2", "TB_3", "TB_4", _MXP_END],
            ),
            _make_record(
                Action.SHORTEN_TROMBONE,
                route_after=[_MXP_START, _MXP_END],
            ),
        ]
        result, _ = convert_actions_to_suggestions(actions)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventArgs"][1] == {"from": _MXP_START, "to": _MXP_END}


# ── BGY airport tests ──────────────────────────────────────────────────────


class TestBgyTrombone:
    def test_bgy_trombone_single_waypoint_produces_dct_from_to(self):
        """LENGTHEN_TROMBONE on BGY OneWaySection adds only 1 waypoint."""
        record = _make_record(
            Action.LENGTHEN_TROMBONE,
            route_before=[_BGY_START_1, _BGY_END, "AVHEB"],
            route_after=[_BGY_START_1, _BGY_START_2, _BGY_END, "AVHEB"],
        )
        result = _convert_trombone(record)
        assert result is not None
        assert result["eventType"] == _EVENT_DCT_FROM_TO
        assert result["eventArgs"][1] == {"from": _BGY_START_2, "to": _BGY_END}

    def test_bgy_trombone_produces_dct_from_to(self):
        record = _make_record(
            Action.LENGTHEN_TROMBONE,
            route_before=[_BGY_START_1, _BGY_END, "AVHEB"],
            route_after=[_BGY_START_1, _BGY_START_2, "ME624", _BGY_END, "AVHEB"],
        )
        result = _convert_trombone(record)
        assert result is not None
        assert result["eventType"] == _EVENT_DCT_FROM_TO
        # added = [ME628, ME624], mid=1 → from=ME628, to=ME624
        assert result["eventArgs"][1] == {"from": "ME624", "to": _BGY_END}

    def test_bgy_full_section_from_me632_returns_none(self):
        """Full 5-waypoint BGY section from ME632 → suppress suggestion."""
        record = _make_record(
            Action.LENGTHEN_TROMBONE,
            route_after=[
                _BGY_START_1,
                _BGY_START_2,
                "ME624",
                "ME620",
                "ME616",
                "ULVUN",
                _BGY_END,
                "AVHEB",
            ],
        )
        assert _convert_trombone(record) is None

    def test_bgy_full_section_from_me628_returns_none(self):
        """Full 4-waypoint BGY section from ME628 → suppress suggestion."""
        record = _make_record(
            Action.LENGTHEN_TROMBONE,
            route_after=[_BGY_START_2, "ME624", "ME620", "ME616", "ULVUN", _BGY_END, "AVHEB"],
        )
        assert _convert_trombone(record) is None

    def test_bgy_shorten_to_zero_falls_back_to_section_bounds(self):
        record = _make_record(
            Action.SHORTEN_TROMBONE,
            route_after=[_BGY_START_1, _BGY_END, "AVHEB"],
        )
        result = _convert_trombone(record)
        assert result is not None
        assert result["eventArgs"][1] == {"from": _BGY_START_1, "to": _BGY_END}


class TestBgyVectorToIls:
    def test_bgy_vector_to_ils_uses_tixum(self):
        record = _make_record(
            Action.VECTOR_TO_ILS,
            route_after=[_BGY_END, "AVHEB"],
            decision_time_s=80.0,
        )
        result = _convert_vector_to_ils(record)
        assert result is not None
        assert result["eventArgs"] == [_DEFAULT_CALL_SIGN, {"name": _BGY_END}]


class TestBgyFallback:
    def test_bgy_fallback_produces_dct_to_tixum(self):
        actions = [_make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0)]
        all_routes = {_DEFAULT_CALL_SIGN: [_BGY_START_1, "WP_MID", _BGY_END]}
        result, _ = convert_actions_to_suggestions(actions, all_routes)
        suggestions = result[_DEFAULT_CALL_SIGN]
        assert len(suggestions) == 2
        assert suggestions[1]["eventType"] == _EVENT_DCT_FROM_TO
        assert suggestions[1]["eventArgs"][1] == {"from": _BGY_START_1, "to": _BGY_END}

    def test_bgy_fallback_with_me628_start(self):
        actions = [_make_record(Action.SLOW_DOWN_SMALL, speed_after=185.0)]
        all_routes = {_DEFAULT_CALL_SIGN: [_BGY_START_2, "WP_MID", _BGY_END]}
        result, _ = convert_actions_to_suggestions(actions, all_routes)
        suggestions = result[_DEFAULT_CALL_SIGN]
        assert len(suggestions) == 2
        assert suggestions[1]["eventArgs"][1] == {"from": _BGY_START_2, "to": _BGY_END}

    def test_bgy_fallback_skipped_when_trombone_action_exists(self):
        actions = [
            _make_record(
                Action.LENGTHEN_TROMBONE,
                route_before=[_BGY_START_1, _BGY_END],
                route_after=[_BGY_START_1, _BGY_START_2, _BGY_END],
            ),
        ]
        all_routes = {_DEFAULT_CALL_SIGN: [_BGY_START_1, "WP_MID", _BGY_END]}
        result, _ = convert_actions_to_suggestions(actions, all_routes)
        assert len(result[_DEFAULT_CALL_SIGN]) == 1
        assert result[_DEFAULT_CALL_SIGN][0]["eventType"] == _EVENT_DCT_FROM_TO
