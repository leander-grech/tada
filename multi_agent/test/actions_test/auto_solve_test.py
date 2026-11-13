"""Tests for staged trombone auto-solve functionality."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from flight_simulator import RoutePoint
from support.geometry import route_point

import actions.auto_solve as auto_solve_module
from actions.auto_solve import (
    apply_auto_solve_to_routes,
    preview_auto_solve_to_routes,
    select_auto_solve_actions,
)
from config.settings import settings_from_config
from simulator.route_section_store import BackAndForthSection, OneWaySection, RouteSectionStore


def _trombone_route() -> list[RoutePoint]:
    return [
        route_point("START", 0, 0),
        route_point("MC761", 5, 0),
        route_point("F1", 6, 1),
        route_point("F2", 7, 2),
        route_point("B2", 7, 3),
        route_point("B1", 6, 4),
        route_point("INLER", 5, 5),
        route_point("END", 10, 5),
    ]


class TestPeekPathPoints:
    def test_back_and_forth_peek_returns_correct_points_without_mutation(self) -> None:
        points = [
            route_point("F1", 0, 0),
            route_point("F2", 1, 0),
            route_point("B2", 1, 1),
            route_point("B1", 0, 1),
        ]
        section = BackAndForthSection(points)

        peeked = section.peek_path_points(1)
        assert [p.waypoint_name for p in peeked] == ["F1", "B1"]
        assert section.num_levels() == 2

        peeked2 = section.peek_path_points(2)
        assert [p.waypoint_name for p in peeked2] == ["F1", "F2", "B2", "B1"]
        assert section.num_levels() == 2

    def test_one_way_peek_returns_correct_points_without_mutation(self) -> None:
        points = [route_point("A", 0, 0), route_point("B", 3, 0), route_point("C", 6, 0)]
        section = OneWaySection(points)

        peeked = section.peek_path_points(2)
        assert [p.waypoint_name for p in peeked] == ["A", "B"]
        assert section.num_levels() == 3

    def test_peek_raises_on_invalid_levels(self) -> None:
        points = [
            route_point("F1", 0, 0),
            route_point("F2", 1, 0),
            route_point("B2", 1, 1),
            route_point("B1", 0, 1),
        ]
        section = BackAndForthSection(points)

        with pytest.raises(ValueError):
            section.peek_path_points(0)
        with pytest.raises(ValueError):
            section.peek_path_points(3)


class TestRouteSectionStorePreview:
    def test_preview_section_to_route_inserts_points_without_consuming(self) -> None:
        store = RouteSectionStore()
        cut_route = store.cut_out_section("AC1", _trombone_route())

        preview_route = store.preview_section_to_route("AC1", cut_route, 1)

        assert [p.waypoint_name for p in preview_route] == [
            "START",
            "MC761",
            "F1",
            "B1",
            "INLER",
            "END",
        ]
        assert store.num_levels_can_add_to_route("AC1", cut_route) == 2

    def test_preview_section_to_route_returns_original_route_for_unknown_aircraft(self) -> None:
        store = RouteSectionStore()
        route = [route_point("A", 0, 0), route_point("B", 1, 0)]

        assert store.preview_section_to_route("UNKNOWN", route, 1) == route


class TestPreviewAutoSolve:
    def test_preview_builds_route_commands_without_consuming_store(self) -> None:
        store = RouteSectionStore()
        cut_route = store.cut_out_section("AC1", _trombone_route())

        commands = preview_auto_solve_to_routes(
            actions={"AC1": 2},
            route_store=store,
            routes_by_aircraft={"AC1": cut_route},
        )

        assert len(commands) == 1
        preview_route = cast(Any, commands[0]).route
        assert [wp.waypoint_name for wp in preview_route] == [
            "START",
            "MC761",
            "F1",
            "F2",
            "B2",
            "B1",
            "INLER",
            "END",
        ]
        assert store.num_levels_can_add_to_route("AC1", cut_route) == 2


class TestSelectAutoSolveActions:
    def test_escalates_only_aircraft_still_outside_target_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RouteSectionStore()
        early_route = store.cut_out_section("EARLY", _trombone_route())
        very_early_route = store.cut_out_section("VERY_EARLY", _trombone_route())

        seen_actions: list[dict[str, int]] = []

        def fake_simulate_timing_for_actions(**kwargs) -> dict[str, float]:
            actions = dict(kwargs["actions"])
            seen_actions.append(actions)
            if actions == {"EARLY": 1, "VERY_EARLY": 1}:
                return {"EARLY": -20.0, "VERY_EARLY": -90.0}
            if actions == {"EARLY": 1, "VERY_EARLY": 2}:
                return {"EARLY": -20.0, "VERY_EARLY": -40.0}
            raise AssertionError(f"unexpected actions: {actions}")

        monkeypatch.setattr(
            auto_solve_module,
            "_simulate_timing_for_actions",
            lambda **kwargs: _exact_evidence(fake_simulate_timing_for_actions(**kwargs)),
        )

        actions = select_auto_solve_actions(
            world=cast(Any, object()),
            baseline_deviations={
                "EARLY": -80.0,
                "VERY_EARLY": -180.0,
                "WITHIN_WINDOW": -20.0,
                "LATE": 40.0,
            },
            route_store=store,
            routes_by_aircraft={
                "EARLY": early_route,
                "VERY_EARLY": very_early_route,
                "WITHIN_WINDOW": [],
            },
            rollout_steps=12,
            target_window_s=60.0,
            time_between_actions=settings_from_config().action.time_between_actions,
        )

        assert actions == {"EARLY": 1, "VERY_EARLY": 2}
        assert seen_actions == [
            {"EARLY": 1, "VERY_EARLY": 1},
            {"EARLY": 1, "VERY_EARLY": 2},
        ]

    def test_skips_aircraft_without_available_levels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen_calls: list[dict[str, int]] = []

        monkeypatch.setattr(
            auto_solve_module,
            "_simulate_timing_for_actions",
            lambda **kwargs: seen_calls.append(dict(kwargs["actions"])) or {},
        )

        actions = select_auto_solve_actions(
            world=cast(Any, object()),
            baseline_deviations={"NOTROM": -100.0, "LATE": 50.0},
            route_store=RouteSectionStore(),
            routes_by_aircraft={"NOTROM": [route_point("A", 0, 0), route_point("B", 1, 0)]},
            rollout_steps=12,
            target_window_s=60.0,
            time_between_actions=settings_from_config().action.time_between_actions,
        )

        assert actions == {}
        assert seen_calls == []

    def test_rejects_level_that_is_worse_than_doing_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RouteSectionStore()
        route = store.cut_out_section("EARLY", _trombone_route())

        monkeypatch.setattr(
            auto_solve_module,
            "_simulate_timing_for_actions",
            lambda **kwargs: _exact_evidence({"EARLY": 180.0}),
        )

        actions = select_auto_solve_actions(
            world=cast(Any, object()),
            baseline_deviations={"EARLY": -80.0},
            route_store=store,
            routes_by_aircraft={"EARLY": route},
            rollout_steps=12,
            target_window_s=60.0,
            time_between_actions=settings_from_config().action.time_between_actions,
        )

        assert actions == {}

    def test_keeps_best_so_far_when_higher_level_is_worse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RouteSectionStore()
        route = store.cut_out_section("EARLY", _trombone_route())

        def fake_simulate_timing_for_actions(**kwargs) -> dict[str, float]:
            if kwargs["actions"] == {"EARLY": 1}:
                return {"EARLY": -40.0}
            if kwargs["actions"] == {"EARLY": 2}:
                return {"EARLY": 120.0}
            raise AssertionError(f"unexpected actions: {kwargs['actions']}")

        monkeypatch.setattr(
            auto_solve_module,
            "_simulate_timing_for_actions",
            lambda **kwargs: _exact_evidence(fake_simulate_timing_for_actions(**kwargs)),
        )

        actions = select_auto_solve_actions(
            world=cast(Any, object()),
            baseline_deviations={"EARLY": -180.0},
            route_store=store,
            routes_by_aircraft={"EARLY": route},
            rollout_steps=12,
            target_window_s=60.0,
            time_between_actions=settings_from_config().action.time_between_actions,
        )

        assert actions == {"EARLY": 1}

    def test_escalates_up_to_store_available_levels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without an artificial cap, escalation uses the store's available levels directly."""
        seen_actions: list[dict[str, int]] = []
        route_store = cast(
            Any,
            SimpleNamespace(num_levels_can_add_to_route=lambda callsign, route: 6),
        )

        def fake_simulate_timing_for_actions(**kwargs) -> dict[str, float]:
            actions = dict(kwargs["actions"])
            seen_actions.append(actions)
            return {"EARLY": -500.0 + 100.0 * actions["EARLY"]}

        monkeypatch.setattr(
            auto_solve_module,
            "_simulate_timing_for_actions",
            lambda **kwargs: _exact_evidence(fake_simulate_timing_for_actions(**kwargs)),
        )

        actions = select_auto_solve_actions(
            world=cast(Any, object()),
            baseline_deviations={"EARLY": -500.0},
            route_store=route_store,
            routes_by_aircraft={"EARLY": [route_point("A", 0, 0), route_point("B", 1, 0)]},
            rollout_steps=12,
            target_window_s=60.0,
            time_between_actions=settings_from_config().action.time_between_actions,
        )

        assert actions == {"EARLY": 5}
        assert seen_actions == [
            {"EARLY": 1},
            {"EARLY": 2},
            {"EARLY": 3},
            {"EARLY": 4},
            {"EARLY": 5},
        ]


class TestApplyAutoSolve:
    def test_applies_trombone_and_consumes_store_levels(self) -> None:
        store = RouteSectionStore()
        cut_route = store.cut_out_section("AC1", _trombone_route())
        assert store.num_levels_can_add_to_route("AC1", cut_route) == 2

        commands, _ = apply_auto_solve_to_routes(
            actions={"AC1": 1},
            route_store=store,
            routes_by_aircraft={"AC1": cut_route},
        )

        assert len(commands) == 1
        applied_route = cast(Any, commands[0]).route
        new_route = [rp.waypoint_name for rp in applied_route]
        assert "F1" in new_route
        assert store.num_levels_can_add_to_route("AC1", applied_route) == 1


def test_interval_with_unknown_sign_does_not_justify_further_lengthening(monkeypatch):
    from outcomes.evidence import ArrivalEvidence, TimingEvidence
    from outcomes.ledger import OutcomeSnapshot

    calls = []

    class Branch:
        def fork(self):
            return self

        def forecast(self, world, _steps):
            calls.append(world)
            self.forecast_snapshot = OutcomeSnapshot(
                (),
                (ArrivalEvidence("A", True, True, TimingEvidence(-100, 20, "interval")),),
                (),
                (),
                False,
            )
            return world

    monkeypatch.setattr(auto_solve_module, "preview_auto_solve_to_routes", lambda **_: [])
    store = SimpleNamespace(num_levels_can_add_to_route=lambda *_: 3)
    selected = select_auto_solve_actions(
        world=cast(Any, object()),
        baseline_deviations={"A": -200},
        route_store=cast(Any, store),
        routes_by_aircraft={"A": []},
        rollout_steps=1,
        target_window_s=60,
        outcome_owner=Branch(),
        time_between_actions=settings_from_config().action.time_between_actions,
    )
    assert selected == {"A": 1}
    assert len(calls) == 1, "overlapping interval sign cannot justify another lengthening"


def _exact_evidence(values):
    from outcomes.evidence import TimingEvidence

    return {key: TimingEvidence(value, value, "fixture") for key, value in values.items()}
