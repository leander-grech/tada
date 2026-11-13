"""Tests for ATCEnv auto-solve integration during reset preparation."""

from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any, cast

import pytest
from support.env_patch import patch_env_global
from support.settings import live_env_settings

import atc_env.environment as environment_module
from actions.actions import Action
from actions.rewards import get_aircraft_time_deviations, reward_options
from atc_env.environment import ATCEnv
from config.settings import settings_from_config
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.ledger import OutcomeSnapshot


class _WorldStub(SimpleNamespace):
    def get_aircraft_by_callsign(self, callsign: str):
        return next((ac for ac in self.aircraft if ac.call_sign == callsign), None)


def _route(*waypoint_names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(waypoint_name=name) for name in waypoint_names]


# Forecast deviation source; tests replace ``read`` to drive fake worlds.
_deviation_source = SimpleNamespace(read=get_aircraft_time_deviations)


def _make_env_for_auto_solve() -> ATCEnv:
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env.max_steps_per_episode = 12
    env._route_trombone_store = SimpleNamespace(has_excluded_section=lambda callsign: True)
    env.step_counter = 0
    env.last_step_action_nums = {}
    env._action_history = []
    env._pending_returns = defaultdict(deque)

    aircraft = [
        SimpleNamespace(
            call_sign="EARLY",
            route=_route("route-a"),
            target_state=SimpleNamespace(speed=None, heading=None),
            state=SimpleNamespace(
                velocity_knots=SimpleNamespace(length=lambda: 180.0, x=180.0, y=0.0)
            ),
        ),
        SimpleNamespace(
            call_sign="ONTIME",
            route=_route("route-b"),
            target_state=SimpleNamespace(speed=None, heading=None),
            state=SimpleNamespace(
                velocity_knots=SimpleNamespace(length=lambda: 190.0, x=190.0, y=0.0)
            ),
        ),
    ]
    initial_world = _WorldStub(
        aircraft=aircraft,
        age_in_seconds=42.0,
        arrival_targets=[SimpleNamespace(eta_s=300.0)],
    )
    env.simulator = SimpleNamespace(world=initial_world)
    env.reward_settings = reward_options(settings=settings_from_config().reward)
    env.outcome_state = SimpleNamespace(
        targets=(ArrivalTargetSpec("EARLY", 300), ArrivalTargetSpec("ONTIME", 300)),
        forecast_snapshot=None,
        advance_next=lambda world, dt, commands: cast(Any, environment_module).simulate_next_state(
            dt, world, commands, None
        ),
    )

    def forecast(world, steps):
        result = env.simulator.advance_multiple_steps(steps, world)
        deviations = _deviation_source.read(result)
        deviations = {"ONTIME": 0.0, **deviations}
        env.outcome_state.forecast_snapshot = OutcomeSnapshot(
            env.arrival_targets,
            tuple(
                ArrivalEvidence(
                    t.callsign,
                    True,
                    True,
                    TimingEvidence(deviations[t.callsign], deviations[t.callsign], "fixture")
                    if t.callsign in deviations
                    else None,
                )
                for t in env.arrival_targets
            ),
            (),
            (),
            False,
        )
        return result

    env._forecast = forecast
    return env


def _patch_baseline_metrics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    deviation_total: float = 120.0,
    infringement_total: float = 1.0,
) -> None:
    monkeypatch.setattr(_deviation_source, "read", lambda world: {"EARLY": -deviation_total})
    patch_env_global(
        monkeypatch,
        "get_world_infringements",
        lambda world, *_, **kwargs: {"EARLY": ["infr"] * int(infringement_total)},
    )
    patch_env_global(
        monkeypatch,
        "compute_weighted_infringement_total",
        lambda infringements, *_args, **_kwargs: float(
            sum(len(values) for values in infringements.values())
        ),
    )


def test_prepare_initial_rollout_records_pre_episode_auto_solve_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env_for_auto_solve()
    initial_world = env.simulator.world
    baseline_world = SimpleNamespace(age_in_seconds=42.0)
    policy_start_world = SimpleNamespace(age_in_seconds=42.0)
    updated_world = _WorldStub(
        aircraft=[
            SimpleNamespace(
                call_sign="EARLY",
                route=_route("route-a", "route-a-extended"),
                target_state=SimpleNamespace(speed=None, heading=None),
                state=SimpleNamespace(
                    velocity_knots=SimpleNamespace(length=lambda: 180.0, x=180.0, y=0.0),
                    position_nm_nm_ft=SimpleNamespace(x=1.5, y=2.5),
                ),
            ),
            SimpleNamespace(
                call_sign="ONTIME",
                route=_route("route-b"),
                target_state=SimpleNamespace(speed=None, heading=None),
                state=SimpleNamespace(
                    velocity_knots=SimpleNamespace(length=lambda: 190.0, x=190.0, y=0.0),
                    position_nm_nm_ft=SimpleNamespace(x=3.5, y=4.5),
                ),
            ),
        ],
        age_in_seconds=42.0,
        arrival_targets=[SimpleNamespace(eta_s=300.0)],
    )
    simulator = cast(Any, env.simulator)
    seen_rollouts: list[object] = []

    def fake_advance_multiple_steps(steps, world):
        _ = steps
        seen_rollouts.append(world)
        if world is initial_world:
            return baseline_world
        if world is updated_world:
            return policy_start_world
        raise AssertionError(f"unexpected rollout world: {world!r}")

    simulator.advance_multiple_steps = fake_advance_multiple_steps

    seen: dict[str, object] = {}

    def fake_select_auto_solve_actions(**kwargs):
        seen["select"] = kwargs
        return {"EARLY": 1}

    def fake_apply_auto_solve_to_routes(**kwargs):
        seen["apply"] = kwargs
        return ["CMD"], {
            "EARLY": {
                "target_speed": 180.0,
                "target_heading": 0.0,
                "route_before": ["route-a"],
                "route_after": ["route-a", "route-a-extended"],
            }
        }

    def fake_simulate_next_state(*args):
        seen["simulate"] = args
        return updated_world

    patch_env_global(
        monkeypatch,
        "select_auto_solve_actions",
        fake_select_auto_solve_actions,
    )
    patch_env_global(
        monkeypatch,
        "apply_auto_solve_to_routes",
        fake_apply_auto_solve_to_routes,
    )
    monkeypatch.setattr(
        environment_module,
        "simulate_next_state",
        fake_simulate_next_state,
        raising=False,
    )
    patch_env_global(
        monkeypatch,
        "capture_aircraft_positions",
        lambda world, callsigns: {"EARLY": (1.0, 2.0)},
    )
    monkeypatch.setattr(
        _deviation_source,
        "read",
        lambda world: {"EARLY": -200.0} if world is baseline_world else {"EARLY": -50.0},
    )
    patch_env_global(
        monkeypatch,
        "compute_weighted_infringement_total",
        lambda infringements, *_args, **_kwargs: float(
            sum(len(values) for values in infringements.values())
        ),
    )
    patch_env_global(
        monkeypatch,
        "get_world_infringements",
        lambda world, *_, **kwargs: (
            {"EARLY": ["raw-1", "raw-2"]} if world is baseline_world else {"EARLY": ["policy"]}
        ),
    )
    prepared_world = ATCEnv._prepare_initial_rollout(env)
    select_kwargs = cast(dict[str, Any], seen["select"])
    apply_kwargs = cast(dict[str, Any], seen["apply"])
    simulate_args = cast(tuple[Any, ...], seen["simulate"])

    assert env.simulator.world is updated_world
    assert prepared_world is policy_start_world
    assert select_kwargs["world"] is initial_world
    assert select_kwargs["baseline_deviations"] == {"EARLY": -200.0}
    assert select_kwargs["route_store"] is env._route_trombone_store
    assert select_kwargs["rollout_steps"] == 12
    assert select_kwargs["target_window_s"] == 60.0
    assert {
        callsign: [route_point.waypoint_name for route_point in route]
        for callsign, route in cast(dict[str, Any], select_kwargs["routes_by_aircraft"]).items()
    } == {"EARLY": ["route-a"]}
    assert apply_kwargs["actions"] == {"EARLY": 1}
    assert apply_kwargs["route_store"] is env._route_trombone_store
    assert {
        callsign: [route_point.waypoint_name for route_point in route]
        for callsign, route in cast(dict[str, Any], apply_kwargs["routes_by_aircraft"]).items()
    } == {"EARLY": ["route-a"]}
    assert simulate_args == (0, initial_world, ["CMD"], None)
    assert seen_rollouts == [initial_world, updated_world]
    assert env.episode_time_horizon == 300.0
    assert env.raw_initial_deviation == 200.0
    assert env.raw_initial_infringements_total == 2.0
    assert env.initial_deviation == 50.0
    assert env.initial_infringements_total == 1.0
    assert env._action_history == [
        {
            "call_sign": "EARLY",
            "action": int(Action.LENGTHEN_TROMBONE),
            "x": 1.0,
            "y": 2.0,
            "step": 0,
            "reward": 0.0,
            "decision_time_s": 42.0,
            "phase": "pre_episode",
            "target_speed": 180.0,
            "target_heading": 0.0,
            "route_before": ["route-a"],
            "route_after": ["route-a", "route-a-extended"],
        }
    ]
    assert env.last_step_action_nums == {}


def test_prepare_initial_rollout_reuses_baseline_when_no_actions_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env_for_auto_solve()
    initial_world = env.simulator.world
    baseline_world = SimpleNamespace(age_in_seconds=42.0)
    simulator = cast(Any, env.simulator)
    simulator.advance_multiple_steps = lambda steps, world: baseline_world
    _patch_baseline_metrics(monkeypatch)

    patch_env_global(
        monkeypatch,
        "select_auto_solve_actions",
        lambda **kwargs: {},
    )
    patch_env_global(
        monkeypatch,
        "apply_auto_solve_to_routes",
        lambda **kwargs: pytest.fail("apply_auto_solve_to_routes should not run"),
    )
    monkeypatch.setattr(
        environment_module,
        "simulate_next_state",
        lambda *args: pytest.fail("simulate_next_state should not run"),
        raising=False,
    )

    prepared_world = ATCEnv._prepare_initial_rollout(env)

    assert env.simulator.world is initial_world
    assert prepared_world is baseline_world
    assert env.raw_initial_deviation == 120.0
    assert env.initial_deviation == 120.0


def test_prepare_initial_rollout_falls_back_to_single_rollout_without_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env_for_auto_solve()
    env._route_trombone_store = SimpleNamespace(has_excluded_section=lambda callsign: False)  # ty: ignore[invalid-assignment]
    planning_world = SimpleNamespace(age_in_seconds=42.0)
    simulator = cast(Any, env.simulator)
    simulator.advance_multiple_steps = lambda *args, **kwargs: planning_world
    _patch_baseline_metrics(monkeypatch)

    prepared_world = ATCEnv._prepare_initial_rollout(env)

    assert prepared_world is planning_world


def test_prepare_initial_rollout_handles_missing_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env_for_auto_solve()
    world = cast(Any, env.simulator.world)
    world.aircraft[1].route = None
    baseline_world = SimpleNamespace(age_in_seconds=42.0)
    simulator = cast(Any, env.simulator)
    simulator.advance_multiple_steps = lambda steps, world: baseline_world
    _patch_baseline_metrics(monkeypatch)

    seen: dict[str, object] = {}

    def fake_select_auto_solve_actions(**kwargs):
        seen["select"] = kwargs
        return {}

    patch_env_global(
        monkeypatch,
        "select_auto_solve_actions",
        fake_select_auto_solve_actions,
    )

    prepared_world = ATCEnv._prepare_initial_rollout(env)
    select_kwargs = cast(dict[str, Any], seen["select"])

    assert select_kwargs["world"] is env.simulator.world
    assert select_kwargs["baseline_deviations"] == {"EARLY": -120.0}
    assert select_kwargs["route_store"] is env._route_trombone_store
    assert select_kwargs["rollout_steps"] == 12
    assert select_kwargs["target_window_s"] == 60.0
    assert {
        callsign: [route_point.waypoint_name for route_point in route]
        for callsign, route in cast(dict[str, Any], select_kwargs["routes_by_aircraft"]).items()
    } == {"EARLY": ["route-a"]}
    assert prepared_world is baseline_world
