"""Helpers for staged pre-episode trombone auto-solve."""

from __future__ import annotations

from collections.abc import Callable

from flight_simulator import (
    AircraftCommand,
    RoutePoint,
    World,
    simulate_n_next_states,
    simulate_next_state,
)

from actions.actions import PostActionState
from actions.rewards import get_aircraft_time_deviations
from outcomes.evidence import TimingEvidence, forecast_deviation_proxy
from simulator.route_section_store import RouteSectionStore
from simulator.simulator import SIMULATOR_TIMESTEP_MS

# Bump on any change that alters outputs; checkpoints record this token and reject mismatches.
SCENARIO_VERSION = "ma-scenario-v1"


def _build_route_commands(
    *,
    actions: dict[str, int],
    route_fn: Callable[[str, list[RoutePoint], int], list[RoutePoint]],
    routes_by_aircraft: dict[str, list[RoutePoint]],
) -> tuple[list[AircraftCommand], dict[str, PostActionState]]:
    commands: list[AircraftCommand] = []
    post_action_state: dict[str, PostActionState] = {}
    for callsign, levels in sorted(actions.items()):
        route = routes_by_aircraft.get(callsign)
        if route is None:
            continue
        new_route = route_fn(callsign, route, levels)
        commands.append(
            AircraftCommand.SET_ROUTE(  # ty: ignore[call-non-callable]
                call_sign=callsign,
                route=new_route,
            )
        )
        post_action_state[callsign] = {
            "route_before": [rp.waypoint_name for rp in route],
            "route_after": [rp.waypoint_name for rp in new_route],
        }
    return commands, post_action_state


def preview_auto_solve_to_routes(
    *,
    actions: dict[str, int],
    route_store: RouteSectionStore,
    routes_by_aircraft: dict[str, list[RoutePoint]],
) -> list[AircraftCommand]:
    """Preview route updates for staged auto-solve without consuming trombone levels."""
    commands, _ = _build_route_commands(
        actions=actions,
        route_fn=route_store.preview_section_to_route,
        routes_by_aircraft=routes_by_aircraft,
    )
    return commands


def _simulate_timing_for_actions(
    *,
    world: World,
    actions: dict[str, int],
    route_store: RouteSectionStore,
    routes_by_aircraft: dict[str, list[RoutePoint]],
    rollout_steps: int,
    time_between_actions: int,
    outcome_owner=None,
) -> dict[str, TimingEvidence | None]:
    trial_commands = preview_auto_solve_to_routes(
        actions=actions,
        route_store=route_store,
        routes_by_aircraft=routes_by_aircraft,
    )
    if outcome_owner is not None:
        branch = outcome_owner.fork()
        trial = branch.advance_next(world, 0, trial_commands) if trial_commands else world
        branch.forecast(trial, rollout_steps * int(time_between_actions))
        return {a.callsign: a.timing for a in branch.forecast_snapshot.arrivals}
    trial_world = simulate_next_state(0, world, trial_commands, None) if trial_commands else world
    terminal_world = simulate_n_next_states(
        rollout_steps * int(time_between_actions),
        SIMULATOR_TIMESTEP_MS,
        trial_world,
        [],
    )
    return {
        callsign: TimingEvidence(value, value, "native_exact_event")
        for callsign, value in get_aircraft_time_deviations(terminal_world).items()
    }


def select_auto_solve_actions(
    *,
    world: World,
    baseline_deviations: dict[str, float],
    route_store: RouteSectionStore,
    routes_by_aircraft: dict[str, list[RoutePoint]],
    rollout_steps: int,
    time_between_actions: int,
    target_window_s: float,
    outcome_owner=None,
) -> dict[str, int]:
    """Choose trombone levels via staged full-rollout simulation.

    Aircraft earlier than the target window are first given level 1. After each
    rollout, only aircraft still earlier than the target window are escalated by
    one additional level. This keeps the search to at most one rollout per
    available trombone level instead of per-aircraft trial searches.
    """
    target_window_s = max(float(target_window_s), 0.0)
    available_levels = {
        callsign: route_store.num_levels_can_add_to_route(callsign, route)
        for callsign, route in routes_by_aircraft.items()
    }
    candidates = [
        callsign
        for callsign, deviation_s in sorted(baseline_deviations.items())
        if deviation_s < -target_window_s and available_levels.get(callsign, 0) > 0
    ]
    if not candidates:
        return {}

    best_abs_deviation = {
        callsign: abs(float(baseline_deviations[callsign])) for callsign in candidates
    }
    actions: dict[str, int] = {}
    active_candidates = list(candidates)
    max_level = max(available_levels[callsign] for callsign in candidates)

    for level in range(1, max_level + 1):
        level_candidates = [
            callsign for callsign in active_candidates if available_levels.get(callsign, 0) >= level
        ]
        if not level_candidates:
            break

        trial_actions = dict(actions)
        for callsign in level_candidates:
            trial_actions[callsign] = level

        simulated_timing = _simulate_timing_for_actions(
            world=world,
            actions=trial_actions,
            route_store=route_store,
            routes_by_aircraft=routes_by_aircraft,
            rollout_steps=rollout_steps,
            time_between_actions=time_between_actions,
            outcome_owner=outcome_owner,
        )
        simulated_deviations = {
            callsign: proxy
            for callsign, evidence in simulated_timing.items()
            if (proxy := forecast_deviation_proxy(evidence)) is not None
        }
        for callsign in level_candidates:
            if callsign not in simulated_deviations:
                continue
            trial_deviation = float(simulated_deviations[callsign])
            trial_abs_deviation = abs(trial_deviation)
            if trial_abs_deviation < best_abs_deviation[callsign]:
                best_abs_deviation[callsign] = trial_abs_deviation
                actions[callsign] = level

        active_candidates = [
            callsign
            for callsign in level_candidates
            if (evidence := simulated_timing.get(callsign)) is not None
            and evidence.upper_error_s < -target_window_s
            and available_levels.get(callsign, 0) > level
        ]
        if not active_candidates:
            break

    return actions


def apply_auto_solve_to_routes(
    *,
    actions: dict[str, int],
    route_store: RouteSectionStore,
    routes_by_aircraft: dict[str, list[RoutePoint]],
) -> tuple[list[AircraftCommand], dict[str, PostActionState]]:
    """Consume selected trombone levels and return route-update commands with post-action state."""
    return _build_route_commands(
        actions=actions,
        route_fn=route_store.add_section_to_route,
        routes_by_aircraft=routes_by_aircraft,
    )
