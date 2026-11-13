# pylint: disable=too-few-public-methods

"""Module for air traffic environment action handling"""

from typing import Callable, Iterable, Literal, NotRequired, TypedDict

import numpy as np
from flight_simulator import (
    AircraftCommand,
    RoutePoint,
    Vec2,
    WaypointType,
)

from actions.action_table import (
    ACTION_TABLE,
    SKIP_SPECS,
    SPEED_DELTAS,
    TURN_ACTION_SPECS,
    Action,
    TurnActionSpec,
    speed_delta,
)
from common.runtime_log import runtime_print
from config.settings import ActionSettings
from geometry.vectoring import (
    find_vector_intercept_index,
    get_pending_virtual_turn_anchor_xy,
    resolve_route_positions,
    split_leading_virtual_turns,
)
from simulator.commands import SimulatorLike, TimedCommand, TimedCommandList
from simulator.route_section_store import RouteSectionStore

# Bump on any change that alters outputs; checkpoints record this token and reject mismatches.
ACTION_VERSION = "ma-action-v1"


class PostActionState(TypedDict, total=False):
    """Post-action state returned by action handlers for suggestion conversion."""

    route_after: list[str]
    route_before: list[str]
    speed_after: float
    heading_after: float


class ActionRecord(TypedDict):
    """Typed record stored in ATCEnv._action_history."""

    call_sign: str
    action: int
    x: float
    y: float
    step: int
    reward: float
    decision_time_s: NotRequired[float]
    phase: NotRequired[str]
    return_wp: NotRequired[str]
    route_after: NotRequired[list[str]]
    route_before: NotRequired[list[str]]
    speed_after: NotRequired[float]
    heading_after: NotRequired[float]


SPEED_ACTIONS: frozenset[int] = frozenset(
    range(int(Action.SLOW_DOWN_SMALL), int(Action.SPEED_UP_MEDIUM) + 1)
)

TROMBONE_ACTIONS: frozenset[int] = frozenset(
    {
        Action.LENGTHEN_TROMBONE,
        Action.SHORTEN_TROMBONE,
    }
)

INTERNAL_ONLY_ACTIONS: frozenset[int] = frozenset({Action.RETURN_TO_ROUTE})


def is_internal_action(action: int) -> bool:
    """Return whether *action* is an environment-emitted internal event."""
    return action in INTERNAL_ONLY_ACTIONS


def filter_render_action_history(history: Iterable[ActionRecord]) -> list[ActionRecord]:
    """Keep only agent-facing nonzero actions for render markers."""
    return [
        entry
        for entry in history
        if (a := int(entry.get("action", 0))) > 0 and not is_internal_action(a)
    ]


def count_nonzero_actions(history: Iterable[ActionRecord]) -> tuple[int, int, int]:
    """Count agent-facing nonzero actions, nonzero-excluding-trombone, and speed actions."""
    nz_total = 0
    nz_ex_trb = 0
    speed = 0
    for entry in history:
        action = int(entry.get("action", 0))
        if action <= 0 or is_internal_action(action):
            continue
        nz_total += 1
        if action not in TROMBONE_ACTIONS:
            nz_ex_trb += 1
        if action in SPEED_ACTIONS:
            speed += 1
    return nz_total, nz_ex_trb, speed


def turn_direction_multiplier(turn_direction: Literal["left", "right"]) -> int:
    return 1 if turn_direction == "left" else -1


ACTION_DEFINITIONS: dict[int, dict[str, float]] = {
    row.action: {"multiplier": row.penalty_multiplier} for row in ACTION_TABLE
}


def interpret_action(
    action,
    call_sign: str,
    simulator: SimulatorLike,
    route_trombone_store: RouteSectionStore,
    settings: ActionSettings,
) -> tuple[TimedCommandList, PostActionState]:
    """Convert discrete action into a list of timed AircraftCommands."""

    assert simulator.world is not None, "World is not initialized"

    # Convert to Action enum to handle numpy arrays and ensure type safety
    try:
        action_enum = Action(action)
    except (ValueError, TypeError):
        return TimedCommandList(commands=[]), PostActionState()

    aircraft = simulator.world.get_aircraft_by_callsign(call_sign)
    _, current_route = simulator.get_current_route(call_sign)
    current_pos = aircraft.state.position_nm_nm_ft

    current_vel = aircraft.state.velocity_knots
    current_heading = (np.degrees(np.arctan2(current_vel.y, current_vel.x))) % 360

    target_vel = aircraft.target_state.speed
    target_heading = aircraft.target_state.heading
    resolved_target_vel = float(target_vel) if target_vel is not None else current_vel.length()
    resolved_target_heading = (
        float(target_heading) if target_heading is not None else current_heading
    )

    commands, post_action_state = _get_command_for_action(
        action_id=action_enum,
        call_sign=call_sign,
        target_vel=resolved_target_vel,
        target_heading=resolved_target_heading,
        current_route=current_route,
        current_pos_xy=(float(current_pos.x), float(current_pos.y)),
        route_trombone_store=route_trombone_store,
        world_waypoints=simulator.world.waypoints,
        settings=settings,
    )

    return TimedCommandList(commands=commands), post_action_state


def _get_command_for_action(
    action_id: Action,
    call_sign: str,
    target_vel: float,
    target_heading: float,
    current_route: list[RoutePoint],
    current_pos_xy: tuple[float, float],
    route_trombone_store: RouteSectionStore,
    world_waypoints: dict | None = None,
    *,
    settings: ActionSettings,
) -> tuple[list[TimedCommand], PostActionState]:
    """Get the appropriate command for the given action"""

    def build_virtual_turn_route(
        turn_direction: Literal["left", "right"], distance_in_nm: int
    ) -> tuple[list[RoutePoint], float] | None:
        heading_adjustment = settings.action_heading_adjustment
        turn_angle = -heading_adjustment if turn_direction == "right" else heading_adjustment
        new_heading = (target_heading + turn_angle) % 360

        pending_virtual_prefix, downstream_route = split_leading_virtual_turns(current_route)
        if len(downstream_route) < settings.waypoints_to_leave_at_minimum:
            return None

        rad = np.deg2rad(new_heading)
        anchor_xy = get_pending_virtual_turn_anchor_xy(pending_virtual_prefix)
        x_pos, y_pos = anchor_xy if anchor_xy is not None else current_pos_xy
        x_new = x_pos + float(distance_in_nm) * float(np.cos(rad))
        y_new = y_pos + float(distance_in_nm) * float(np.sin(rad))

        # Find the best re-entry point along the downstream route.
        # On straight segments intercept_idx == 0 (same as old behaviour);
        # on trombone legs it may skip past tight 180° turn-backs.
        intercept_idx = 0
        if world_waypoints is not None:
            downstream_positions = resolve_route_positions(downstream_route, world_waypoints)
            if len(downstream_positions) >= 2:
                max_return_angle_rad = (
                    settings.max_turn_angle_after_heading_adjustment * np.pi / 180
                )
                idx = find_vector_intercept_index(
                    turn_point_xy=(x_new, y_new),
                    new_heading_rad=float(rad),
                    downstream_positions=downstream_positions,
                    max_return_angle_rad=float(max_return_angle_rad),
                    max_intercept_angle_deg=settings.skip_max_intercept_angle_deg,
                    min_remaining_waypoints=settings.waypoints_to_leave_at_minimum,
                )
                if idx < 0:
                    runtime_print(
                        "Actions",
                        "warning",
                        message="vector intercept infeasible — mask/execution mismatch",
                    )
                    return None
                intercept_idx = idx

        turning_wp = RoutePoint.create(
            waypoint_name=f"VEC_TURN_{distance_in_nm}NM",
            position_nm=Vec2(x_new, y_new),
            waypoint_type=WaypointType.FLY_BY,
            min_altitude_ft=None,
            max_altitude_ft=None,
            target_speed_knots=None,
        )
        return pending_virtual_prefix + [turning_wp] + downstream_route[intercept_idx:], new_heading

    def create_turn_command(
        turn_direction: Literal["left", "right"], distance_in_nm: int
    ) -> tuple[list[TimedCommand], PostActionState]:
        route_build = build_virtual_turn_route(turn_direction, distance_in_nm)
        if route_build is None:
            return [], {}

        new_route, new_heading = route_build
        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # ty: ignore[call-non-callable]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
            )
        ], {
            "route_after": [wp.waypoint_name for wp in new_route],
            "heading_after": new_heading,
        }

    def create_lengthen_trombone_command(
        num_groups_to_skip: int,
    ) -> tuple[list[TimedCommand], PostActionState]:
        if not current_route:
            return [], {}

        route_before = [wp.waypoint_name for wp in current_route]
        new_route = route_trombone_store.preview_section_to_route(
            call_sign, current_route, num_groups_to_skip
        )
        if len(new_route) < settings.waypoints_to_leave_at_minimum:
            return [], {}

        def apply_lengthen_to_store() -> None:
            route_trombone_store.add_section_to_route(call_sign, current_route, num_groups_to_skip)

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # ty: ignore[call-non-callable]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
                on_apply=apply_lengthen_to_store,
            )
        ], {
            "route_before": route_before,
            "route_after": [wp.waypoint_name for wp in new_route],
        }

    def create_shorten_trombone_command() -> tuple[list[TimedCommand], PostActionState]:
        if not current_route:
            return [], {}
        route_before = [wp.waypoint_name for wp in current_route]
        new_route = route_trombone_store.preview_retract_section_from_route(
            call_sign, current_route
        )
        if new_route is None or len(new_route) < settings.waypoints_to_leave_at_minimum:
            return [], {}

        def apply_retract_to_store() -> None:
            result = route_trombone_store.retract_section_from_route(call_sign, current_route)
            assert result is not None, "retraction failed after preview succeeded"

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # ty: ignore[call-non-callable]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
                on_apply=apply_retract_to_store,
            )
        ], {
            "route_before": route_before,
            "route_after": [wp.waypoint_name for wp in new_route],
        }

    def create_skip_waypoints_command(
        num_wps: int, offset: int
    ) -> tuple[list[TimedCommand], PostActionState]:
        if not current_route:
            return [], {}

        skip_end = offset + num_wps
        # Never skip into the protected tail.
        protected_start = len(current_route) - settings.waypoints_to_leave_at_minimum
        if skip_end > protected_start:
            return [], {}

        new_route = current_route[:offset] + current_route[skip_end:]

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # ty: ignore[call-non-callable]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
            )
        ], {
            "route_before": [wp.waypoint_name for wp in current_route],
            "route_after": [wp.waypoint_name for wp in new_route],
        }

    def create_speed_command(
        speed_delta: float,
    ) -> tuple[list[TimedCommand], PostActionState]:
        r = settings.speed_resolution
        new_target_vel = round((target_vel + speed_delta) / r) * r

        return [
            TimedCommand(
                command=AircraftCommand.SET_TARGET_SPEED(  # ty: ignore[call-non-callable]
                    call_sign=call_sign,
                    speed=new_target_vel,
                ),
                action_id=action_id,
            )
        ], {"speed_after": new_target_vel}

    def create_vector_to_ils_command() -> tuple[list[TimedCommand], PostActionState]:
        if not current_route:
            return [], {}

        if not route_trombone_store.has_excluded_section(call_sign):
            return [], {}
        if not route_trombone_store.has_reached_section(
            call_sign, current_route
        ) or not route_trombone_store.is_before_upwind_start(call_sign, current_route):
            return [], {}

        if len(current_route) <= settings.waypoints_to_leave_at_minimum:
            return [], {}

        new_route = current_route[-settings.waypoints_to_leave_at_minimum :]

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # ty: ignore[call-non-callable]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
            )
        ], {
            "route_before": [wp.waypoint_name for wp in current_route],
            "route_after": [wp.waypoint_name for wp in new_route],
        }

    def build_turn_command_handler(
        spec: TurnActionSpec,
    ) -> Callable[[], tuple[list[TimedCommand], PostActionState]]:
        def _handler() -> tuple[list[TimedCommand], PostActionState]:
            return create_turn_command(
                spec.turn_direction,
                spec.distance_in_nm,
            )

        return _handler

    action_map: dict[Action, Callable[[], tuple[list[TimedCommand], PostActionState]]] = {
        Action.DO_NOTHING: lambda: (
            [TimedCommand(time_step=0, command=None, action_id=action_id)],
            {},
        ),
        **{
            action: (lambda action=action: create_speed_command(speed_delta(action, settings)))
            for action in SPEED_DELTAS
        },
        **{action: build_turn_command_handler(spec) for action, spec in TURN_ACTION_SPECS.items()},
        Action.LENGTHEN_TROMBONE: lambda: create_lengthen_trombone_command(1),
        Action.SHORTEN_TROMBONE: lambda: create_shorten_trombone_command(),
        **{
            action: (lambda spec=spec: create_skip_waypoints_command(*spec))
            for action, spec in SKIP_SPECS.items()
        },
        Action.VECTOR_TO_ILS: create_vector_to_ils_command,
    }

    # Get the command function for the action, or return empty list if not found
    command_function = action_map.get(action_id)
    if command_function is None:
        return [], {}
    return command_function()
