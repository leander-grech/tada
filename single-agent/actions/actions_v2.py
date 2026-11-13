# pylint: disable=too-few-public-methods

"""Trombone clearance definitions."""

from enum import IntEnum, unique
from typing import List, Literal

import numpy as np
from config.config import Config
from flight_simulator import AircraftCommand  # type: ignore[attr-defined, import-untyped]
from models.commands import TimedCommand, TimedCommandList
from simulator.simulator import Simulator
from simulator.route_section_store import RouteSectionStore

@unique
class Action(IntEnum):
    """Clearances the agent may issue. Contiguous 0..14; NUM_ACTIONS must equal len()."""

    DO_NOTHING = 0
    SLOW_DOWN_SMALL = 1
    SLOW_DOWN_MEDIUM = 2
    SLOW_DOWN_LARGE = 3
    SPEED_UP_SMALL = 4
    SPEED_UP_MEDIUM = 5
    TURN_LEFT_AND_RETURN_TO_ROUTE = 6
    TURN_RIGHT_AND_RETURN_TO_ROUTE = 7
    LENGTHEN_TROMBONE = 8
    SHORTEN_TROMBONE = 9
    SKIP_1_WAYPOINT = 10
    SKIP_2_WAYPOINTS_NOT_NEXT = 11
    SKIP_3_WAYPOINTS_NOT_NEXT = 12
    SKIP_4_WAYPOINTS_NOT_NEXT = 13
    VECTOR_TO_ILS = 14


# Action names and the penalty multipliers
ACTION_DEFINITIONS_BASE: dict[int, dict[str, float]] = {
    # Negative value for multiplier indicates a reward.
    # Slowing costs more than speeding, following v1, and MEDIUM interpolates.
    Action.DO_NOTHING: {"multiplier": 0.001},
    Action.SLOW_DOWN_SMALL: {"multiplier": 1.2},
    Action.SLOW_DOWN_MEDIUM: {"multiplier": 1.4},
    Action.SLOW_DOWN_LARGE: {"multiplier": 1.6},
    Action.SPEED_UP_SMALL: {"multiplier": 1.0},
    Action.SPEED_UP_MEDIUM: {"multiplier": 1.0},
    # Turn off route and rejoin 10 nm later
    Action.TURN_LEFT_AND_RETURN_TO_ROUTE: {"multiplier": 1.0},
    Action.TURN_RIGHT_AND_RETURN_TO_ROUTE: {"multiplier": 1.0},
    # Reversible pair: one trombone level each, stackable
    Action.LENGTHEN_TROMBONE: {"multiplier": 1.0},
    Action.SHORTEN_TROMBONE: {"multiplier": 1.0},
    # Skip N number of waypoints
    Action.SKIP_1_WAYPOINT: {"multiplier": 1.0},
    Action.SKIP_2_WAYPOINTS_NOT_NEXT: {"multiplier": 1.0},
    Action.SKIP_3_WAYPOINTS_NOT_NEXT: {"multiplier": 1.0},
    Action.SKIP_4_WAYPOINTS_NOT_NEXT: {"multiplier": 1.0},
    # Vector directly to ILS
    Action.VECTOR_TO_ILS: {"multiplier": 1.0},
}


# Read-only probes clone the store before interpreting these clearances.
STORE_MUTATING_CLEARANCES = frozenset({Action.LENGTHEN_TROMBONE, Action.SHORTEN_TROMBONE})


def get_action_definitions() -> dict:
    """Get action definitions with penalties calculated based on config."""
    return {k: {**v, "penalty": float(v["multiplier"])} for k, v in ACTION_DEFINITIONS_BASE.items()}


def interpret_action(
        action,
        call_sign: str,
        simulator: Simulator,
        route_trombone_utils: RouteSectionStore) -> TimedCommandList:
    """Convert discrete action into a list of timed AircraftCommands."""

    assert simulator.world is not None, "World is not initialized"

    # Convert to Action enum to handle numpy arrays and ensure type safety
    try:
        action_enum = Action(action)
    except (ValueError, TypeError):
        return TimedCommandList(commands=[])

    action_definitions = get_action_definitions()
    action_info = action_definitions.get(action_enum)

    if action_info is None:
        return TimedCommandList(commands=[])

    try:
        aircraft = simulator.world.get_aircraft_by_callsign(call_sign)
    except Exception as e:
        print(call_sign)
        raise e

    current_vel = aircraft.state.velocity_knots
    current_heading = (np.degrees(np.arctan2(current_vel.y, current_vel.x))) % 360

    target_vel = aircraft.target_state.speed
    target_heading = aircraft.target_state.heading

    commands = _get_command_for_action(
        action_enum,  # Pass only the enum value
        call_sign,
        target_vel or current_vel.length(),
        target_heading or current_heading,
        simulator,
        route_trombone_utils,
    )

    return TimedCommandList(commands=commands)


def _get_command_for_action(
    action_id: Action,  # Change to Action enum type
    call_sign: str,
    target_vel: float,
    target_heading: float,
    simulator: Simulator,
    route_trombone_store: RouteSectionStore,
) -> List[TimedCommand]:
    """Get the appropriate command for the given action"""

    def create_turn_command(
        turn_direction: Literal["left", "right"], distance_in_nm: int, skip_count: int
    ) -> List[TimedCommand]:
        # Convert nautical miles to meters (1 nm = 1852 meters)
        distance_in_m = 1852 * distance_in_nm

        # Calculate time to travel the distance in seconds
        # ground_speed is in knots (nm/hour), so we need to convert to m/s
        # 1 knot = 1 nm/hour = 1852 m/hour = 0.5144 m/s
        speed_in_m_s = target_vel * 0.5144
        time_to_travel_seconds = distance_in_m / speed_in_m_s if speed_in_m_s > 0 else 0
        return_to_route_seconds = int(time_to_travel_seconds)

        heading_adjustment = Config.ACTION_HEADING_ADJUSTMENT
        turn_angle = -heading_adjustment if turn_direction == "right" else heading_adjustment
        new_heading = (target_heading + turn_angle) % 360
        _, current_route = simulator.get_current_route(call_sign)

        actions = [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE_FOLLOWING(  # type: ignore [operator]
                    call_sign=call_sign,
                    follow_route=False,
                ),
                action_id=action_id,
            ),
            TimedCommand(
                time_step=1,
                command=AircraftCommand.SET_TARGET_HEADING(  # type: ignore [operator]
                    call_sign=call_sign,
                    heading=new_heading,
                ),
                action_id=action_id,
            ),
            TimedCommand(
                time_step=return_to_route_seconds,
                command=AircraftCommand.SET_ROUTE_FOLLOWING(  # type: ignore [operator]
                    call_sign=call_sign,
                    follow_route=True,
                ),
                action_id=action_id,
            ),
        ]

        if current_route and skip_count > 0:
            new_route = current_route[skip_count:]
            actions.insert(
                1,  # Insert after SET_ROUTE_FOLLOWING but before SET_TARGET_HEADING
                TimedCommand(
                    # Known simulator limitation: commands issued to one aircraft in the same
                    # time step are not applied reliably, so the route update is scheduled at a later time_step.
                    time_step=2,
                    command=AircraftCommand.SET_ROUTE(  # type: ignore [operator]
                        call_sign=call_sign,
                        route=new_route,
                    ),
                    action_id=action_id,
                ),
            )

        return actions

    def create_lengthen_trombone_command(num_groups_to_skip: int) -> List[TimedCommand]:

        _, current_route = simulator.get_current_route(call_sign)
        if not current_route:
            return []

        new_route = route_trombone_store.add_section_to_route(call_sign, current_route, num_groups_to_skip)
        if len(new_route) < Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM:
            return []

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # type: ignore [operator]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
            )
        ]

    def create_shorten_trombone_command(num_levels: int) -> List[TimedCommand]:
        """Exact inverse of the lengthen command: hand levels back to the store.

        Only levels the aircraft has not yet flown can be returned; the store stops the
        unwind at the first one it has passed, so this is a no-op rather than an error when
        nothing is removable. The mask normally prevents that case anyway.
        """
        _, current_route = simulator.get_current_route(call_sign)
        if not current_route:
            return []

        new_route = route_trombone_store.remove_section_from_route(
            call_sign, current_route, num_levels
        )
        if len(new_route) == len(current_route):
            return []  # nothing was removable
        if len(new_route) < Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM:
            return []

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # type: ignore [operator]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
            )
        ]

    def create_skip_waypoints_command(num_wps: int, offset: int) -> List[TimedCommand]:
        _, current_route = simulator.get_current_route(call_sign)
        if current_route is None:
            return []

        skip_end = offset + num_wps
        new_route = current_route[:offset] + current_route[skip_end:]

        # Require at least two waypoints to remain
        if len(new_route) < Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM:
            return []

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # type: ignore [operator]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
            )
        ]

    def create_speed_command(speed_delta: float) -> List[TimedCommand]:
        new_target_vel = target_vel + speed_delta

        return [
            TimedCommand(
                command=AircraftCommand.SET_TARGET_SPEED(  # type: ignore [operator]
                    call_sign=call_sign,
                    speed=new_target_vel,
                ),
                action_id=action_id,
            )
        ]

    def create_vector_to_ils_command() -> List[TimedCommand]:
        _, current_route = simulator.get_current_route(call_sign)
        if not current_route:
            return []

        new_route = current_route[-Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM :]

        return [
            TimedCommand(
                command=AircraftCommand.SET_ROUTE(  # type: ignore [operator]
                    call_sign=call_sign,
                    route=new_route,
                ),
                action_id=action_id,
            )
        ]

    action_map = {
        Action.DO_NOTHING: lambda: [TimedCommand(time_step=0, command=None, action_id=action_id)],
        Action.SLOW_DOWN_SMALL: lambda: create_speed_command(-Config.ACTION_SPEED_ADJUSTMENT_SMALL),
        Action.SLOW_DOWN_MEDIUM: lambda: create_speed_command(-Config.ACTION_SPEED_ADJUSTMENT_MEDIUM),
        Action.SLOW_DOWN_LARGE: lambda: create_speed_command(-Config.ACTION_SPEED_ADJUSTMENT_LARGE),
        Action.SPEED_UP_SMALL: lambda: create_speed_command(Config.ACTION_SPEED_ADJUSTMENT_SMALL),
        Action.SPEED_UP_MEDIUM: lambda: create_speed_command(Config.ACTION_SPEED_ADJUSTMENT_MEDIUM),
        Action.TURN_LEFT_AND_RETURN_TO_ROUTE: lambda: create_turn_command("left", 10, 0),
        Action.TURN_RIGHT_AND_RETURN_TO_ROUTE: lambda: create_turn_command("right", 10, 0),
        # One level each way. Stacking is what gives cumulative range: three lengthens with
        # no shorten between them is +3, and each is individually undoable.
        Action.LENGTHEN_TROMBONE: lambda: create_lengthen_trombone_command(1),
        Action.SHORTEN_TROMBONE: lambda: create_shorten_trombone_command(1),
        Action.SKIP_1_WAYPOINT: lambda: create_skip_waypoints_command(1, 0),
        Action.SKIP_2_WAYPOINTS_NOT_NEXT: lambda: create_skip_waypoints_command(2, 1),
        Action.SKIP_3_WAYPOINTS_NOT_NEXT: lambda: create_skip_waypoints_command(3, 1),
        Action.SKIP_4_WAYPOINTS_NOT_NEXT: lambda: create_skip_waypoints_command(4, 1),
        Action.VECTOR_TO_ILS: create_vector_to_ils_command,
    }

    # Get the command function for the action, or return empty list if not found
    command_function = action_map.get(action_id)
    return command_function() if command_function else []