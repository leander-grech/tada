# pylint: disable=too-few-public-methods

"""Clearance set v2pms: v2 for the canonical point-merge procedure (BGY, use case 2).

Clearances depend on the procedure. A trombone (TMB, MXP) flight is cut to
its shortest path at reset and the controller lengthens or shortens the trombone, or vectors it
to the ILS. That is v2, unchanged. A point-merge (PMS) flight instead flies its WHOLE sequencing
leg by default (simulator.route_section_store.PointMergeLeg). The controller times its exit with
a direct-to the merge point, so:

  * slot 14 is DIRECT_TO_MERGE_POINT, in place of VECTOR_TO_ILS. It sets the route to the merge
    point (TIXUM) onward and is legal while the flight is on its leg.
  * slots 8-9 (lengthen/shorten trombone) have no point-merge meaning and are never legal.
    They keep their slots so the network has v2's width, and keep v2's names, so they show
    up as 0 counts in the per-type tallies.

Speed, turn and skip clearances are v2's own commands and masks.
"""

from enum import IntEnum, unique

from flight_simulator import AircraftCommand  # type: ignore[attr-defined, import-untyped]
from models.commands import TimedCommand, TimedCommandList
from simulator.simulator import Simulator
from simulator.route_section_store import RouteSectionStore

from actions import actions_v2


@unique
class Action(IntEnum):
    """Clearances on point merge. Same slots as v2; NUM_ACTIONS must equal len()."""

    DO_NOTHING = 0
    SLOW_DOWN_SMALL = 1
    SLOW_DOWN_MEDIUM = 2
    SLOW_DOWN_LARGE = 3
    SPEED_UP_SMALL = 4
    SPEED_UP_MEDIUM = 5
    TURN_LEFT_AND_RETURN_TO_ROUTE = 6
    TURN_RIGHT_AND_RETURN_TO_ROUTE = 7
    LENGTHEN_TROMBONE = 8   # never legal on point merge
    SHORTEN_TROMBONE = 9    # never legal on point merge
    SKIP_1_WAYPOINT = 10
    SKIP_2_WAYPOINTS_NOT_NEXT = 11
    SKIP_3_WAYPOINTS_NOT_NEXT = 12
    SKIP_4_WAYPOINTS_NOT_NEXT = 13
    DIRECT_TO_MERGE_POINT = 14


NEVER_LEGAL = frozenset({Action.LENGTHEN_TROMBONE, Action.SHORTEN_TROMBONE})

ACTION_DEFINITIONS_BASE: dict[int, dict[str, float]] = {
    Action(int(k)): dict(v) for k, v in actions_v2.ACTION_DEFINITIONS_BASE.items()
}

# Nothing here edits the RouteSectionStore: a direct-to only reads the flight's leg.
STORE_MUTATING_CLEARANCES: frozenset = frozenset()


def get_action_definitions() -> dict:
    """Get action definitions with penalties calculated based on config."""
    return {k: {**v, "penalty": float(v["multiplier"])} for k, v in ACTION_DEFINITIONS_BASE.items()}


def interpret_action(
        action,
        call_sign: str,
        simulator: Simulator,
        route_trombone_utils: RouteSectionStore) -> TimedCommandList:
    """Convert a discrete clearance into timed AircraftCommands."""
    assert simulator.world is not None, "World is not initialized"
    try:
        action_enum = Action(action)
    except (ValueError, TypeError):
        return TimedCommandList(commands=[])

    if action_enum in NEVER_LEGAL:
        return TimedCommandList(commands=[])

    if action_enum == Action.DIRECT_TO_MERGE_POINT:
        _, route = simulator.get_current_route(call_sign)
        if not route or not route_trombone_utils.can_direct_to_merge(call_sign, route):
            return TimedCommandList(commands=[])
        return TimedCommandList(commands=[TimedCommand(
            command=AircraftCommand.SET_ROUTE(  # type: ignore [operator]
                call_sign=call_sign,
                route=route_trombone_utils.direct_to_merge_route(call_sign, route),
            ),
            action_id=action_enum,
        )])

    # Speed, turns and skips: v2's commands (same slot numbers).
    return actions_v2.interpret_action(int(action_enum), call_sign, simulator, route_trombone_utils)
