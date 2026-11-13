"""Single source of truth for the agent action vocabulary and per-action facts."""

from dataclasses import dataclass
from enum import IntEnum, unique
from types import MappingProxyType
from typing import Literal, Mapping

from config.settings import ActionSettings


@unique
class Action(IntEnum):
    """Enumerations for actions of this env"""

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
    RETURN_TO_ROUTE = 15  # Internal-only: logged when vectoring aircraft returns to route


SpeedStep = Literal["small", "medium", "large"]
ActionKind = Literal["noop", "speed", "turn", "trombone", "skip", "ils"]


@dataclass(frozen=True)
class ActionRow:
    """One agent-facing action and its reward penalty multiplier."""

    action: Action
    penalty_multiplier: float
    kind: ActionKind


@dataclass(frozen=True)
class TurnActionSpec:
    """Shared definition for vectoring turn actions."""

    turn_direction: Literal["left", "right"]
    distance_in_nm: int


# Penalty multipliers are tiered to guide action preference:
# trombones (near-free) → slow-down (low) → ILS (moderate) → speed-up/skips/vectoring (high)
ACTION_TABLE: tuple[ActionRow, ...] = (
    ActionRow(Action.DO_NOTHING, 0.0, "noop"),
    # Slow-down: low cost, keeps route intact and natural in TMA.
    ActionRow(Action.SLOW_DOWN_SMALL, 0.7, "speed"),
    ActionRow(Action.SLOW_DOWN_MEDIUM, 0.7, "speed"),
    ActionRow(Action.SLOW_DOWN_LARGE, 0.7, "speed"),
    # Speed-up: counterintuitive in TMA — penalised at 2× slow-down rate.
    ActionRow(Action.SPEED_UP_SMALL, 1.5, "speed"),
    ActionRow(Action.SPEED_UP_MEDIUM, 1.5, "speed"),
    # Vectoring turns: reduced cost (was 3.0) — still disrupts geometry but less punitive
    ActionRow(Action.TURN_LEFT_AND_RETURN_TO_ROUTE, 1.5, "turn"),
    ActionRow(Action.TURN_RIGHT_AND_RETURN_TO_ROUTE, 1.5, "turn"),
    # Trombones: virtually free — lengthening is the natural delay tool; shortening at 1.5× cost.
    ActionRow(Action.LENGTHEN_TROMBONE, 0.01, "trombone"),
    ActionRow(Action.SHORTEN_TROMBONE, 0.15, "trombone"),
    # Waypoint skips: medium cost, irreversible route shortening (flat within tier)
    ActionRow(Action.SKIP_1_WAYPOINT, 1.5, "skip"),
    ActionRow(Action.SKIP_2_WAYPOINTS_NOT_NEXT, 1.5, "skip"),
    ActionRow(Action.SKIP_3_WAYPOINTS_NOT_NEXT, 1.5, "skip"),
    ActionRow(Action.SKIP_4_WAYPOINTS_NOT_NEXT, 1.5, "skip"),
    # Vector directly to ILS
    ActionRow(Action.VECTOR_TO_ILS, 1.0, "ils"),
)

NUM_ACTIONS = len(ACTION_TABLE)

# Speed action -> (sign, ``ActionSettings`` step size) of its speed-target delta.
SPEED_DELTAS: Mapping[Action, tuple[int, SpeedStep]] = MappingProxyType(
    {
        Action.SLOW_DOWN_SMALL: (-1, "small"),
        Action.SLOW_DOWN_MEDIUM: (-1, "medium"),
        Action.SLOW_DOWN_LARGE: (-1, "large"),
        Action.SPEED_UP_SMALL: (+1, "small"),
        Action.SPEED_UP_MEDIUM: (+1, "medium"),
    }
)


def speed_delta(action: Action, settings: ActionSettings) -> float:
    """Signed speed-target delta in knots for a speed *action*."""
    sign, step = SPEED_DELTAS[action]
    return sign * getattr(settings, f"action_speed_delta_{step}")


# Skip action -> (waypoints skipped, offset of the first skipped waypoint).
SKIP_SPECS: Mapping[Action, tuple[int, int]] = MappingProxyType(
    {
        Action.SKIP_1_WAYPOINT: (1, 0),
        Action.SKIP_2_WAYPOINTS_NOT_NEXT: (2, 1),
        Action.SKIP_3_WAYPOINTS_NOT_NEXT: (3, 1),
        Action.SKIP_4_WAYPOINTS_NOT_NEXT: (4, 1),
    }
)

TURN_ACTION_SPECS: Mapping[Action, TurnActionSpec] = MappingProxyType(
    {
        Action.TURN_LEFT_AND_RETURN_TO_ROUTE: TurnActionSpec("left", 15),
        Action.TURN_RIGHT_AND_RETURN_TO_ROUTE: TurnActionSpec("right", 15),
    }
)
