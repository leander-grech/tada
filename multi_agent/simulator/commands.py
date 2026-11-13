"""Models for air traffic control actions and commands"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from flight_simulator import AircraftCommand, RoutePoint, World


class SimulatorLike(Protocol):
    """Simulator surface that command builders read."""

    world: World | None

    def get_current_route(self, callsign: str) -> tuple[str, list[RoutePoint]]: ...


@dataclass
class TimedCommand:
    """Represents a command to be executed at a specific time step"""

    command: AircraftCommand | None
    action_id: int
    time_step: int = 0
    on_apply: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def to_tuple(self) -> tuple[int, AircraftCommand | None]:
        """Convert to tuple format expected by the simulator"""
        return (self.time_step, self.command)

    def apply_side_effects(self) -> None:
        """Run any deferred state updates associated with the command."""
        if self.on_apply is not None:
            self.on_apply()


@dataclass
class TimedCommandList:
    """Represents a list of timed commands"""

    commands: list[TimedCommand]

    def to_tuple_list(self) -> list[tuple[int, AircraftCommand | None]]:
        """Convert to the legacy tuple format"""
        return [cmd.to_tuple() for cmd in self.commands]

    def apply_side_effects(self) -> None:
        """Apply deferred state updates for all executed commands."""
        for cmd in self.commands:
            cmd.apply_side_effects()
