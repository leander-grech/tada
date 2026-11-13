"""Models for air traffic control actions and commands"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from flight_simulator import AircraftCommand  # type: ignore[attr-defined, import-untyped]


@dataclass
class TimedCommand:
    """Represents a command to be executed at a specific time step"""

    command: Optional[AircraftCommand]
    action_id: int
    time_step: int = 0

    def to_tuple(self) -> Tuple[int, Optional[AircraftCommand]]:
        """Convert to tuple format expected by the simulator"""
        return (self.time_step, self.command)


@dataclass
class TimedCommandList:
    """Represents a list of timed commands"""

    commands: List[TimedCommand]

    def to_tuple_list(self) -> List[Tuple[int, Optional[AircraftCommand]]]:
        """Convert to simulator tuple format"""
        return [cmd.to_tuple() for cmd in self.commands]
