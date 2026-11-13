"""Narrow scenario ownership interface for the native simulator adapter."""

from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from flight_simulator import World


class ScenarioOutcomeOwner(Protocol):
    @property
    def current_identity(self) -> tuple[int, str | None]: ...

    def begin(self, world: "World") -> None: ...

    def advance(self, world: "World", steps: int, commands=()) -> "World": ...

    def advance_next(self, world: "World", dt_ms: int, commands=()) -> "World": ...

    def fork(self) -> Self: ...

    def commit(self, branch: Self, accepted_world: "World") -> None: ...
