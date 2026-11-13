"""Tests for simulator command application side effects."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from simulator import simulator as simulator_module
from simulator.commands import TimedCommand, TimedCommandList
from simulator.simulator import Simulator


def _make_simulator(world: Any) -> Simulator:
    simulator = Simulator.__new__(Simulator)
    simulator.world = world
    return simulator


def test_apply_action_and_advance_runs_deferred_side_effects_after_sim(monkeypatch) -> None:
    """Side-effect closures must fire after simulation advances the world."""
    original_world = SimpleNamespace(name="before")
    advanced_world = SimpleNamespace(name="after")
    simulator = _make_simulator(original_world)
    applied: list[str] = []

    def fake_simulate(
        steps: int,
        timestep_ms: int,
        world: Any,
        commands: list[tuple[int, Any]],
        _recorder: Any = None,
    ) -> Any:
        # Side effects must NOT have fired yet during simulation.
        assert applied == [], "on_apply fired before simulation completed"
        return advanced_world

    monkeypatch.setattr(simulator_module, "simulate_n_next_states", fake_simulate)

    actions = TimedCommandList(
        commands=[
            TimedCommand(
                command=cast(Any, object()),
                action_id=0,
                on_apply=lambda: applied.append("done"),
            )
        ]
    )

    simulator.apply_action_and_advance(actions, time_between_actions=7)

    assert simulator.world is advanced_world
    assert applied == ["done"]
