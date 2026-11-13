"""Tests for scenario start-time offset sampling behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from flight_simulator import MissionDifficulty

from config.settings import settings_from_config
from simulator import simulator as simulator_module
from simulator.simulator import Simulator

MAX_OFFSET_S = 120


def _make_simulator() -> Simulator:
    simulator = Simulator.__new__(Simulator)
    simulator.settings = settings_from_config().simulator
    return simulator


def _patch_simulate_with_capture(monkeypatch, advanced_world: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_simulate(offset_s: int, timestep_ms: int, world: Any, commands: Any):
        captured["offset_s"] = offset_s
        captured["timestep_ms"] = timestep_ms
        captured["world"] = world
        captured["commands"] = commands
        return advanced_world

    monkeypatch.setattr(simulator_module, "simulate_n_next_states", fake_simulate)
    return captured


def test_start_offset_is_deterministic_for_seed() -> None:
    simulator = _make_simulator()

    first = simulator._sample_start_time_offset_seconds(
        seed=123456,
        max_offset_s=MAX_OFFSET_S,
    )
    second = simulator._sample_start_time_offset_seconds(
        seed=123456,
        max_offset_s=MAX_OFFSET_S,
    )

    assert first == second
    assert 0 <= first <= MAX_OFFSET_S


def test_start_offset_is_zero_when_max_is_zero() -> None:
    simulator = _make_simulator()

    eval_offset = simulator._sample_start_time_offset_seconds(
        seed=999,
        max_offset_s=0,
    )
    train_offset = simulator._sample_start_time_offset_seconds(
        seed=999,
        max_offset_s=0,
    )

    assert eval_offset == 0
    assert train_offset == 0


def test_reset_requires_seed_when_creating_world(monkeypatch) -> None:
    simulator = _make_simulator()

    class FakeRecorder:
        def init(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.setattr(simulator_module, "DataRecorder", FakeRecorder)

    with pytest.raises(ValueError, match="seed must be set when create_world=True"):
        simulator.reset(
            difficulty=MissionDifficulty.VALIDATION_USE_CASE_2,
            create_world=True,
            seed=None,
        )


def test_advance_world_with_start_offset_falls_back_when_all_missions_completed(
    monkeypatch,
) -> None:
    simulator = _make_simulator()

    original_world = SimpleNamespace(name="original")
    completed_aircraft = SimpleNamespace(mission=SimpleNamespace(completed=True))
    advanced_world = SimpleNamespace(
        arrival_targets=[SimpleNamespace(ac_callsign="TEST001")],
        aircraft=[completed_aircraft],
        get_aircraft_by_callsign=lambda _callsign: completed_aircraft,
    )

    monkeypatch.setattr(
        simulator,
        "_sample_start_time_offset_seconds",
        lambda **_kwargs: 5,
    )
    captured_call = _patch_simulate_with_capture(monkeypatch, advanced_world)

    result = simulator._advance_world_with_start_offset(
        cast(Any, original_world),
        seed=123,
        max_offset_s=MAX_OFFSET_S,
    )

    assert result is original_world
    assert captured_call["offset_s"] == 5
    assert isinstance(captured_call["timestep_ms"], int)
    assert captured_call["timestep_ms"] > 0
    assert captured_call["world"] is original_world
    assert captured_call["commands"] == []


def test_advance_world_with_start_offset_uses_advanced_world_when_targets_unfinished(monkeypatch):
    simulator = _make_simulator()

    original_world = SimpleNamespace(name="original")
    active_aircraft = SimpleNamespace(mission=SimpleNamespace(completed=False))
    advanced_world = SimpleNamespace(
        arrival_targets=[SimpleNamespace(ac_callsign="TEST001")],
        aircraft=[active_aircraft],
        get_aircraft_by_callsign=lambda _callsign: active_aircraft,
    )

    monkeypatch.setattr(
        simulator,
        "_sample_start_time_offset_seconds",
        lambda **_kwargs: 5,
    )
    captured_call = _patch_simulate_with_capture(monkeypatch, advanced_world)

    result = simulator._advance_world_with_start_offset(
        cast(Any, original_world),
        seed=123,
        max_offset_s=MAX_OFFSET_S,
    )

    assert result is advanced_world
    assert captured_call["offset_s"] == 5
    assert isinstance(captured_call["timestep_ms"], int)
    assert captured_call["timestep_ms"] > 0
    assert captured_call["world"] is original_world
    assert captured_call["commands"] == []
