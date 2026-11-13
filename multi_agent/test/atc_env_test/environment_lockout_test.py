"""Tests for on-schedule aircraft lockout from selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from support.env_patch import patch_env_global
from support.settings import live_env_settings

from atc_env.environment import ATCEnv
from atc_env.outcome_state import OutcomeState
from config.config import Config
from config.settings import settings_from_config
from outcomes.infringement import InfringementTuple


def _make_env_shell() -> ATCEnv:
    """Create an uninitialized env shell for method-level unit tests."""
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env.evaluation_mode = False
    return env


def _make_lockout_env(
    agents: list[str],
    controlled: set[str],
    *,
    already_locked: set[str] | None = None,
) -> ATCEnv:
    """Build an env stub wired for lockout/selection tests."""
    env = _make_env_shell()
    env.agents = list(agents)
    env.last_acted_agent = None
    env.consecutive_acts = {}
    env.stage_cooldown_by_agent = {}
    env._locked_out_agents = set(already_locked) if already_locked else set()
    env._conflict_override_cache = None
    env._step_deviation_proxies = None
    env._step_selection_infr_map = None
    env._step_prev_world_all_infr = None

    # Route stub includes a section-end marker so aircraft are not treated as past-section-end.
    _route_stub = [SimpleNamespace(waypoint_name="INLER")]

    class _WorldStub:
        age_in_seconds = 100.0
        arrival_targets = []

        @staticmethod
        def is_under_control(callsign: str) -> bool:
            return callsign in controlled

        @staticmethod
        def get_aircraft_by_callsign(_callsign: str):
            return SimpleNamespace(route=_route_stub)

    env.simulator = SimpleNamespace(world=_WorldStub())  # ty: ignore[invalid-assignment]
    env.outcome_state = OutcomeState(settings_from_config().safety)
    env.lockout_threshold_s = 5.0
    env.prev_world = object()
    return env


def _patch_lockout_deps(
    monkeypatch: pytest.MonkeyPatch,
    deviations: dict[str, float],
    infringements: dict[str, list[InfringementTuple]] | None = None,
) -> None:
    """Monkeypatch external dependencies for lockout tests."""
    infr = infringements or {}
    patch_env_global(
        monkeypatch,
        "proxy_map",
        lambda _prev_world: deviations,
    )
    patch_env_global(
        monkeypatch,
        "get_world_infringements",
        lambda *_, **__: infr,
    )
    monkeypatch.setattr(Config, "LOCKOUT_DEVIATION_THRESHOLD_S", 5.0)
    monkeypatch.setattr(Config, "NEIGHBOR_LOOKAHEAD_MULTIPLIER", 1)
    monkeypatch.setattr(Config, "TIME_BETWEEN_ACTIONS", 45)


# ---------------------------------------------------------------------------
# Basic lockout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deviation", [2.0, -3.0], ids=["positive", "negative"])
def test_aircraft_below_threshold_gets_locked_out(
    monkeypatch: pytest.MonkeyPatch, deviation: float
) -> None:
    """Aircraft with |deviation| < 5s is removed from candidates (sign doesn't matter)."""
    env = _make_lockout_env(agents=["A", "B"], controlled={"A", "B"})
    _patch_lockout_deps(monkeypatch, deviations={"A": deviation, "B": 30.0})

    candidates = env._resolve_controlled_candidates(None)

    assert "A" not in candidates
    assert "B" in candidates
    assert "A" in env._locked_out_agents


def test_aircraft_at_threshold_not_locked_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aircraft exactly at the threshold is NOT locked out (< not <=)."""
    env = _make_lockout_env(agents=["A"], controlled={"A"})
    _patch_lockout_deps(monkeypatch, deviations={"A": 5.0})

    candidates = env._resolve_controlled_candidates(None)

    assert "A" in candidates
    assert "A" not in env._locked_out_agents


def test_uncontrolled_aircraft_not_locked_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aircraft not under ATC control (not yet in airspace) should not be locked."""
    env = _make_lockout_env(agents=["A", "B"], controlled={"B"})
    _patch_lockout_deps(monkeypatch, deviations={"A": 0.0, "B": 30.0})

    env._resolve_controlled_candidates(None)

    # A isn't controlled so it's excluded by the base filter, not lockout
    assert "A" not in env._locked_out_agents


# ---------------------------------------------------------------------------
# Reversible lockout
# ---------------------------------------------------------------------------


def test_lockout_releases_when_deviation_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Locked aircraft becomes selectable again after deviation clearly recovers."""
    env = _make_lockout_env(agents=["A", "B"], controlled={"A", "B"}, already_locked={"A"})
    _patch_lockout_deps(monkeypatch, deviations={"A": 50.0, "B": 30.0})

    candidates = env._resolve_controlled_candidates(None)

    assert "A" in candidates
    assert "A" not in env._locked_out_agents


# ---------------------------------------------------------------------------
# Conflict override
# ---------------------------------------------------------------------------


def test_conflict_between_locked_aircraft_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two locked-out aircraft with mutual conflict both become selectable."""
    env = _make_lockout_env(
        agents=["A", "B", "C"], controlled={"A", "B", "C"}, already_locked={"A", "B"}
    )
    _patch_lockout_deps(
        monkeypatch,
        deviations={"A": 2.0, "B": 3.0, "C": 30.0},
        infringements={
            "A": [InfringementTuple(value=0.5, timestamp=120.0, counterpart="B")],
            "B": [InfringementTuple(value=0.5, timestamp=120.0, counterpart="A")],
        },
    )

    candidates = env._resolve_controlled_candidates(None)

    assert "A" in candidates
    assert "B" in candidates
    assert "C" in candidates


def test_significant_conflict_with_non_locked_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Locked aircraft with significant conflict against unlocked counterpart gets re-enabled."""
    env = _make_lockout_env(agents=["A", "B"], controlled={"A", "B"}, already_locked={"A"})
    _patch_lockout_deps(
        monkeypatch,
        deviations={"A": 2.0, "B": 30.0},
        infringements={
            "A": [InfringementTuple(value=0.8, timestamp=120.0, counterpart="B")],
            "B": [InfringementTuple(value=0.8, timestamp=120.0, counterpart="A")],
        },
    )

    candidates = env._resolve_controlled_candidates(None)

    assert "A" in candidates
    assert "B" in candidates


def test_low_severity_conflict_with_non_locked_stays_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locked aircraft with sub-threshold conflict against unlocked counterpart stays locked."""
    env = _make_lockout_env(agents=["A", "B"], controlled={"A", "B"}, already_locked={"A"})
    _patch_lockout_deps(
        monkeypatch,
        deviations={"A": 2.0, "B": 30.0},
        infringements={
            "A": [InfringementTuple(value=0.5, timestamp=120.0, counterpart="B")],
            "B": [InfringementTuple(value=0.5, timestamp=120.0, counterpart="A")],
        },
    )

    candidates = env._resolve_controlled_candidates(None)

    assert "A" not in candidates
    assert "B" in candidates


def test_no_conflict_keeps_locked_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Locked aircraft remains locked while deviation stays below the threshold."""
    env = _make_lockout_env(agents=["A", "B"], controlled={"A", "B"}, already_locked={"A", "B"})
    _patch_lockout_deps(monkeypatch, deviations={"A": 2.0, "B": 4.0})

    candidates = env._resolve_controlled_candidates(None)

    assert candidates == []


def test_active_unknown_timing_remains_controllable(monkeypatch):
    env = _make_lockout_env(agents=["A"], controlled={"A"}, already_locked={"A"})
    _patch_lockout_deps(monkeypatch, deviations={})
    assert env._resolve_controlled_candidates(None) == ["A"]
    assert not env._locked_out_agents
