"""Tests for environment action-selection mode dispatch and per-step intake limits."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from support.env_patch import patch_env_global
from support.settings import live_env_settings

from actions.actions import Action
from atc_env.environment import ATCEnv, SelectionNoopHoldoff
from atc_env.environment_helpers import distribute_step_reward, target_agents_per_step
from atc_env.outcome_state import OutcomeState
from config.config import Config
from config.settings import settings_from_config
from simulator.commands import TimedCommandList

# Route stub with section-end marker so aircraft are not treated as past-section-end.
_ROUTE_STUB = [SimpleNamespace(waypoint_name="INLER")]


def _make_env_shell() -> ATCEnv:
    """Create an uninitialized env shell for method-level unit tests."""
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env.evaluation_mode = False
    return env


def _selected_group_index_stub(selected_groups: tuple[tuple[str, ...], ...]) -> SimpleNamespace:
    sector_order = tuple(sector for group in selected_groups for sector in group)
    sector_to_group = {sector: group for group in selected_groups for sector in group}
    sector_defs = {
        sector: {
            "polygon": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            "bbox": [0.0, 0.0, 1.0, 1.0],
        }
        for sector in sector_order
    }
    return SimpleNamespace(
        selected_groups=selected_groups,
        sector_order=sector_order,
        sector_to_group=sector_to_group,
        sector_defs=sector_defs,
    )


def _make_sector_mode_env_with_world(
    initial_cooldown: dict[str, int],
    recent_noop_holdoffs: dict[str, SelectionNoopHoldoff] | None = None,
) -> ATCEnv:
    env = _make_env_shell()
    env.agents = ["A", "B"]
    env.last_acted_agent = None
    env.consecutive_acts = {}
    env.stage_cooldown_by_agent = dict(initial_cooldown)
    env.recent_noop_holdoff_by_agent = dict(recent_noop_holdoffs or {})
    env._last_action_mask_counts = {}
    env.step_counter = 0

    class _WorldStub:
        age_in_seconds = 0.0
        arrival_targets = []

        @staticmethod
        def is_under_control(callsign: str) -> bool:
            return callsign in {"A", "B"}

        @staticmethod
        def get_aircraft_by_callsign(callsign: str):
            return SimpleNamespace(
                route=_ROUTE_STUB,
                state=SimpleNamespace(
                    position_nm_nm_ft=SimpleNamespace(
                        x=1.0 if callsign == "A" else 2.0,
                        y=0.0,
                    )
                ),
            )

    env._locked_out_agents = set()
    env._conflict_override_cache = None
    env._step_deviation_proxies = None
    env._step_selection_infr_map = None
    env._step_prev_world_all_infr = None
    env.simulator = SimpleNamespace(world=_WorldStub())  # ty: ignore[invalid-assignment]
    env.outcome_state = OutcomeState(settings_from_config().safety)
    env.lockout_threshold_s = 0.0
    env.prev_world = object()
    return env


def _patch_sector_selection_defaults(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_groups: tuple[tuple[str, ...], ...],
    grouped_candidates: dict[tuple[str, ...], list[str]],
    deviations: dict[str, float],
    infringements: dict[str, list[SimpleNamespace]] | None = None,
) -> None:
    monkeypatch.setattr(Config, "SELECTION_COOLDOWN_STEPS", 1)
    monkeypatch.setattr(Config, "SELECTION_SOFT_SUPPRESSION_DOMINANCE_MARGIN", 0.5)
    monkeypatch.setattr(Config, "SELECTION_RECENT_NOOP_HOLDOFF_STEPS", 2)
    monkeypatch.setattr(Config, "NEIGHBOR_LOOKAHEAD_MULTIPLIER", 1)
    monkeypatch.setattr(Config, "TIME_BETWEEN_ACTIONS", 1)
    monkeypatch.setattr(Config, "LOCKOUT_DEVIATION_THRESHOLD_S", 0.0)  # Disable lockout
    patch_env_global(monkeypatch, "get_sector_selection_groups", lambda _settings: selected_groups)
    patch_env_global(
        monkeypatch,
        "get_selected_group_index_cached_from_normalized",
        lambda _groups: _selected_group_index_stub(selected_groups),
    )
    patch_env_global(
        monkeypatch,
        "assign_aircraft_to_selected_groups",
        lambda _positions, _index: grouped_candidates,
    )
    patch_env_global(
        monkeypatch,
        "get_world_infringements",
        lambda *_, **__: infringements or {},
    )
    patch_env_global(
        monkeypatch,
        "proxy_map",
        lambda _prev_world: deviations,
    )
    patch_env_global(
        monkeypatch,
        "compute_stage_c_priority_score",
        lambda max_conflict, abs_deviation_s, **_kw: abs_deviation_s + max_conflict,
    )
    patch_env_global(monkeypatch, "tiny_stable_noise", lambda _agent_id: 0.0)


@pytest.mark.parametrize(
    ("configured_mode", "expected"),
    [("single", "A1"), ("sector_based", "B1")],
    ids=["single", "sector_based"],
)
def test_mode_dispatch_uses_configured_selector(
    monkeypatch: pytest.MonkeyPatch, configured_mode: str, expected: str
) -> None:
    env = _make_env_shell()
    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", configured_mode)

    def fake_single_selector(self, agents_to_choose_from=None, did_act=None):  # pylint: disable=unused-argument
        return ["A1"]

    def fake_sector_selector(self, agents_to_choose_from=None, did_act=None):  # pylint: disable=unused-argument
        return ["B1"]

    monkeypatch.setattr(ATCEnv, "_select_single_next_agents", fake_single_selector)
    monkeypatch.setattr(ATCEnv, "_select_sector_based_next_agents", fake_sector_selector)

    assert ATCEnv._get_next_agents_rr(env) == [expected]


def test_invalid_selection_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _make_env_shell()
    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", "not_a_mode")

    with pytest.raises(ValueError, match="Invalid ACTION_SELECTION_MODE"):
        ATCEnv._get_next_agents_rr(env)


def test_per_step_target_count_matches_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", "single")
    monkeypatch.setattr(Config, "MAX_ACTING_AGENTS_BASE", 1)
    assert target_agents_per_step(settings_from_config().selection) == 1

    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", "sector_based")
    monkeypatch.setattr(Config, "SECTOR_SELECTION_GROUPS", [["G1"], ["G2"], ["G3"]])
    assert target_agents_per_step(settings_from_config().selection) == 3

    monkeypatch.setattr(Config, "SECTOR_SELECTION_GROUPS", [["G1", "G2"], ["G3"]])
    assert target_agents_per_step(settings_from_config().selection) == 2


def test_sector_mode_requires_non_empty_selected_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", "sector_based")
    monkeypatch.setattr(Config, "SECTOR_SELECTION_GROUPS", [])

    with pytest.raises(ValueError, match="non-empty"):
        target_agents_per_step(settings_from_config().selection)


@pytest.mark.parametrize(
    ("allowed_to_act", "configured_groups", "expected_callsigns"),
    [
        (None, [["G1"], ["G2"]], ["A", "B"]),
        ({"B"}, [["G1"], ["G2"], ["G3"]], ["B"]),
    ],
    ids=["target_count_budget", "allowlist_filter"],
)
def test_action_intake_respects_budget_and_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    allowed_to_act: set[str] | None,
    configured_groups: list[list[str]],
    expected_callsigns: list[str],
) -> None:
    env = _make_env_shell()
    env.agents = ["A", "B", "C"]
    if allowed_to_act is not None:
        env._agents_allowed_to_act = allowed_to_act
    env.simulator = object()  # ty: ignore[invalid-assignment]
    env._route_trombone_store = object()  # ty: ignore[invalid-assignment]

    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", "sector_based")
    monkeypatch.setattr(Config, "SECTOR_SELECTION_GROUPS", configured_groups)

    interpreted_callsigns: list[str] = []

    def fake_interpret_action(action, call_sign, simulator, route_trombone_store, settings):
        _ = action, simulator, route_trombone_store, settings
        interpreted_callsigns.append(call_sign)
        return TimedCommandList(commands=[]), {}

    patch_env_global(monkeypatch, "interpret_action", fake_interpret_action)

    interpreted_actions, acting_agents, action_nums, _ = ATCEnv._interpret_actions(
        env,
        {"A": 1, "B": 2, "C": 3},
    )

    assert interpreted_actions.commands == []
    assert acting_agents == expected_callsigns
    assert interpreted_callsigns == expected_callsigns
    assert action_nums == {
        callsign: {"A": 1, "B": 2, "C": 3}[callsign] for callsign in expected_callsigns
    }


@pytest.mark.parametrize(
    (
        "initial_cooldown",
        "recent_noop_holdoffs",
        "infringements",
        "deviations",
        "expected_choice",
        "expected_cooldown",
    ),
    [
        ({"A": 1}, {}, {}, {"A": 10.0, "B": 1.0}, "A", {"A": 1}),
        (
            {},
            {"A": SelectionNoopHoldoff(remaining_steps=2, baseline_score=10.0)},
            {},
            {"A": 10.0, "B": 9.7},
            "B",
            {"B": 1},
        ),
    ],
    ids=[
        "dominant_candidate_bypasses_cooldown",
        "recent_noop_suppresses_stale_reselection",
    ],
)
def test_sector_mode_selection_applies_soft_suppression(
    monkeypatch: pytest.MonkeyPatch,
    initial_cooldown: dict[str, int],
    recent_noop_holdoffs: dict[str, SelectionNoopHoldoff],
    infringements: dict[str, list[SimpleNamespace]],
    deviations: dict[str, float],
    expected_choice: str,
    expected_cooldown: dict[str, int],
) -> None:
    env = _make_sector_mode_env_with_world(
        initial_cooldown=initial_cooldown,
        recent_noop_holdoffs=recent_noop_holdoffs,
    )
    _patch_sector_selection_defaults(
        monkeypatch,
        selected_groups=(("G1",),),
        grouped_candidates={("G1",): ["A", "B"]},
        deviations=deviations,
        infringements=infringements,
    )

    chosen = ATCEnv._select_sector_based_next_agents(env)

    assert chosen == [expected_choice]
    assert env.stage_cooldown_by_agent == expected_cooldown


def test_sector_mode_selects_up_to_one_agent_per_configured_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env_shell()
    env.agents = ["A", "B", "C"]
    env.last_acted_agent = None
    env.consecutive_acts = {}
    env.stage_cooldown_by_agent = {}
    env.recent_noop_holdoff_by_agent = {}
    env.step_counter = 0
    env._locked_out_agents = set()
    env._conflict_override_cache = None
    env._step_deviation_proxies = None
    env._step_selection_infr_map = None
    env._step_prev_world_all_infr = None

    class _WorldStub:
        age_in_seconds = 0.0
        arrival_targets = []

        @staticmethod
        def is_under_control(callsign: str) -> bool:
            return callsign in {"A", "B", "C"}

        @staticmethod
        def get_aircraft_by_callsign(callsign: str):
            x = {"A": 1.0, "B": 2.0, "C": 3.0}[callsign]
            return SimpleNamespace(
                route=_ROUTE_STUB,
                state=SimpleNamespace(position_nm_nm_ft=SimpleNamespace(x=x, y=0.0)),
            )

    env.simulator = SimpleNamespace(world=_WorldStub())  # ty: ignore[invalid-assignment]
    env.outcome_state = OutcomeState(settings_from_config().safety)
    env.lockout_threshold_s = 0.0
    env.prev_world = object()
    _patch_sector_selection_defaults(
        monkeypatch,
        selected_groups=(("G1",), ("G2",)),
        grouped_candidates={("G1",): ["A", "B"], ("G2",): ["C"]},
        deviations={"A": 1.0, "B": 5.0, "C": 2.0},
    )

    chosen = ATCEnv._select_sector_based_next_agents(env)

    assert chosen == ["B", "C"]
    assert env.stage_cooldown_by_agent == {"B": 1, "C": 1}


def test_sector_mode_leaves_outside_candidates_unselected_when_group_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env_shell()
    env.agents = ["A", "B", "C"]
    env.last_acted_agent = None
    env.consecutive_acts = {}
    env.stage_cooldown_by_agent = {}
    env.recent_noop_holdoff_by_agent = {}
    env.step_counter = 0
    env._locked_out_agents = set()
    env._conflict_override_cache = None
    env._step_deviation_proxies = None
    env._step_selection_infr_map = None
    env._step_prev_world_all_infr = None

    class _WorldStub:
        age_in_seconds = 0.0
        arrival_targets = []

        @staticmethod
        def is_under_control(callsign: str) -> bool:
            return callsign in {"A", "B", "C"}

        @staticmethod
        def get_aircraft_by_callsign(callsign: str):
            x = {"A": 1.0, "B": 2.0, "C": 3.0}[callsign]
            return SimpleNamespace(
                route=_ROUTE_STUB,
                state=SimpleNamespace(position_nm_nm_ft=SimpleNamespace(x=x, y=0.0)),
            )

    env.simulator = SimpleNamespace(world=_WorldStub())  # ty: ignore[invalid-assignment]
    env.outcome_state = OutcomeState(settings_from_config().safety)
    env.lockout_threshold_s = 0.0
    env.prev_world = object()
    _patch_sector_selection_defaults(
        monkeypatch,
        selected_groups=(("G1",), ("G2",)),
        grouped_candidates={("G1",): ["A"]},
        deviations={"A": 1.0, "B": 2.0, "C": 3.0},
    )

    chosen = ATCEnv._select_sector_based_next_agents(env)

    assert chosen == ["A"]
    assert env.last_controlled_candidate_count == 3
    assert env.last_outside_excluded_count == 2


def test_sector_mode_empty_groups_return_no_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_sector_mode_env_with_world(initial_cooldown={"A": 1})
    monkeypatch.setattr(Config, "SECTOR_SELECTION_GROUPS", [["G1"]])
    _patch_sector_selection_defaults(
        monkeypatch,
        selected_groups=(("G1",),),
        grouped_candidates={},
        deviations={"A": 1.0, "B": 5.0},
    )

    chosen = ATCEnv._select_sector_based_next_agents(env)

    assert chosen == []
    assert env.stage_cooldown_by_agent == {}
    assert env.last_controlled_candidate_count == 2
    assert env.last_outside_excluded_count == 2


def test_prepare_observations_tracks_observed_vs_allowed_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env_shell()
    env.outcome_state = OutcomeState(settings_from_config().safety)
    env.lockout_threshold_s = 0.0
    env.prev_world = object()
    env.agents = ["A", "B"]
    env.last_non_zero_action_time_s = {"A": 0.0, "B": 0.0}
    env.stage_cooldown_by_agent = {}
    env.recent_noop_holdoff_by_agent = {}
    env.consecutive_acts = {}

    monkeypatch.setattr(ATCEnv, "_get_next_agents_rr", lambda self, *_args, **_kwargs: ["A"])
    monkeypatch.setattr(
        ATCEnv,
        "_build_observations",
        lambda self, observe_list: {
            callsign: {
                "action_mask": np.array([1, 1], dtype=np.uint8)
                if callsign == "A"
                else np.array([1, 0], dtype=np.uint8),
            }
            for callsign in observe_list
        },
    )

    observations = ATCEnv._prepare_observations_and_update_agents(
        env,
        terminated={"A": False, "B": False, "__all__": False},
        truncated={"A": False, "B": False, "__all__": False},
        just_finished=[],
        enders={"B"},
        did_act=None,
    )

    assert set(observations.keys()) == {"A", "B"}
    assert env.last_observation_count == 2
    assert env.last_blocked_observation_count == 1
    assert env._last_action_mask_counts == {"A": 2, "B": 1}


def test_prepare_observations_terminal_step_preserves_previous_scheduler_counters() -> None:
    env = _make_env_shell()
    env.outcome_state = OutcomeState(settings_from_config().safety)
    env.lockout_threshold_s = 0.0
    env.prev_world = object()
    env.agents = ["A", "B"]
    env.last_non_zero_action_time_s = {"A": 0.0, "B": 0.0}
    env.last_speed_action_time_s = {"A": 0.0, "B": 0.0}
    env.stage_cooldown_by_agent = {}
    env.recent_noop_holdoff_by_agent = {}
    env.consecutive_acts = {}
    env._locked_out_agents = set()
    env.last_observation_count = 2
    env.last_blocked_observation_count = 1
    env._last_action_mask_counts = {"A": 2, "B": 1}
    env._build_observations = lambda observe_list: {callsign: {} for callsign in observe_list}  # ty: ignore[invalid-assignment]

    observations = ATCEnv._prepare_observations_and_update_agents(
        env,
        terminated={"A": False, "B": True, "__all__": True},
        truncated={"A": False, "B": False, "__all__": False},
        just_finished=["B"],
        enders={"B"},
        did_act=None,
    )

    assert observations == {"B": {}}
    assert env.last_observation_count == 2
    assert env.last_blocked_observation_count == 1
    assert env._last_action_mask_counts == {"A": 2}


@pytest.mark.parametrize(
    ("action_dict", "expected_action_nums", "expected_rewards"),
    [
        (
            {"A": int(Action.SLOW_DOWN_SMALL)},
            {"A": int(Action.SLOW_DOWN_SMALL)},
            {"A": 4.0, "B": 1.0, "C": 1.0},
        ),
        (
            {"A": int(Action.SLOW_DOWN_SMALL), "B": int(Action.DO_NOTHING)},
            {"A": int(Action.SLOW_DOWN_SMALL), "B": int(Action.DO_NOTHING)},
            {"A": 2.5, "B": 2.5, "C": 1.0},
        ),
    ],
    ids=["omission_not_acting", "explicit_do_nothing_is_acting"],
)
def test_reward_split_differs_for_omission_vs_explicit_do_nothing(
    monkeypatch: pytest.MonkeyPatch,
    action_dict: dict[str, int],
    expected_action_nums: dict[str, int],
    expected_rewards: dict[str, float],
) -> None:
    env = _make_env_shell()
    env.agents = ["A", "B", "C"]
    env.simulator = object()  # ty: ignore[invalid-assignment]
    env._route_trombone_store = object()  # ty: ignore[invalid-assignment]

    monkeypatch.setattr(Config, "ACTION_SELECTION_MODE", "sector_based")
    monkeypatch.setattr(Config, "SECTOR_SELECTION_GROUPS", [["G1"], ["G2"]])
    patch_env_global(
        monkeypatch,
        "interpret_action",
        lambda action, call_sign, simulator, route_trombone_store, settings: (
            TimedCommandList(commands=[]),
            {},
        ),
    )

    _, acting_agents, action_nums, _ = ATCEnv._interpret_actions(env, action_dict)
    rewards = distribute_step_reward(
        total_reward=6.0,
        acting_agents=set(acting_agents),
        active_agents=set(env.agents),
        global_share=0.5,
    )

    assert action_nums == expected_action_nums
    assert set(rewards) == {"A", "B", "C"}
    assert rewards == pytest.approx(expected_rewards)
    assert sum(rewards.values()) == pytest.approx(6.0)
