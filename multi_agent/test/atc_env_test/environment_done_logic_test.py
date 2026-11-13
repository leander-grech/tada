"""Native current-world classifications drive physical terminality and bootstrap flags."""

import json
from types import SimpleNamespace

import pytest
from flight_simulator import World
from support.settings import live_env_settings

from actions.rewards import reward_options
from atc_env.environment import ATCEnv
from atc_env.outcome_state import OutcomeState
from config.settings import settings_from_config


def make_env(world, *, threshold=60, cap=2000, final=False, planning=1):
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env._training_payload = None
    env.outcome_state = OutcomeState(settings_from_config().safety)
    env.outcome_state.begin(world)
    env.simulator = SimpleNamespace(world=world)
    env.prev_world = env.outcome_state.forecast(world, planning)
    env.agents = [t.callsign for t in env.arrival_targets]
    env.step_counter = 1
    env.max_steps_per_episode = 1 if final else 10
    env.success_time_dev_s = threshold
    env.noop_verification_max_seconds = cap
    env.reward_settings = reward_options(settings=settings_from_config().reward)
    return env


@pytest.mark.parametrize(
    "error,threshold,reason", [(45, 30, "delayed"), (45, 60, "success"), (60, 60, "success")]
)
def test_physical_completion_uses_inclusive_instance_window(native_world, error, threshold, reason):
    env = make_env(native_world(error=error), threshold=threshold)
    env.simulator.world = env.outcome_state.advance(env.simulator.world, 10000)
    term, trunc, _, actual = env._compute_dones()
    assert actual == reason
    assert term["__all__"] and not trunc["__all__"]
    assert env.outcome_state.physically_complete


def test_unknown_final_forecast_can_verify_airborne_success(native_world):
    env = make_env(native_world(error=0, start=20), final=True)
    assert env.forecast_coverage.arrivals[0].timing is None
    term, trunc, _, reason = env._compute_dones()
    assert reason == "success"
    assert not term["__all__"] and trunc["__all__"]
    assert env.last_outcome.kind == "verified_noop_success"
    assert not env.outcome_state.physically_complete


def test_unknown_final_forecast_with_short_cap_stays_unresolved(native_world):
    env = make_env(native_world(error=0), final=True, cap=1)
    term, trunc, _, reason = env._compute_dones()
    assert reason == "timeout" and env.last_outcome.kind == "unresolved"
    assert not term["__all__"] and trunc["__all__"]
    assert env.last_success_margin_bonus == 0


def test_verified_airborne_delayed_does_not_earn_physical_delayed_reason(native_world):
    env = make_env(native_world(error=100), final=True)
    term, trunc, _, reason = env._compute_dones()
    assert env.last_outcome.kind == "safe_delayed"
    assert reason == "timeout" and not term["__all__"] and trunc["__all__"]
    assert env.last_success_margin_bonus == 0


def test_completed_unknown_timing_is_physical_terminal_without_success(native_world):
    env = make_env(native_world(error=0))
    completed = env.outcome_state.advance(env.simulator.world, 10000)
    # Imported event-free completion lacks trusted history in the fresh scenario.
    env = make_env(World.from_json(completed.json()))
    term, trunc, _, reason = env._compute_dones()
    assert env.last_outcome.kind == "unresolved" and reason == "unresolved"
    assert term["__all__"] and not trunc["__all__"]
    assert env.last_success_margin_bonus == 0


def test_changed_live_state_and_reset_do_not_reuse_verification(native_world):
    env = make_env(native_world(error=0), final=True)
    assert env._compute_dones()[-1] == "success"
    assert env.last_verification is not None
    env.simulator.world = env.outcome_state.advance(env.simulator.world, 1)
    env.max_steps_per_episode = 10
    assert env.last_verification is None
    assert env._compute_dones()[-1] is None
    assert env.last_outcome.kind == "unresolved"
    env.outcome_state.begin(env.simulator.world)
    assert env.last_outcome is None and env.last_verification is None


def test_realized_exit_beats_good_forecast_and_timeout(native_world):
    world = native_world(error=0)
    data = json.loads(world.json())
    data["infringements"] = [{"AirspaceExit": {"timestamp": 0, "aircraft": "TEST001"}}]
    env = make_env(World.from_json(json.dumps(data)), final=True, planning=10000)
    term, trunc, _, reason = env._compute_dones()
    assert reason == "infringement" and env.last_outcome.kind == "infringement"
    assert term["__all__"] and not trunc["__all__"]
    assert env.last_success_margin_bonus == 0
    assert env.outcome_ledger.consume_failure()
    assert not env.outcome_ledger.consume_failure()


def test_unknown_success_margin_is_zero():
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()
    env._training_payload = None
    assert env._compute_success_margin(deviation_proxies={}, arrival_target_callsigns={"A"}) == 0


def test_non_positive_success_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        ATCEnv({"success_time_dev_s": 0.0})


def test_non_finite_success_window_is_rejected() -> None:
    """NaN would silently fail every ``abs(deviation) < window`` check."""
    for window in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            ATCEnv({"success_time_dev_s": window})


def test_dropped_actor_is_closed_as_terminated() -> None:
    """RLlib keeps acting for an aircraft this env already dropped; close it.

    Left hanging, that action makes RLlib fail the episode with
    ``MultiAgentEnvError`` once the scenario truncates.
    """
    terminated = {"A": False, "__all__": False}
    truncated = {"A": False, "__all__": False}

    ATCEnv._close_untracked_actors({"A": 0, "GONE": 3}, terminated, truncated)

    assert terminated == {"A": False, "GONE": True, "__all__": False}
    assert truncated == {"A": False, "GONE": False, "__all__": False}


def test_closing_dropped_actors_keeps_tracked_flags() -> None:
    """A tracked agent's own terminal state must survive the sweep."""
    terminated = {"A": True, "B": False, "__all__": False}
    truncated = {"A": False, "B": True, "__all__": False}

    ATCEnv._close_untracked_actors({"A": 0, "B": 1}, terminated, truncated)

    assert terminated == {"A": True, "B": False, "__all__": False}
    assert truncated == {"A": False, "B": True, "__all__": False}


def test_native_failed_mission_is_terminal_unresolved_without_fabricated_safety_event(native_world):
    data = json.loads(native_world(error=0).json())
    data["aircraft"][0]["mission"]["failed"] = True
    env = make_env(World.from_json(json.dumps(data)))
    term, trunc, _, reason = env._compute_dones()
    assert term["__all__"] and not trunc["__all__"]
    assert reason == "unresolved" and env.last_outcome.kind == "unresolved"
    assert "native_mission_failed" in env.last_outcome.reasons
    assert env.failed_target_callsigns == (env.arrival_targets[0].callsign,)
    assert not env.outcome_ledger.snapshot().failed
    assert env.outcome_ledger.snapshot().arrivals[0].timing is None
    assert env.last_success_margin_bonus == 0


def test_absent_uncompleted_target_is_unknown_not_physical_failure(native_world):
    world = native_world(error=0)
    world.aircraft = []
    env = make_env(world)
    term, trunc, _, reason = env._compute_dones()
    assert not term["__all__"] and not trunc["__all__"]
    assert reason is None and env.last_outcome.kind == "unresolved"
    assert not env.failed_target_callsigns
    assert env.last_success_margin_bonus == 0


def test_completed_live_evidence_survives_later_absent_forecast_population(native_world):
    env = make_env(native_world(error=0))
    completed = env.outcome_state.advance(env.simulator.world, 10000)
    evidence = env.outcome_ledger.snapshot().arrivals[0]
    completed.aircraft = []
    env.outcome_state.observe_live(completed)
    env.simulator.world = completed
    env.prev_world = env.outcome_state.forecast(completed, 0)
    term, trunc, _, reason = env._compute_dones()
    assert env.forecast_coverage.arrivals[0] == evidence
    assert reason == "success" and env.last_outcome.kind == "physical_success"
    assert term["__all__"] and not trunc["__all__"]


@pytest.mark.parametrize(
    "hard,exit_too", [(False, False), (True, False), (True, True), (False, True)]
)
def test_native_retained_safety_terminal_charge_matches_hard_events(hard, exit_too):
    env = ATCEnv(
        {
            "evaluation_mode": True,
            "evaluation_seeds": [4000000011],
            "max_aircraft_count": 1,
            "scenario_start_time_offset_max_s": 0,
        }
    )
    try:
        env.reset()
        assert env.simulator.world is not None
        data = json.loads(env.simulator.world.json())
        callsign = env.arrival_targets[0].callsign
        data["infringements"] = [
            {
                "LossOfSeparation": {
                    "timestamp": env.simulator.world.age_in_seconds,
                    "aircraft_pair": [callsign, "OTHER"],
                    "horizontal_separation": (2 if hard else 4) * 1852,
                    "vertical_separation": 250 * 0.3048,
                }
            }
        ]
        if exit_too:
            data["infringements"].append(
                {
                    "AirspaceExit": {
                        "timestamp": env.simulator.world.age_in_seconds,
                        "aircraft": callsign,
                    }
                }
            )
        env.simulator.world = World.from_json(json.dumps(data))
        env.outcome_state.observe_live(env.simulator.world)
        _, _, terminated, truncated, _ = env.step({})
        qualifies = hard or exit_too
        assert (env.last_outcome.kind == "infringement") is qualifies
        assert bool(terminated["__all__"]) is qualifies
        if qualifies:
            assert env.last_reward.bonus == pytest.approx(
                -env.reward_settings["success_bonus"] / env.reward_settings["count_baseline"]
            )
        else:
            # A safe verified completion may earn a positive bonus, never a failure charge.
            assert env.last_reward.bonus >= 0
        assert not env.outcome_ledger.consume_failure()
        from common.runtime_log import runtime_print

        runtime_print(
            "audit-b",
            "terminal-safety",
            hard=hard,
            exit_too=exit_too,
            kind=env.last_outcome.kind,
            terminated=terminated["__all__"],
            truncated=truncated["__all__"],
            bonus=env.last_reward.bonus,
        )
    finally:
        env.close()
