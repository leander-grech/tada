"""Real reserved fixture environments must not confuse unfinished with punctual."""

import json
from pathlib import Path

from training.env import TrainingATCEnv
from training.run_config import load_run_config, scenario_config_digest


def test_actual_actor_critic_bytes_use_selected_encoding_after_raw_statistics(
    config_path, tmp_path
):
    import math
    from dataclasses import replace

    import numpy as np

    from atc_env.coverage_consumers import proxy_map, timing_report
    from common.runtime_log import runtime_print
    from models.observations import make_agent_observations_from_simulator
    from outcomes.evidence import TimingEvidence

    outputs = []
    reports = []
    for mode in ("linear", "signed_log"):
        data = json.loads(config_path.read_text())
        data["features"] = {
            "deviation_encoding": mode,
            "linear_scale_s": 200,
            "knee_s": 20,
            "cap_s": 180,
        }
        config_path.write_text(json.dumps(data))
        env = make_environment(config_path, tmp_path / mode, aircraft_count=3)
        try:
            env.reset()
            arrivals = env.forecast_coverage.arrivals
            values = (-20.0, 80.0, None)
            env.outcome_state.forecast_snapshot = replace(
                env.forecast_coverage,
                arrivals=tuple(
                    replace(
                        a, timing=None if value is None else TimingEvidence(value, value, "fixture")
                    )
                    for a, value in zip(arrivals, values, strict=True)
                ),
            )
            proxies = proxy_map(env.forecast_coverage)
            context = env._build_global_context(
                forecast_infringements={}, deviation_proxies=proxies, current_time_s=0
            )
            obs = make_agent_observations_from_simulator(
                env.simulator,
                [a.callsign for a in arrivals],
                {},
                proxies,
                env._route_trombone_store,
                global_context=context,
                settings=env.feature_settings,
                scales=env.settings.core.observation,
                action_settings=env.settings.core.action,
            )
            # abs(-20,80) -> mean 50, max 80, population std 30, not std(signed)=50.
            expected_stats = np.array(
                [0.25, 0.4, 0.15]
                if mode == "linear"
                else [math.log10(3.5), math.log10(5), math.log10(2.5)],
                dtype=np.float32,
            )
            expected_actor = np.array(
                [-0.1, 0.4, 0] if mode == "linear" else [-math.log10(2), math.log10(5), 0],
                dtype=np.float32,
            )
            for index, arrival in enumerate(arrivals):
                packed = obs[arrival.callsign]["observations"]
                scalar, critic = packed["aircraft_scalar"], packed["global_context"]
                assert scalar.dtype == critic.dtype == np.float32
                assert scalar[5].tobytes() == expected_actor[index].tobytes()
                assert critic[[5, 6, 8]].tobytes() == expected_stats.tobytes()
                assert scalar[8] == (values[index] is not None)
                assert critic[9] == 0.5 and critic[11] == np.float32(2 / 3)
                assert np.all((critic >= 0) & (critic <= 1))
            outputs.append(obs)
            reports.append(timing_report(env.forecast_coverage))
            runtime_print(
                "FeatureOptions",
                "actual bytes",
                mode=mode,
                actor=expected_actor.tolist(),
                critic=expected_stats.tolist(),
            )
        finally:
            env.close()
    assert reports[0] == reports[1]
    assert reports[0]["exact_mean_absolute_error_s"] == 50
    for callsign, first in outputs[0].items():
        second = outputs[1][callsign]
        for name, value in first["observations"].items():
            other = second["observations"][name]
            channels = (
                [i for i in range(len(value)) if i != 5]
                if name == "aircraft_scalar"
                else [i for i in range(len(value)) if i not in (5, 6, 8)]
                if name == "global_context"
                else slice(None)
            )
            assert value[channels].tobytes() == other[channels].tobytes()


def make_environment(config_path, tmp_path, *, aircraft_count=1, episode_steps=1):
    data = json.loads(config_path.read_text())
    root = Path("config/simulation_config").resolve()
    data["scenario"].update(
        airport="BGY",
        config_root=str(root),
        config_sha256=scenario_config_digest(root),
        aircraft_count=aircraft_count,
        episode_steps=episode_steps,
        start_offset_max_s=0,
        noop_verification_max_seconds=1,
    )
    config_path.write_text(json.dumps(data))
    resolved = load_run_config(config_path)
    campaign = json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
    reserved = {
        row["seed"]
        for group in ("validation", "heldout", "replacements")
        for row in campaign[group]
    }
    assert not resolved.reserved_seeds & reserved
    env = TrainingATCEnv(
        {
            "resolved_env": resolved.env_payload(),
            "expected_semantic_digest": resolved.semantic_digest,
            "scenario_artifact_dir": str(tmp_path / "scenarios"),
        }
    )
    from copy import deepcopy

    assert env._training_stream is not None
    assert deepcopy(env._training_stream).next_seed() not in reserved
    return env


def test_one_step_unfinished_arrival_is_not_success(config_path, tmp_path):
    env = make_environment(config_path, tmp_path)
    env.reset()
    assert not env._locked_out_agents, "unknown arrival timing must remain controllable"
    _, _, terminated, truncated, _ = env.step({})
    assert not terminated["__all__"]
    assert truncated["__all__"]
    assert env.last_end_reason == "timeout"
    assert env.last_success_margin_bonus == 0
    assert env.last_reward.deviation == 0
    assert env.last_outcome.kind == "unresolved"


def test_actual_unknown_observations_match_space_and_expose_coverage(config_path, tmp_path):
    env = make_environment(config_path, tmp_path)
    observations, _ = env.reset()
    assert not env._action_history, "unknown initial timing cannot justify automatic actions"
    assert env.raw_initial_deviation is None, "unknown numerical arrival total is null, not zero"
    assert env.raw_initial_timing_report["unknown_count"] == 1
    for callsign, observation in observations.items():
        scalar = observation["observations"]["aircraft_scalar"]
        context = observation["observations"]["global_context"]
        assert scalar.shape == (9,)
        assert context.shape == (12,)
        assert scalar[5] == 0 and scalar[-1] == 0
        assert context[-1] == 0
        assert env.get_observation_space(callsign).contains(observation)


def test_instance_reward_and_feature_settings_drive_actual_consumers(
    config_path, tmp_path, monkeypatch
):
    import math
    from dataclasses import replace

    import pytest

    from config.config import Config
    from outcomes.evidence import TimingEvidence
    from simulator.commands import TimedCommandList

    before = (Config.TIME_DEVIATION_MAX, Config.MAX_WAYPOINTS, Config.REWARD_MAX_FOR_ZERO_DELAY)
    outputs = []
    for index, (scale, coefficient, waypoints) in enumerate(((200, 6, 7), (400, 15, 11))):
        data = json.loads(config_path.read_text())
        data["features"] = {
            "linear_scale_s": scale,
            "deviation_encoding": "linear" if index == 0 else "signed_log",
            "knee_s": 37,
            "cap_s": 740,
            "dimensions": {"max_waypoints": waypoints},
        }
        data["reward"] = {"max_for_zero_delay": coefficient, "count_baseline": 1}
        config_path.write_text(json.dumps(data))
        env = make_environment(config_path, tmp_path / str(index))
        native_forecast = env.outcome_state.forecast
        values = iter((100, None, 200, 100))

        def forecast(world, steps):
            result = native_forecast(world, steps)
            value = next(values)
            evidence = (
                None if value is None else TimingEvidence(value, value, "adversarial_forecast")
            )
            snapshot = env.forecast_coverage
            env.outcome_state.forecast_snapshot = replace(
                snapshot, arrivals=tuple(replace(a, timing=evidence) for a in snapshot.arrivals)
            )
            return result

        monkeypatch.setattr(env.outcome_state, "forecast", forecast)
        obs, _ = env.reset()
        observation = next(iter(obs.values()))["observations"]
        expected = 100 / scale if index == 0 else math.log1p(100 / 37) / math.log(21)
        assert observation["aircraft_scalar"][5] == pytest.approx(expected)
        assert observation["aircraft_scalar"][-1] == 1
        assert observation["flight_plans_rel"].shape[0] == waypoints
        rewards = [
            env._simulate_and_reward(TimedCommandList(commands=[]))[0].deviation for _ in range(3)
        ]
        assert rewards == pytest.approx([0, -coefficient, coefficient])
        assert sum(rewards) <= 0
        assert len(env.arrival_targets) == env.initial_agent_count == 1
        outputs.append(observation["aircraft_scalar"][5])
    assert outputs[0] != outputs[1]
    assert before == (
        Config.TIME_DEVIATION_MAX,
        Config.MAX_WAYPOINTS,
        Config.REWARD_MAX_FOR_ZERO_DELAY,
    )


def test_joint_timing_credit_conserves_emitted_rewards_when_recipients_change(
    config_path, tmp_path, monkeypatch
):
    from dataclasses import replace

    import pytest

    from outcomes.evidence import TimingEvidence
    from simulator.commands import TimedCommandList

    data = json.loads(config_path.read_text())
    data["reward"] = {
        "max_for_zero_delay": 6,
        "max_for_zero_infringement": 0,
        "count_baseline": 2,
        "global_share": 0.4,
    }
    config_path.write_text(json.dumps(data))
    env = make_environment(config_path, tmp_path, aircraft_count=2, episode_steps=10)
    native_forecast = env.outcome_state.forecast
    values = iter((100, 50, None, 100))

    def forecast(world, steps):
        result = native_forecast(world, steps)
        value = next(values)
        evidence = None if value is None else TimingEvidence(value, value, "adversarial_forecast")
        env.outcome_state.forecast_snapshot = replace(
            env.forecast_coverage,
            arrivals=tuple(replace(a, timing=evidence) for a in env.forecast_coverage.arrivals),
        )
        return result

    monkeypatch.setattr(env.outcome_state, "forecast", forecast)
    env.reset()
    callsigns = sorted(t.callsign for t in env.arrival_targets)
    assert len(callsigns) == 2
    credits = []
    for actor, expected in ((callsigns[0], 3), (callsigns[0], 0), (callsigns[1], -3)):
        monkeypatch.setattr(
            env, "_interpret_actions", lambda _: (TimedCommandList(commands=[]), [actor], {}, {})
        )
        _, rewards, _, _, _ = env.step({})
        assert env.last_reward.deviation == pytest.approx(expected)
        assert sum(rewards.values()) == pytest.approx(expected)
        assert rewards[actor] == pytest.approx(0.8 * expected)
        assert rewards[next(a for a in callsigns if a != actor)] == pytest.approx(0.2 * expected)
        credits.append(env.last_reward.deviation)
    assert sum(credits) == 0
    assert len(env.timing_credit.targets) == 2


def test_simultaneous_realized_los_and_exit_apply_existing_failure_bonus_once(
    config_path, tmp_path
):
    import pytest
    from flight_simulator import World

    env = make_environment(config_path, tmp_path)
    env.reset()
    data = json.loads(env.simulator.world.json())
    callsign = env.arrival_targets[0].callsign
    age = data["age"] if "age" in data else env.simulator.world.age_in_seconds
    data["infringements"] = [
        {"AirspaceExit": {"timestamp": age, "aircraft": callsign}},
        {
            "LossOfSeparation": {
                "timestamp": age,
                "aircraft_pair": [callsign, "OTHER"],
                "horizontal_separation": 1,
                "vertical_separation": 1,
            }
        },
    ]
    env.simulator.world = World.from_json(json.dumps(data))
    env.outcome_state.observe_live(env.simulator.world)
    _, _, terminated, truncated, _ = env.step({})
    assert terminated["__all__"] and not truncated["__all__"]
    assert env.last_outcome.kind == "infringement"
    assert (
        len(env.outcome_ledger.snapshot().exit_events)
        == len(env.outcome_ledger.snapshot().separation_events)
        == 1
    )
    assert env.last_reward.bonus == pytest.approx(
        -env.reward_settings["realized_failure_cost"] / env.reward_settings["count_baseline"]
    )
    env.step({})
    assert env.last_reward.bonus == 0


def test_critic_uses_known_only_magnitude_and_proved_early_denominators(config_path, tmp_path):
    from dataclasses import replace

    import pytest

    from atc_env.coverage_consumers import proxy_map
    from outcomes.evidence import TimingEvidence

    env = make_environment(config_path, tmp_path, aircraft_count=2)
    env.reset()
    arrivals = env.forecast_coverage.arrivals
    env.outcome_state.forecast_snapshot = replace(
        env.forecast_coverage,
        arrivals=(
            replace(arrivals[0], timing=TimingEvidence(-100, -100, "exact")),
            replace(arrivals[1], timing=None),
        ),
    )
    context = env._build_global_context(
        forecast_infringements={},
        deviation_proxies=proxy_map(env.forecast_coverage),
        current_time_s=0,
    )
    assert context[5] == pytest.approx(100 / env.feature_settings.linear_scale_s)
    assert context[8] == 0
    assert context[9] == 1
    assert context[-1] == 0.5
    env.outcome_state.forecast_snapshot = replace(
        env.forecast_coverage,
        arrivals=(
            replace(arrivals[0], timing=TimingEvidence(-100, 20, "interval")),
            replace(arrivals[1], timing=None),
        ),
    )
    context = env._build_global_context(
        forecast_infringements={},
        deviation_proxies=proxy_map(env.forecast_coverage),
        current_time_s=0,
    )
    assert context[9] == 0, "sign-ambiguous interval contributes no proved-early evidence"
    assert context[-1] == 0.5


def test_reset_damping_uses_instance_horizons_for_first_and_repeated_actions(config_path, tmp_path):
    from actions.rewards import compute_action_damping_penalty, compute_speed_damping_penalty
    from config.config import Config

    data = json.loads(config_path.read_text())
    data["reward"] = {
        "action_damping_steps": 2 * Config.ACTION_DAMPING_THRESHOLD_STEPS,
        "speed_damping_steps": 2 * Config.SPEED_DAMPING_THRESHOLD_STEPS,
    }
    config_path.write_text(json.dumps(data))
    env = make_environment(config_path, tmp_path)
    env.reset()
    agent = env.agents[0]
    now = float(env.simulator.world.age_in_seconds)
    for action, history, penalty, update in (
        (
            6,
            env.last_non_zero_action_time_s,
            compute_action_damping_penalty,
            env._update_last_non_zero_action_times,
        ),
        (
            1,
            env.last_speed_action_time_s,
            compute_speed_damping_penalty,
            env._update_last_speed_action_times,
        ),
    ):
        assert (
            penalty(
                {agent: action},
                history,
                now,
                options=env.reward_settings,
                time_between_actions=env.settings.env.time_between_actions,
            )
            == 0
        )
        update({agent: action}, now)
        assert (
            penalty(
                {agent: action},
                history,
                now,
                options=env.reward_settings,
                time_between_actions=env.settings.env.time_between_actions,
            )
            < 0
        )
