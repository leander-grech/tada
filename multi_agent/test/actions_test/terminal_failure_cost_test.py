"""Independent realized failure cost reaches native environment step rewards."""

import json
from pathlib import Path

import pytest
from flight_simulator import World

from training.env import TrainingATCEnv
from training.run_config import load_run_config, scenario_config_digest


@pytest.mark.parametrize("success_bonus", [1.25, 19.0])
@pytest.mark.parametrize("aircraft_count", [2, 3])
def test_one_joint_failure_cost_conserves_step_and_episode_rewards(
    config_path, tmp_path, success_bonus, aircraft_count
):
    data = json.loads(config_path.read_text())
    data["reward"] = {
        "success_bonus": success_bonus,
        "realized_failure_cost": 7.5,
        "count_baseline": 2.0,
        "global_share": 0.4,
    }
    config_path.write_text(json.dumps(data))
    root = Path("config/simulation_config").resolve()
    data["scenario"].update(
        airport="BGY",
        config_root=str(root),
        config_sha256=scenario_config_digest(root),
        aircraft_count=aircraft_count,
        episode_steps=1,
        start_offset_max_s=0,
        noop_verification_max_seconds=1,
    )
    config_path.write_text(json.dumps(data))
    run = load_run_config(config_path)
    env = TrainingATCEnv(
        {
            "resolved_env": run.env_payload(),
            "expected_semantic_digest": run.semantic_digest,
            "scenario_artifact_dir": str(tmp_path / "scenarios"),
        }
    )
    try:
        env.reset()
        assert env.simulator.world is not None
        data = json.loads(env.simulator.world.json())
        callsign = env.arrival_targets[0].callsign
        data["infringements"] = [
            {"AirspaceExit": {"timestamp": 0, "aircraft": callsign}},
            {
                "LossOfSeparation": {
                    "timestamp": 0,
                    "aircraft_pair": [callsign, env.arrival_targets[1].callsign],
                    "horizontal_separation": 1,
                    "vertical_separation": 1,
                }
            },
        ]
        env.simulator.world = World.from_json(json.dumps(data))
        env.outcome_state.observe_live(env.simulator.world)
        _, rewards, terminated, _, _ = env.step({})
        assert terminated["__all__"]
        assert env.last_reward.bonus == pytest.approx(-7.5 * aircraft_count / 2)
        assert sum(rewards.values()) == pytest.approx(env.last_reward.total())
        assert env.cumulative_reward == pytest.approx(sum(rewards.values()))
        assert sum(env.episode_reward_components.values()) == pytest.approx(env.cumulative_reward)
        env.step({})
        assert env.last_reward.bonus == 0
    finally:
        env.close()
