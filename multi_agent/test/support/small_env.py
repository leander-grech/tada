"""A one-step native BGY run and environment whose episode outcome a test can choose."""

import json
from dataclasses import replace
from pathlib import Path

from training.env import TrainingATCEnv
from training.run_config import load_run_config, scenario_config_digest


def small_bgy_config(config_path, *, aircraft_count=3, reward=None, **scenario):
    """Rewrite the injected run config to a one-step BGY scenario and return its JSON data."""
    data = json.loads(config_path.read_text())
    root = Path("config/simulation_config").resolve()
    data["scenario"].update(
        airport="BGY",
        config_root=str(root),
        config_sha256=scenario_config_digest(root),
        aircraft_count=aircraft_count,
        episode_steps=1,
        start_offset_max_s=0,
        noop_verification_max_seconds=1,
        **scenario,
    )
    if reward is not None:
        data["reward"] = reward
    config_path.write_text(json.dumps(data))
    return data


def small_bgy_run(config_path, **options):
    small_bgy_config(config_path, **options)
    return load_run_config(config_path)


def training_env(run, tmp_path, payload=None):
    """A training environment for ``run``, optionally consuming an older saved ``payload``."""
    return TrainingATCEnv(
        {
            "resolved_env": run.env_payload() if payload is None else payload,
            "expected_semantic_digest": run.semantic_digest,
            "scenario_artifact_dir": str(tmp_path / "scenarios"),
        }
    )


def force_outcome(env, **fields):
    """Replace fields of every outcome the environment classifies until the next reset."""
    classify = env.outcome_state.classify

    def forced(*args, **kwargs):
        outcome = replace(classify(*args, **kwargs), **fields)
        env.outcome_state.last_outcome = outcome
        return outcome

    env.outcome_state.classify = forced
