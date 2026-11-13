"""Native fixtures are isolated from every reserved campaign seed."""

import json
from pathlib import Path

import pytest
from flight_simulator import (
    Mission,
    MissionDifficulty,
    World,
    generate_random_mission,
    load_config,
    load_world,
    set_seed,
    simulate_n_next_states,
)
from support.bootstrap import ray_env_registry  # noqa: F401  (fixture registration)

from common.hashing import content_digest
from evaluation.seeds import build_reservation_manifest
from training.frozen_settings import (
    read_current_initial_preprocessing_settings,
    read_current_physics_settings,
)


@pytest.fixture(scope="session")
def native_base_json():
    seed = 4000000011
    campaign = json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
    assert seed not in {
        row["seed"]
        for group in ("validation", "heldout", "replacements")
        for row in campaign[group]
    }
    set_seed(seed)
    config = load_config("config/simulation_config", MissionDifficulty.VALIDATION_USE_CASE_2)
    mission = Mission(
        completed=False,
        failed=False,
        aircraft_horizontal_safezone_radius_nm=5,
        aircraft_vertical_safezone_radius_ft=1000,
    )
    aircraft, waypoints, airspace, targets, *_ = generate_random_mission(
        1, config, mission, False, None, MissionDifficulty.VALIDATION_USE_CASE_2
    )
    return load_world(aircraft, targets, config.aircraft_designs, airspace, waypoints).json()


@pytest.fixture
def native_world(native_base_json):
    """Use native completion age only to arrange ETA variants, never as evidence."""

    def make(*, error=0.0, start=0.0):
        data = json.loads(native_base_json)
        data["aircraft"][0]["start_time_s"] = start
        trial = World.from_json(json.dumps(data))
        finished = simulate_n_next_states(10000, 1000, trial, [], None)
        assert finished.aircraft[0].mission.completed
        data["arrival_targets"][0]["eta_s"] = finished.age_in_seconds - error
        return World.from_json(json.dumps(data))

    return make


@pytest.fixture
def config_path(tmp_path):
    """An injected reservation never shares campaign seeds or checkpoint lineage."""
    from common.native_runtime import verify_native

    native = verify_native()
    manifest = build_reservation_manifest(
        [5, 9],
        root=617,
        heldout_count=3,
        replacement_count=2,
        stratum={
            "id": "fixture",
            "airport": "TEST",
            "difficulty": "VALIDATION_USE_CASE_2",
            "aircraft_count": 3,
            "weight": 1.0,
            "scenario_config_sha256": "a" * 64,
        },
        purpose="fixture",
    )
    (tmp_path / "reserved.json").write_text(json.dumps(manifest))
    conditions = {
        "version": 1,
        "purpose": "fixture",
        "reservation_sha256": manifest["manifest_sha256"],
        "physics": read_current_physics_settings(),
        "benchmark": {
            "stratum": manifest["strata"][0],
            "episode_steps": 12,
            "start_offset_max_s": 4,
            "preprocessing": "fixture",
        },
        "baseline_source": {"purpose": "injected fixture; not a campaign parent"},
        "native_artifact": {
            key: native[key]
            for key in (
                "mode",
                "version",
                "source_identity",
                "source_provenance",
                "wheel_sha256",
                "extension_sha256",
            )
        },
    }
    conditions["conditions_sha256"] = content_digest(conditions)
    (tmp_path / "conditions.json").write_text(json.dumps(conditions))
    data = {
        "version": 1,
        "run_id": "fixture-run",
        "phase_id": "first",
        "root_seed": 987,
        "reserved_manifest": "reserved.json",
        "reserved_conditions": "conditions.json",
        "topology": {"phase_slot": 0, "env_runners": 2, "envs_per_runner": 1},
        "scenario": {
            "airport": "TEST",
            "difficulty": "VALIDATION_USE_CASE_2",
            "aircraft_count": 3,
            "config_root": ".",
            "config_sha256": "a" * 64,
            "episode_steps": 12,
            "start_offset_max_s": 4,
            "verification": "native_generation",
        },
        "thresholds": {
            "training_success_s": 30.0,
            "operational_success_s": 60.0,
            "lockout_s": 20.0,
            "auto_solve_s": read_current_initial_preprocessing_settings()[
                "early_auto_solve_target_window_s"
            ],
        },
        "features": {},
        "reward": {},
        "optimizer": {
            "learning_rate": 0.001,
            "gamma": 0.9,
            "gae_lambda": 0.8,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "grad_clip": 1.0,
            "train_batch_size": 64,
            "minibatch_size": 16,
            "epochs": 2,
        },
        "resources": {
            "backend": "cpu",
            "learner_device": "cpu",
            "num_learners": 1,
            "cpus_per_runner": 1.0,
            "cpus_per_learner": 2.0,
            "gpus_per_learner": 0.0,
        },
        "cadences": {"checkpoint_iterations": 3, "evaluation_iterations": 2},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def native_run(config_path, tmp_path):
    """Small native diagnostic, explicitly bound to the selected test artifact."""
    import torch.version
    from support.small_env import small_bgy_config

    from training.run_config import load_run_config

    data = small_bgy_config(config_path, aircraft_count=2)
    data["resources"].update(backend="cuda" if torch.version.cuda else "cpu", cpus_per_learner=1)
    data["topology"]["env_runners"] = 1
    data["execution"] = {"stop_iterations": 2, "output_root": str(tmp_path / "diagnostic")}
    data["optimizer"].update(train_batch_size=8, minibatch_size=8, epochs=1)
    data["features"] = {
        "deviation_encoding": "signed_log",
        "neighbor_selection": "risk",
        "knee_s": 17.0,
        "cap_s": 170.0,
        "dimensions": {"max_neighbors": 2, "max_waypoints": 8},
        "architecture": {"gru_hidden": 16, "mlp_hidden": 16, "max_seq_len": 2},
    }
    config_path.write_text(json.dumps(data))
    return load_run_config(config_path)
