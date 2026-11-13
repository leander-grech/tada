"""Explicit campaign identity and native MXP generation, independent of defaults."""

import json
from pathlib import Path

import pytest

from common.hashing import content_digest
from config.settings import settings_from_config
from evaluation.benchmark.contract import procedure_contract, materialize_manifest
from evaluation.benchmark.worlds import environment_options, validate_development_manifest
from training.reservations import validate_reservation_manifest, validate_reserved_conditions


def test_mxp_campaign_selection_transports_difficulty_and_root():
    contract = procedure_contract("development", campaign="mxp")
    options = environment_options(contract, [123])
    assert options["difficulty"] == "VALIDATION_USE_CASE_1"
    assert (
        Path(options["simulation_config_path"].simulation_config_root).name
        == "simulation_config_mxp"
    )
    assert (
        contract.compatibility["reservation_sha256"]
        != procedure_contract("development").compatibility["reservation_sha256"]
    )


@pytest.mark.parametrize("count", [10, 15, 20])
def test_native_mxp_generation_replay_and_count(tmp_path, count):
    contract = procedure_contract("development", campaign="mxp", aircraft_count=count)
    reservation = json.loads(Path("config/contracts/reserved-mxp-v1.json").read_text())
    seed = reservation["validation"][0]["seed"]
    path = materialize_manifest(contract, [seed], tmp_path / "inputs.json")
    manifest = json.loads(path.read_text())
    validate_development_manifest(manifest)
    from actions.action_mask import detect_airport_from_route
    from config.config import Config
    from evaluation.benchmark.worlds import MaterializingEnv, validate_replay

    defaults = (Config.DEFAULT_EVALUATION_DIFFICULTY, Config.SIMULATOR_CONFIG_ROOT)
    env = MaterializingEnv(environment_options(contract, [seed]))
    try:
        observations, _ = env.reset()
        assert env.simulator.world is not None
        assert {
            detect_airport_from_route(
                a.route, airports=settings_from_config().action.custom_action_mask_by_airport
            )
            for a in env.simulator.world.aircraft
        } == {"MXP"}
        assert {
            r["kind"]
            for r in manifest["worlds"][0]["effective_controller"]["route_store"]["routes"].values()
        } == {"BackAndForthSection"}
        airport_mask = Config.CUSTOM_ACTION_MASK_BY_AIRPORT["MXP"]
        assert airport_mask is not None
        for observation in observations.values():
            assert all(
                not enabled or permitted
                for enabled, permitted in zip(
                    observation["action_mask"],
                    airport_mask,
                    strict=True,
                )
            )
        assert defaults == (Config.DEFAULT_EVALUATION_DIFFICULTY, Config.SIMULATOR_CONFIG_ROOT)
        assert env._airport == "MXP"
        assert str(env.difficulty).endswith("VALIDATION_USE_CASE_1")
        assert len(env.arrival_targets) == count
        validate_replay(manifest["worlds"][0], env.benchmark_world)
    finally:
        env.close()


def test_mxp_reservation_rejects_rehashed_other_capacity():
    data = json.loads(Path("config/contracts/reserved-mxp-v1.json").read_text())
    data["strata"][0]["aircraft_count"] = 15
    data.pop("manifest_sha256")
    data["manifest_sha256"] = content_digest(data)
    with pytest.raises(ValueError):
        validate_reservation_manifest(data)


def test_mxp_conditions_reject_wrong_lineage():
    manifest = json.loads(Path("config/contracts/reserved-mxp-v1.json").read_text())
    conditions = json.loads(Path("config/contracts/bgy/conditions-v2.json").read_text())
    conditions["reservation_sha256"] = manifest["manifest_sha256"]
    conditions["benchmark"]["stratum"] = manifest["strata"][0]
    conditions.pop("conditions_sha256")
    conditions["conditions_sha256"] = content_digest(conditions)
    with pytest.raises(ValueError):
        validate_reserved_conditions(conditions, manifest)


@pytest.mark.parametrize("role,count", [("heldout", 15), ("replacements", 20), ("development", 12)])
def test_density_extension_only_declared_development_diagnostics(role, count):
    with pytest.raises(ValueError):
        procedure_contract(role, campaign="mxp", aircraft_count=count)


@pytest.mark.parametrize(
    "field,value",
    [
        ("airport", "BGY"),
        ("difficulty", "VALIDATION_USE_CASE_2"),
        ("config_sha256", "0" * 64),
    ],
)
def test_campaign_training_rejects_mixed_scenario(tmp_path, field, value):
    from training.run_config import load_run_config

    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            {
                "base": str(Path("config/contracts/mxp-base.json").resolve()),
                "patch": {"scenario": {field: value}},
            }
        )
    )
    with pytest.raises(ValueError, match="campaign|stratum"):
        load_run_config(path)


def test_authenticated_density_binding_rejects_mixed_inputs():
    from evaluation.benchmark.binding import benchmark_aircraft_count
    from training.run_config import load_run_config

    run = load_run_config(Path("config/contracts/mxp-base.json"))
    contract = procedure_contract("development", campaign="mxp", aircraft_count=20)
    assert benchmark_aircraft_count(run.env_payload(), contract.compatibility) == 20
    for section, field, value in [
        ("scenario", "difficulty", "VALIDATION_USE_CASE_2"),
        ("scenario", "airport", "BGY"),
        ("scenario", "config_root", "config/simulation_config"),
        ("scenario", "aircraft_count", 15),
        ("reserved_manifest", "manifest_sha256", "0" * 64),
        ("reserved_conditions", "conditions_sha256", "0" * 64),
    ]:
        payload = json.loads(json.dumps(run.env_payload()))
        payload[section][field] = value
        with pytest.raises(ValueError):
            benchmark_aircraft_count(payload, contract.compatibility)
    forged = json.loads(json.dumps(contract.compatibility))
    forged["strata"][0]["aircraft_count"] = 15
    with pytest.raises(ValueError):
        benchmark_aircraft_count(run.env_payload(), forged)
