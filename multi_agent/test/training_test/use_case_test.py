"""A run-config use case selects one consistent BGY or MXP world and reservation set."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.benchmark.binding import benchmark_aircraft_count
from evaluation.benchmark.use_cases import procedure_selection, selection
from evaluation.benchmark.contract import procedure_contract
from evaluation.checkpoint_adapter import bind_use_case
from training.checkpoint_boundary import CheckpointBoundaryError
from training.phase import PhaseTransition, apply_sets
from training.run_config import load_run_config, reseal, scenario_config_digest

BASE = Path("config/contracts/mxp-base.json").resolve()


def _load(tmp_path, scenario):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"base": str(BASE), "patch": {"scenario": scenario}}))
    return load_run_config(path)


@pytest.fixture
def mxp():
    return load_run_config(BASE)


def test_absent_use_case_is_mxp_with_identical_digest(tmp_path, mxp):
    assert _load(tmp_path, {"use_case": "mxp"}).semantic_digest == mxp.semantic_digest


def test_bgy_selects_its_world_reservation_and_config_root(tmp_path, mxp):
    bgy = _load(tmp_path, {"use_case": "bgy"})
    assert (bgy.scenario["airport"], bgy.scenario["difficulty"]) == (
        "BGY",
        "VALIDATION_USE_CASE_2",
    )
    root = Path(bgy.scenario["config_root"])
    assert root.name == "simulation_config_bgy"
    assert scenario_config_digest(root) == bgy.scenario["config_sha256"]
    assert bgy.reserved_manifest["manifest_sha256"] != mxp.reserved_manifest["manifest_sha256"]
    assert bgy.semantic_digest != mxp.semantic_digest


def test_unknown_use_case_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="use case"):
        _load(tmp_path, {"use_case": "lin"})


def test_mxp_to_bgy_phase_passes_the_transition_check(tmp_path, mxp):
    requested = _load(tmp_path, {"use_case": "bgy", "min_aircraft_count": 1})
    requested, changes = apply_sets(
        replace(requested, phase_id="pms"), ["use_case=bgy", "min_aircraft_count=1"]
    )
    transition = PhaseTransition.between(mxp, requested, changes)
    assert transition.target_digest == requested.semantic_digest
    with pytest.raises(CheckpointBoundaryError):
        PhaseTransition.between(
            mxp, requested, {"min_aircraft_count": changes["min_aircraft_count"]}
        )


def test_phase_rebinding_equals_loading_the_use_case(tmp_path, mxp):
    phased, _ = apply_sets(mxp, ["use_case=bgy"])
    assert phased.semantic_digest == _load(tmp_path, {"use_case": "bgy"}).semantic_digest
    assert bind_use_case(mxp, "bgy").semantic_digest == phased.semantic_digest
    assert bind_use_case(mxp, "mxp") is mxp


def test_switch_rejects_different_physics(tmp_path, mxp):
    requested, changes = apply_sets(replace(mxp, phase_id="pms"), ["use_case=bgy"])
    conditions = json.loads(json.dumps(requested.reserved_conditions))
    conditions["physics"]["changed"] = True
    tampered = reseal(requested, reserved_conditions=conditions)
    with pytest.raises(CheckpointBoundaryError, match="physics"):
        PhaseTransition.between(mxp, tampered, changes)


@pytest.mark.parametrize("campaign", ["mxp", "bgy"])
@pytest.mark.parametrize("count", [10, 15, 20])
def test_development_density_selection_round_trips(campaign, count):
    contract = procedure_contract("development", campaign=campaign, aircraft_count=count)
    assert contract.compatibility.get("campaign_selection") == procedure_selection(campaign, count)
    assert selection(contract.compatibility) == (campaign, count)


def test_twenty_aircraft_bgy_checkpoint_binds_to_bgy_development(tmp_path):
    bgy = _load(tmp_path, {"use_case": "bgy", "aircraft_count": 20})
    contract = procedure_contract("development", campaign="bgy", aircraft_count=20)
    assert benchmark_aircraft_count(bgy.env_payload(), contract.compatibility) == 20
