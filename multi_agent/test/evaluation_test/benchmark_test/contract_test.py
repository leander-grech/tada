"""Frozen protocol identity and role authorization."""

import json

import pytest

from common.hashing import content_digest
from evaluation.benchmark.access import authorize_development
from evaluation.benchmark.contract import (
    BenchmarkContract,
    attempt_key,
    attempt_rng_id,
    attempt_rng_ids,
)


def test_rng_is_world_attempt_domain_identity_not_schedule():
    keys = [("w1", 0, "policy"), ("w2", 0, "policy"), ("w1", 1, "policy"), ("w1", 0, "frame")]

    def derive(order):
        return {key: attempt_rng_id("benchmark", *key) for key in order}

    assert derive(keys) == derive(reversed(keys))
    assert len(set(derive(keys).values())) == len(keys)
    with pytest.raises(ValueError):
        attempt_rng_id("benchmark", "w", -1, "policy")


def test_contract_accepts_different_method_and_rejects_protocol_change():
    compatibility = {
        "threshold_s": 60,
        "verification_cap_s": 6000,
        "strata": [{"id": "common", "count": 2, "weight": 1.0}],
    }
    left = BenchmarkContract(compatibility, {"checkpoint_sha256": "a"})
    right = BenchmarkContract(compatibility, {"checkpoint_sha256": "b"})
    assert left.compatibility_sha256 == right.compatibility_sha256
    left.require_compatible(right)
    with pytest.raises(ValueError):
        left.require_compatible(
            BenchmarkContract({**compatibility, "verification_cap_s": 5000}, {})
        )
    with pytest.raises(ValueError):
        BenchmarkContract({**compatibility, "threshold_s": 30}, {})


def test_development_authorization_uses_frozen_seed_roles_not_labels():
    reservation = json.loads(open("config/contracts/bgy/reserved-v1.json").read())
    published = [r["seed"] for r in reservation["validation"]]
    authorize_development(published)
    for role in ("heldout", "replacements"):
        with pytest.raises(ValueError, match="development"):
            authorize_development([reservation[role][0]["seed"]])
    with pytest.raises(ValueError):
        authorize_development(published + [published[0]])


@pytest.mark.parametrize("campaign", ["mxp", "bgy"])
def test_supplied_cohort_admits_listed_seeds_but_never_reserved_identities(
    campaign, monkeypatch, tmp_path
):
    from evaluation.benchmark import supplied
    from evaluation.benchmark.use_cases import procedure_inputs

    reservation, _ = procedure_inputs(campaign)
    roles = {role: [r["seed"] for r in reservation[role]] for role in ("heldout", "replacements")}
    reserved = {r["seed"] for role in ("validation", *roles) for r in reservation[role]}
    external = next(seed for seed in range(1, 1000) if seed not in reserved)
    with pytest.raises(ValueError, match="development"):
        authorize_development([external], campaign=campaign)

    path = tmp_path / "supplied.csv"
    rows = [external, roles["heldout"][0], roles["replacements"][0]]
    path.write_text("seed,source\n" + "".join(f"{seed},topup\n" for seed in rows))
    monkeypatch.setattr(supplied, "SUPPLIED_SEEDS", path)
    authorize_development([external], campaign=campaign)
    for seeds in roles.values():
        with pytest.raises(ValueError, match="development"):
            authorize_development([seeds[0]], campaign=campaign)


@pytest.mark.parametrize("campaign", ["mxp", "bgy"])
def test_committed_supplied_cohort_is_authorized_for_development(campaign):
    from evaluation.benchmark.supplied import supplied_seeds

    authorize_development(list(supplied_seeds()), campaign=campaign)


def test_native_materialization_and_replay_bind_effective_route_state(tmp_path):
    from evaluation.benchmark.contract import (
        procedure_contract,
        materialize_manifest,
        validate_manifest,
    )
    from evaluation.benchmark.worlds import MaterializingEnv, environment_options, validate_replay

    contract = procedure_contract("development")
    reservation = json.loads(open("config/contracts/bgy/reserved-v1.json").read())
    # Published input only: this test never executes policy or verifier outcomes.
    seed = reservation["validation"][0]["seed"]
    path = materialize_manifest(contract, [seed], tmp_path / "inputs.json")
    manifest = validate_manifest(json.loads(path.read_text()))
    env = MaterializingEnv(environment_options(contract, [seed]))
    try:
        env.reset()
        expected = manifest["worlds"][0]
        validate_replay(expected, env.benchmark_world)
        changed = json.loads(json.dumps(env.benchmark_world))
        changed["effective_controller"]["route_store"]["exclusion_polygon"] = []
        with pytest.raises(ValueError, match="world"):
            validate_replay(expected, changed)
        changed = json.loads(path.read_text())
        changed["worlds"][0]["generation_attempts"] += 1
        with pytest.raises(ValueError):
            validate_manifest(changed)
        with pytest.raises(ValueError):
            materialize_manifest(contract, [seed, seed], tmp_path / "duplicates.json")
    finally:
        env.close()


def test_attempt_evidence_rejects_scalar_and_clusters_missing_attempts():
    from evaluation.benchmark.evidence import AttemptEvidence, summarize_world

    with pytest.raises(ValueError, match="canonical"):
        AttemptEvidence.from_dict({"score": 1})
    summary = summarize_world([], k=3)
    assert summary["attempt_count"] == 3
    assert summary["missing_count"] == 3
    assert summary["any_attempt_hazard"] is None
    assert summary["selected_candidate_hazard"] is None
    assert summary["any_eligible_success"] is False


def test_development_gate_rejects_relabelled_heldout_before_checkpoint_loading(tmp_path):
    from evaluation.benchmark.contract import procedure_contract
    from evaluation.benchmark.worlds import validate_development_manifest

    reservation = json.loads(open("config/contracts/bgy/reserved-v1.json").read())
    for role in ("heldout", "replacements"):
        world = {"seed": reservation[role][0]["seed"], "raw_world": {}}
        world["world_id"] = content_digest(world)
        contract = procedure_contract("development")
        manifest = {
            "contract": contract.to_dict(),
            "compatibility_sha256": contract.compatibility_sha256,
            "worlds": [world],
        }
        manifest["manifest_sha256"] = content_digest(manifest)
        with pytest.raises(ValueError, match="development"):
            validate_development_manifest(manifest)


def test_benchmark_sampling_distinguishes_primary_and_pass_at_one():
    from evaluation.benchmark.evaluate import inference_protocol

    assert inference_protocol(k=1, explore=False)["label"] == "deterministic-primary"
    assert inference_protocol(k=1, explore=True)["label"] == "stochastic-pass@1"
    with pytest.raises(ValueError):
        inference_protocol(k=2, explore=False)


def test_empty_target_evidence_is_unknown_even_if_flagged_complete():
    from atc_env.completed_scenario import CompletedScenarioResult
    from evaluation.benchmark.evidence import AttemptEvidence, summarize_world
    from outcomes.ledger import OutcomeSnapshot
    from outcomes.verification import classify_outcome

    ledger = OutcomeSnapshot((), (), (), (), False)
    result = CompletedScenarioResult(
        attempt_key("benchmark", "world", "deterministic", 0, "a"),
        classify_outcome(ledger, None, threshold_s=60, policy_limit_reached=True),
        ledger,
        None,
        (),
        0,
        {"complete": True},
    )
    attempt = AttemptEvidence(
        "benchmark",
        "world",
        0,
        "deterministic",
        attempt_rng_ids("benchmark", "world", 0),
        {"checkpoint_sha256": "a"},
        {"world_id": "world"},
        result,
    )
    assert summarize_world([attempt], k=1)["any_attempt_hazard"] is None
    with pytest.raises(ValueError):
        summarize_world([attempt, attempt], k=1)


def test_seed_audit_rejects_consumed_seeds_of_the_run_reservation(config_path, tmp_path):
    from training.provenance import audit_consumed_training_seeds
    from training.run_config import load_run_config

    run = load_run_config(config_path)
    reservation = run.reserved_manifest
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps({"status": "consumed", "seed": 987654321}) + "\n")
    assert audit_consumed_training_seeds([path], run.reserved_manifest)["consumed_count"] == 1
    for seed in [
        reservation[role][0]["seed"] for role in ("validation", "heldout", "replacements")
    ]:
        path.write_text(json.dumps({"status": "consumed", "seed": seed}) + "\n")
        with pytest.raises(ValueError, match="reserved|reservations"):
            audit_consumed_training_seeds([path], run.reserved_manifest)
