"""Resolved payloads must be immutable, validated and tied to their semantics."""

import json
import pickle
import re

import pytest

from common.sealed import consume
from training.run_config import load_run_config, reseal
from training.seed_stream import content_digest


def test_payloads_are_immutable_serializable_and_consumed_by_role(config_path) -> None:
    config = load_run_config(config_path)
    for role in ("env", "model", "reward", "metrics"):
        payload = getattr(config, f"{role}_payload")()
        assert consume(payload, config.semantic_digest, role=role) == config.semantic_digest
        assert json.loads(json.dumps(payload)) == json.loads(
            json.dumps(pickle.loads(pickle.dumps(payload)))
        )
        with pytest.raises(TypeError):
            payload["semantic_digest"] = "tampered"
    with pytest.raises(TypeError):
        config.scenario["aircraft_count"] = 19
    with pytest.raises(ValueError, match="model"):
        consume(config.env_payload(), config.semantic_digest, role="model")


def test_semantic_digest_tracks_version_tokens(config_path, monkeypatch):
    base = load_run_config(config_path)
    monkeypatch.setattr("actions.rewards.REWARD_VERSION", "ma-reward-bumped")
    assert load_run_config(config_path).semantic_digest != base.semantic_digest


def test_consume_rejects_a_foreign_digest(config_path):
    run = load_run_config(config_path)
    consume(run.env_payload(), run.semantic_digest, role="env")
    with pytest.raises(ValueError, match="env"):
        consume(run.env_payload(), "0" * 64, role="env")


def test_reseal_recomputes_sections_and_digest(config_path):
    run = load_run_config(config_path)
    lowered = reseal(
        run, optimizer=dict(run.optimizer, learning_rate=run.optimizer["learning_rate"] / 3)
    )
    assert lowered.semantic_digest != run.semantic_digest
    assert lowered.semantic_sections["optimizer"] == content_digest(lowered.optimizer)


def test_semantic_hash_changes_with_behavior_but_not_run_label(config_path) -> None:
    first = load_run_config(config_path)
    data = json.loads(config_path.read_text())
    data["run_id"] = "relabelled"
    config_path.write_text(json.dumps(data))
    assert load_run_config(config_path).semantic_digest == first.semantic_digest
    data["thresholds"]["training_success_s"] = 60.0
    config_path.write_text(json.dumps(data))
    assert load_run_config(config_path).semantic_digest != first.semantic_digest


INVALID = [
    ("version", True),
    ("version", 2),
    ("run_id", ".."),
    ("run_id", "a/b"),
    ("phase_id", "."),
    ("root_seed", -1),
    ("root_seed", 10**12),
    ("topology.env_runners", -1),
    ("topology.envs_per_runner", 0),
    ("scenario.aircraft_count", True),
    ("scenario.aircraft_count", 0),
    ("scenario.verification", "other"),
    ("scenario.noop_verification_max_seconds", 1.5),
    ("scenario.noop_verification_max_seconds", True),
    ("thresholds.operational_success_s", 30),
    ("optimizer.learning_rate", float("nan")),
    ("optimizer.learning_rate", -1),
    ("optimizer.entropy_coeff", -0.1),
    ("optimizer.minibatch_size", 0),
    ("optimizer.gamma", 1.5),
    ("resources.backend", "magic"),
    ("resources.learner_device", "tpu"),
    ("resources.rollout_backend", "gpu_batch"),
    ("resources.learner_gpu_uuid", "bad"),
    ("cadences.checkpoint_iterations", 0),
    ("reward.count_baseline", 0),
    ("reward.global_share", 1.5),
    ("features.neighbor_selection", "legacy"),
    ("features.knee_s", 1000.0),  # knee_s > cap_s
]


@pytest.mark.parametrize(("path", "value"), INVALID)
def test_invalid_values_are_rejected_naming_the_field_before_any_output(
    config_path, tmp_path, path, value
):
    raw = json.loads(config_path.read_text())
    raw.setdefault("execution", {})["output_root"] = str(tmp_path / "out")
    target = raw
    *parents, leaf = path.split(".")
    for key in parents:
        target = target[key]
    target[leaf] = value
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=re.escape(leaf)):
        load_run_config(config_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("section", "mutation", "message"),
    [
        ("optimizer", {"minibatch_size": 7}, "minibatch_size"),  # does not divide train_batch_size
        ("optimizer", {"minibatch_size": 1}, "max_seq_len"),  # below recurrent max_seq_len
        (
            "resources",
            {"learner_device": "cuda"},
            "backend",
        ),  # cuda learner with gpus_per_learner == 0
        ("scenario", {"surprise": 1}, "scenario"),  # unknown key
    ],
)
def test_cross_field_and_exact_key_rules(config_path, section, mutation, message):
    raw = json.loads(config_path.read_text())
    raw[section].update(mutation)
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        load_run_config(config_path)


def test_reservations_are_ordered_disjoint_and_protected_by_hash(config_path) -> None:
    config = load_run_config(config_path)
    manifest = config.reserved_manifest
    seeds = [
        row["seed"]
        for group in ("validation", "heldout", "replacements")
        for row in manifest[group]
    ]
    assert len(seeds) == len(set(seeds)) == 7
    assert [row["seed"] for row in manifest["validation"]] == [5, 9]
    assert config.reserved_seeds == frozenset(seeds)
    path = config_path.parent / "reserved.json"
    data = json.loads(path.read_text())
    data["heldout"][0]["seed"] = 19
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_run_config(config_path)


def test_training_restores_before_first_generation_and_journals_actual_worlds(
    config_path, tmp_path
) -> None:
    from pathlib import Path

    from training.env import TrainingATCEnv
    from training.run_config import scenario_config_digest

    data = json.loads(config_path.read_text())
    scenario_root = Path("config/simulation_config").resolve()
    data["scenario"].update(
        airport="BGY",
        config_root=str(scenario_root),
        config_sha256=scenario_config_digest(scenario_root),
    )
    config_path.write_text(json.dumps(data))
    config = load_run_config(config_path)
    campaign = json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
    campaign_seeds = {
        row["seed"]
        for group in ("validation", "heldout", "replacements")
        for row in campaign[group]
    }
    assert not (config.reserved_seeds & campaign_seeds)

    def construct(directory):
        return TrainingATCEnv(
            {
                "resolved_env": config.env_payload(),
                "expected_semantic_digest": config.semantic_digest,
                "scenario_artifact_dir": str(directory),
            }
        )

    env = construct(tmp_path / "original")
    assert env.simulator.world is None
    assert env.get_training_stream_state()["stream"]["cursor"] == 0
    assert not list((tmp_path / "original").glob("*.jsonl"))
    env.reset()
    state = env.get_training_stream_state()
    env.reset()
    resumed = construct(tmp_path / "resumed")
    resumed.set_training_stream_state(state)
    assert resumed.simulator.world is None
    resumed.reset()
    assert resumed.scenario_seed == env.scenario_seed
    assert resumed.consumed_semantic_digest == config.semantic_digest
    with pytest.raises(ValueError, match="first reset"):
        resumed.set_training_stream_state(state)

    original = [
        json.loads(line)
        for path in (tmp_path / "original").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    replay = [
        json.loads(line)
        for path in (tmp_path / "resumed").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    generated = [row for row in original if row["status"] == "generated"]
    replay_generated = next(row for row in replay if row["status"] == "generated")
    assert len(generated) == 2
    assert generated[0]["world_sha256"] != generated[1]["world_sha256"]
    assert generated[1]["world_sha256"] == replay_generated["world_sha256"]
    assert generated[1]["generation_attempts"] == replay_generated["generation_attempts"]
    assert all(
        row["seed"] not in campaign_seeds | config.reserved_seeds for row in original + replay
    )
    assert sum(row["status"] == "consumed" for row in original) == 2
    assert all(
        row["reservation_sha256"] == config.reserved_manifest["manifest_sha256"]
        for row in original + replay
    )


def test_frozen_conditions_are_hash_linked_to_the_seed_reservation(config_path) -> None:
    config = load_run_config(config_path)
    assert (
        config.reserved_conditions["reservation_sha256"]
        == config.reserved_manifest["manifest_sha256"]
    )
    path = config_path.parent / "conditions.json"
    conditions = json.loads(path.read_text())
    conditions["physics"]["horizontal_safezone_radius_nm"] *= 2
    path.write_text(json.dumps(conditions))
    with pytest.raises(ValueError, match="conditions.*hash"):
        load_run_config(config_path)
    del conditions["conditions_sha256"]
    conditions["reservation_sha256"] = "f" * 64
    conditions["conditions_sha256"] = content_digest(conditions)
    path.write_text(json.dumps(conditions))
    with pytest.raises(ValueError, match="reservation"):
        load_run_config(config_path)


def test_initial_preprocessing_setting_changes_scenario_identity(config_path):
    original = load_run_config(config_path)
    data = json.loads(config_path.read_text())
    data["scenario"]["start_offset_max_s"] += 1
    config_path.write_text(json.dumps(data))
    changed = load_run_config(config_path)
    differing = {
        key
        for key, digest in changed.semantic_sections.items()
        if original.semantic_sections[key] != digest
    }
    assert differing == {"scenario"}
    assert changed.semantic_digest != original.semantic_digest


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_heldout",
        "empty_replacements",
        "unknown_stratum",
        "bad_weight",
        "missing_native",
        "missing_source",
        "changed_validation",
        "reordered_replacements",
    ],
)
def test_rehashed_incomplete_procedure_contract_is_rejected(mutation):
    from pathlib import Path

    from training.run_config import validate_reservation_manifest, validate_reserved_conditions

    root = Path("config/contracts/bgy")
    manifest = json.loads((root / "reserved-v1.json").read_text())
    # v2 replaces the active base pointer; immutable v1 remains archival evidence.
    base = json.loads((root / "base.json").read_text())
    conditions = json.loads((root / base["reserved_conditions"]).read_text())
    if mutation.startswith("empty_"):
        manifest[mutation.removeprefix("empty_")] = []
    elif mutation == "unknown_stratum":
        manifest["heldout"][0]["stratum"] = "absent"
    elif mutation == "bad_weight":
        manifest["strata"][0]["weight"] = -1
        conditions["benchmark"]["stratum"]["weight"] = -1
    elif mutation == "changed_validation":
        manifest["validation"][0]["seed"] = 7
    elif mutation == "reordered_replacements":
        manifest["replacements"][0], manifest["replacements"][1] = (
            manifest["replacements"][1],
            manifest["replacements"][0],
        )
    elif mutation == "missing_native":
        conditions["native_artifact"] = {"mode": "external"}
    elif mutation == "missing_source":
        conditions["baseline_source"] = {"placeholder": True}
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = content_digest(manifest)
    conditions["reservation_sha256"] = manifest["manifest_sha256"]
    conditions.pop("conditions_sha256")
    conditions["conditions_sha256"] = content_digest(conditions)
    with pytest.raises(ValueError):
        validate_reservation_manifest(manifest)
        validate_reserved_conditions(conditions, manifest)


def _rehashed(manifest):
    manifest = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return {**manifest, "manifest_sha256": content_digest(manifest)}


def test_mxp_stream_identity_is_unchanged_by_the_frozen_bgy_exclusion():
    from pathlib import Path

    from training.reservations import effective_reserved_seeds, validate_reservation_manifest
    from training.seed_stream import ScenarioSeedStream

    mxp = json.loads(Path("config/contracts/reserved-mxp-v1.json").read_text())
    own, effective = validate_reservation_manifest(mxp), effective_reserved_seeds(mxp)
    assert (
        ScenarioSeedStream(1, (0, 0, 0), own).get_state()
        == ScenarioSeedStream(1, (0, 0, 0), effective).get_state()
    )


def test_fixture_training_stream_never_draws_a_frozen_bgy_seed(config_path):
    from pathlib import Path

    from training.reservations import effective_reserved_seeds, validate_reservation_manifest
    from training.seed_stream import ScenarioSeedStream

    bgy = validate_reservation_manifest(
        json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
    )
    fixture = json.loads((config_path.parent / "reserved.json").read_text())
    # Root 1795590's first raw draw is the frozen BGY seed 1128224981.
    assert ScenarioSeedStream(1795590, (0, 0, 0), frozenset()).next_seed() in bgy
    stream = ScenarioSeedStream(1795590, (0, 0, 0), effective_reserved_seeds(fixture))
    assert not {stream.next_seed() for _ in range(8)} & bgy
    assert stream.rejected == 1


@pytest.mark.parametrize("aircraft_count, accepted", [(10, True), (15, False)])
def test_campaign_stratum_is_judged_by_capacity_not_identity(aircraft_count, accepted):
    from pathlib import Path

    from training.reservations import validate_reservation_manifest

    manifest = json.loads(Path("config/contracts/reserved-mxp-v1.json").read_text())
    old = manifest["strata"][0]["id"]
    manifest["strata"][0].update(id="any-name", aircraft_count=aircraft_count)
    for group in ("heldout", "replacements"):
        for row in manifest[group]:
            assert row["stratum"] == old
            row["stratum"] = "any-name"
    if accepted:
        validate_reservation_manifest(_rehashed(manifest))
    else:
        with pytest.raises(ValueError, match="common-capacity"):
            validate_reservation_manifest(_rehashed(manifest))


@pytest.mark.parametrize(
    "field, value", [("airport", "BGY"), ("difficulty", "VALIDATION_USE_CASE_2")]
)
def test_freshly_hashed_campaign_stratum_must_match_its_frozen_conditions(field, value):
    from pathlib import Path

    from training.reservations import validate_reservation_manifest

    manifest = json.loads(Path("config/contracts/reserved-mxp-v1.json").read_text())
    manifest["strata"][0][field] = value
    with pytest.raises(ValueError, match="airport/difficulty"):
        validate_reservation_manifest(_rehashed(manifest))


def test_campaign_conditions_keep_lineage_and_fixture_purpose(config_path):
    from pathlib import Path

    from training.run_config import validate_reserved_conditions

    campaign = load_run_config(Path("config/contracts/bgy/base.json"))
    prior = json.loads(Path("config/contracts/bgy/conditions-v1.json").read_text())
    assert campaign.reserved_conditions["previous_conditions_sha256"] == prior["conditions_sha256"]
    assert campaign.reserved_conditions["native_artifact"] == prior["native_artifact"]
    assert campaign.reserved_conditions["physics"] == prior["physics"]
    fixture = load_run_config(config_path)
    assert (
        fixture.reserved_manifest["purpose"] == fixture.reserved_conditions["purpose"] == "fixture"
    )
    with pytest.raises(ValueError, match="reservation identity"):
        validate_reserved_conditions(fixture.reserved_conditions, campaign.reserved_manifest)
    promoted = json.loads(json.dumps(fixture.reserved_conditions))
    promoted["purpose"] = "campaign"
    promoted["reservation_sha256"] = campaign.reserved_manifest["manifest_sha256"]
    promoted.pop("conditions_sha256")
    promoted["conditions_sha256"] = content_digest(promoted)
    with pytest.raises(ValueError):
        validate_reserved_conditions(promoted, campaign.reserved_manifest)


def test_startup_rejects_changed_frozen_initial_window_before_generation(config_path, tmp_path):
    from training.env import TrainingATCEnv
    from training.frozen_settings import read_current_initial_preprocessing_settings

    path = config_path.parent / "conditions.json"
    conditions = json.loads(path.read_text())
    conditions["version"] = 2
    conditions["previous_conditions_sha256"] = conditions.pop("conditions_sha256")
    conditions["benchmark"].update(read_current_initial_preprocessing_settings())
    conditions["benchmark"]["early_auto_solve_target_window_s"] += 1
    conditions["conditions_sha256"] = content_digest(conditions)
    path.write_text(json.dumps(conditions))
    config = load_run_config(config_path)
    with pytest.raises(ValueError, match="initial preprocessing"):
        TrainingATCEnv(
            {
                "resolved_env": config.env_payload(),
                "expected_semantic_digest": config.semantic_digest,
                "scenario_artifact_dir": str(tmp_path / "scenarios"),
            }
        )
    assert not (tmp_path / "scenarios").exists()


def test_env_payload_binds_instance_reward_and_feature_sections(config_path):
    config = load_run_config(config_path)
    payload = config.env_payload()
    assert payload.get("reward") == config.reward
    assert payload.get("features") == config.features


def test_verification_cap_is_independent_and_semantically_bound(config_path):
    original = load_run_config(config_path)
    data = json.loads(config_path.read_text())
    data["scenario"]["noop_verification_max_seconds"] = 73
    config_path.write_text(json.dumps(data))
    changed = load_run_config(config_path)
    assert changed.scenario["episode_steps"] == original.scenario["episode_steps"]
    assert changed.scenario["noop_verification_max_seconds"] == 73
    assert changed.semantic_digest != original.semantic_digest
    assert changed.reserved_conditions == original.reserved_conditions
    assert {
        key
        for key, digest in changed.semantic_sections.items()
        if original.semantic_sections[key] != digest
    } == {"scenario"}


def test_omitted_verification_cap_resolves_same_as_explicit_value(config_path):
    omitted = load_run_config(config_path)
    data = json.loads(config_path.read_text())
    data["scenario"]["noop_verification_max_seconds"] = omitted.scenario[
        "noop_verification_max_seconds"
    ]
    config_path.write_text(json.dumps(data))
    explicit = load_run_config(config_path)
    assert omitted.semantic_digest == explicit.semantic_digest
    assert omitted.env_payload() == explicit.env_payload()


def test_resolved_env_instances_retain_independent_consumer_values(config_path):
    first = load_run_config(config_path)
    original_payload = first.env_payload()
    data = json.loads(config_path.read_text())
    data["reward"]["count_baseline"] = 7.0
    data["features"]["linear_scale_s"] = 777.0
    config_path.write_text(json.dumps(data))
    second = load_run_config(config_path)
    assert second.env_payload()["reward"]["count_baseline"] == 7.0
    assert second.env_payload()["features"]["linear_scale_s"] == 777.0
    assert first.env_payload() == original_payload
    assert first.semantic_digest != second.semantic_digest
    for resolved in (first, second):
        assert (
            consume(resolved.env_payload(), resolved.semantic_digest, role="env")
            == resolved.semantic_digest
        )


def test_punctuation_identifiers_preserve_output_ancestry(config_path):
    from pathlib import Path

    data = json.loads(config_path.read_text())
    data.update(run_id="run.v2-test", phase_id="phase_2.1")
    config_path.write_text(json.dumps(data))
    run = load_run_config(config_path)
    root = Path(run.execution["output_root"])
    segment = (root / run.run_id / run.phase_id / "segment-fixture").resolve()
    assert segment.is_relative_to(root)
    assert segment.relative_to(root).parts == ("run.v2-test", "phase_2.1", "segment-fixture")


@pytest.mark.parametrize("document", ["reserved", "conditions"])
def test_boolean_protocol_version_is_not_integer_one(config_path, document):
    path = config_path.parent / f"{document}.json"
    data = json.loads(path.read_text())
    data["version"] = True
    hash_key = {"reserved": "manifest_sha256", "conditions": "conditions_sha256"}[document]
    data.pop(hash_key)
    data[hash_key] = content_digest(data)
    path.write_text(json.dumps(data))
    from training.run_config import validate_reservation_manifest, validate_reserved_conditions

    with pytest.raises(ValueError):
        if document == "reserved":
            validate_reservation_manifest(data)
        else:
            validate_reserved_conditions(
                data, json.loads((config_path.parent / "reserved.json").read_text())
            )


@pytest.mark.parametrize("runners", [3, 1_000_019])
def test_root_seed_range_accounts_for_all_owned_ray_worker_offsets(config_path, runners):
    data = json.loads(config_path.read_text())
    data["topology"]["env_runners"] = runners
    # Installed Ray uses +worker_index, with an extra +1,000,000 for evaluation.
    maximum = 2**32 - 1 - max(runners, 1_000_001)
    data["root_seed"] = maximum
    output = config_path.parent / "not-created"
    data["execution"] = {"output_root": str(output)}
    config_path.write_text(json.dumps(data))
    assert load_run_config(config_path).root_seed == maximum
    data["root_seed"] += 1
    config_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="root_seed.*Ray|Ray.*root_seed"):
        load_run_config(config_path)
    assert not output.exists()


def _campaign_scenario(**changes):
    from pathlib import Path

    conditions = json.loads(Path("config/contracts/bgy/conditions-v2.json").read_text())
    scenario = dict(
        json.loads(Path("config/contracts/bgy/base.json").read_text())["scenario"]
    )
    return {**scenario, **changes}, {"purpose": "campaign"}, conditions


def test_campaign_run_may_use_a_different_aircraft_count_at_the_same_stratum():
    from training.run_config import validate_campaign_stratum

    validate_campaign_stratum(*_campaign_scenario(aircraft_count=20))


def test_campaign_run_still_rejects_a_different_airport_or_episode_shape():
    from training.run_config import validate_campaign_stratum

    with pytest.raises(ValueError, match="airport"):
        validate_campaign_stratum(*_campaign_scenario(airport="MXP"))
    with pytest.raises(ValueError, match="episode_steps"):
        validate_campaign_stratum(*_campaign_scenario(aircraft_count=20, episode_steps=1))
    validate_campaign_stratum(*_campaign_scenario(episode_steps=1))
