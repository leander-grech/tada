"""A run can train on the authenticated no-random-extra-TTL derivative of the frozen MXP generator."""

import json

import pytest
from support.small_env import small_bgy_config, small_bgy_run, training_env

from training.generator_variant import NO_RANDOM_EXTRA_TTL, snapshot_root, variant_config_sha256
from training.phase import apply_sets
from training.run_config import load_run_config, scenario_config_digest
from training.use_case import use_case


def _mxp_run(config_path, **scenario):
    small_bgy_config(config_path, **scenario)
    data = json.loads(config_path.read_text())
    case = use_case("mxp")
    data["scenario"].update(case.scenario_fields(), config_root=str(case.config_root))
    config_path.write_text(json.dumps(data))
    return load_run_config(config_path)


def _first_world(run, tmp_path):
    env = training_env(run, tmp_path)
    try:
        env.reset()
        return env.scenario_seed, env.simulator.world.json(), env._scenario_config_root
    finally:
        env.close()


def test_variant_workers_load_the_derived_generator_on_the_same_seed_stream(config_path, tmp_path):
    parent = _mxp_run(config_path)
    variant = _mxp_run(config_path, generator_variant=NO_RANDOM_EXTRA_TTL)
    parent_seed, parent_world, parent_root = _first_world(parent, tmp_path / "parent")
    seed, world, root = _first_world(variant, tmp_path / "variant")
    assert seed == parent_seed
    assert root == snapshot_root(NO_RANDOM_EXTRA_TTL) != parent_root
    assert scenario_config_digest(root) == variant_config_sha256()
    assert world != parent_world
    events = [
        json.loads(line)
        for path in (tmp_path / "variant/scenarios").iterdir()
        for line in path.read_text().splitlines()
    ]
    assert events and all(event["generator_variant"] == NO_RANDOM_EXTRA_TTL for event in events)


def test_phase_switches_the_variant_and_the_disabled_variant_keeps_old_digests(config_path):
    run = small_bgy_run(config_path)
    assert small_bgy_run(config_path, generator_variant="").semantic_digest == run.semantic_digest
    phased, change = apply_sets(run, [f"generator_variant={NO_RANDOM_EXTRA_TTL}"])
    assert change == {"generator_variant": ["", NO_RANDOM_EXTRA_TTL]}
    assert phased.semantic_digest != run.semantic_digest
    with pytest.raises(ValueError):
        apply_sets(run, ["generator_variant=arbitrary"])


def test_variant_rejects_a_non_mxp_parent_and_changed_snapshot_bytes(config_path, tmp_path):
    with pytest.raises(ValueError, match="frozen MXP"):
        training_env(small_bgy_run(config_path, generator_variant=NO_RANDOM_EXTRA_TTL), tmp_path)
    config = snapshot_root(NO_RANDOM_EXTRA_TTL) / "scenario_generator/config.toml"
    original = config.read_bytes()
    try:
        config.write_bytes(original + b"\n# unauthenticated change\n")
        with pytest.raises(ValueError, match="content mismatch"):
            snapshot_root(NO_RANDOM_EXTRA_TTL)
    finally:
        config.write_bytes(original)
