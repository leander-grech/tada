"""Generator diagnostics authenticate their only permitted change."""

import shutil

import pytest

from evaluation.benchmark.generator_variant import (
    VARIANT,
    generator_variant,
    validate_variant_root,
    variant_root,
)
from training.use_case import use_case


def test_variant_has_distinct_authenticated_identity_and_preserves_parent():
    parent = use_case("mxp").config_root
    before = (parent / "scenario_generator/config.toml").read_bytes()
    identity = generator_variant("mxp", VARIANT)
    root = variant_root(identity)
    validate_variant_root(root, identity)
    assert identity["config_sha256"] != identity["parent_config_sha256"]
    assert (parent / "scenario_generator/config.toml").read_bytes() == before
    with pytest.raises(ValueError):
        generator_variant("bgy", VARIANT)
    with pytest.raises(ValueError):
        generator_variant("mxp", "arbitrary")


def test_variant_rejects_other_changed_bytes_and_tampered_identity(tmp_path):
    identity = generator_variant("mxp", VARIANT)
    root = tmp_path / "variant"
    shutil.copytree(variant_root(identity), root)
    config = root / "scenario_generator/config.toml"
    config.write_bytes(config.read_bytes() + b"\n# unauthenticated change\n")
    with pytest.raises(ValueError):
        validate_variant_root(root, identity)
    with pytest.raises(ValueError):
        variant_root({**identity, "config_sha256": "0" * 64})


def test_variant_materialization_replays_and_changes_protocol_identity(tmp_path):
    from common.jsonio import read_json
    from evaluation.benchmark.contract import procedure_contract, materialize_manifest
    from evaluation.benchmark.worlds import (
        MaterializingEnv,
        environment_options,
        validate_development_manifest,
        validate_replay,
    )

    contract = procedure_contract("development", campaign="mxp", generator_variant=VARIANT)
    original = procedure_contract("development", campaign="mxp")
    assert contract.compatibility_sha256 != original.compatibility_sha256
    seed = use_case("mxp").inputs()[0]["validation"][0]["seed"]
    path = materialize_manifest(contract, [seed], tmp_path / "manifest.json")
    manifest = validate_development_manifest(read_json(path))
    env = MaterializingEnv(environment_options(contract, [seed]))
    try:
        env.reset()
        validate_replay(manifest["worlds"][0], env.benchmark_world)
    finally:
        env.close()
    with pytest.raises(ValueError):
        procedure_contract("heldout", campaign="mxp", generator_variant=VARIANT)
