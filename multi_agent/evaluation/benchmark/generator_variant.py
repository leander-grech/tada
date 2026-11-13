"""Allowlisted evaluation-only generator diagnostics, derived from frozen input bytes."""

from pathlib import Path

from common.hashing import content_digest
from training.generator_variant import (
    NO_RANDOM_EXTRA_TTL,
    parent_config_sha256,
    snapshot_root,
    variant_config_sha256,
)
from training.run_config import scenario_config_digest

VARIANT = NO_RANDOM_EXTRA_TTL


def generator_variant(campaign, name):
    if campaign != "mxp" or name != VARIANT:
        raise ValueError("unsupported evaluation generator variant")
    return {
        "version": 1,
        "name": VARIANT,
        "purpose": "evaluation-only-diagnostic",
        "parent_config_sha256": parent_config_sha256(),
        "config_sha256": variant_config_sha256(),
        "changes": {"poisson_generation.aman_use_random_extra_ttl": False},
    }


def variant_name(compatibility):
    identity = compatibility.get("generator_variant")
    if identity is None:
        return None
    from evaluation.benchmark.use_cases import selection

    campaign, _ = selection(compatibility)
    expected = generator_variant(campaign, identity.get("name"))
    if compatibility["role"] != "development" or content_digest(identity) != content_digest(
        expected
    ):
        raise ValueError("invalid evaluation generator variant identity or role")
    return expected["name"]


def validate_variant_root(root, identity):
    expected = generator_variant("mxp", identity.get("name"))
    if (
        content_digest(identity) != content_digest(expected)
        or scenario_config_digest(Path(root)) != expected["config_sha256"]
    ):
        raise ValueError("evaluation generator variant content mismatch")


def variant_root(identity):
    root = snapshot_root(identity.get("name"))
    validate_variant_root(root, identity)
    return root
