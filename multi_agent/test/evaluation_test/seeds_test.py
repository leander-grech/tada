"""Tests that the materialized eval manifest matches what ATCEnv would draw."""

import csv
from pathlib import Path

import pytest

from atc_env.environment import ATCEnv
from config.config import Config
from evaluation.run import load_seed_manifest
from evaluation.seeds import SeedAuthorizer, build_evaluation_seed_manifest

_FIXED = [11, 22, 33]
PINNED_MANIFEST = Path(__file__).resolve().parents[2] / "config" / "evaluation_seed_manifest.csv"


def _env_seed_sequence(*, fixed: list[int], total: int) -> list[int]:
    """Collect the seeds ATCEnv would draw, without building any worlds.

    ``ATCEnv.__init__`` ends with a ``reset()``, which already consumes the
    first manifest entry, so the walk index is rewound here to compare the
    generator itself rather than that constructor side effect.
    """
    env = ATCEnv(
        {
            "evaluation_mode": True,
            "evaluation_seeds": fixed,
            "evaluation_randomize_scenario_count": max(0, total - len(fixed)),
            "worker_index": 0,
            "vector_index": 0,
        }
    )
    # pylint: disable=protected-access
    env._eval_episode_index = -1
    env._eval_randomize_used = 0
    return [env._next_evaluation_scenario_seed() for _ in range(total)]


def test_manifest_returns_fixed_seeds_in_order() -> None:
    assert build_evaluation_seed_manifest(_FIXED, 3, rng_seed=Config.SEED) == _FIXED


def test_manifest_truncates_when_total_is_smaller() -> None:
    assert build_evaluation_seed_manifest(_FIXED, 2, rng_seed=Config.SEED) == [11, 22]


def test_manifest_tops_up_beyond_fixed_seeds() -> None:
    manifest = build_evaluation_seed_manifest(_FIXED, 8, rng_seed=Config.SEED)

    assert len(manifest) == 8
    assert manifest[:3] == _FIXED
    assert all(seed not in set(_FIXED) for seed in manifest[3:])


def test_manifest_is_deterministic_for_a_given_rng_seed() -> None:
    first = build_evaluation_seed_manifest(_FIXED, 12, rng_seed=Config.SEED)
    second = build_evaluation_seed_manifest(_FIXED, 12, rng_seed=Config.SEED)

    assert first == second


def test_manifest_is_a_prefix_of_longer_manifests() -> None:
    short = build_evaluation_seed_manifest(_FIXED, 6, rng_seed=Config.SEED)
    long = build_evaluation_seed_manifest(_FIXED, 20, rng_seed=Config.SEED)

    assert long[: len(short)] == short


def test_manifest_matches_environment_draw_sequence() -> None:
    """The contract that keeps attempts mode comparable to a plain eval run."""
    total = 10

    assert build_evaluation_seed_manifest(
        _FIXED, total, rng_seed=Config.SEED
    ) == _env_seed_sequence(fixed=_FIXED, total=total)


def test_pinned_manifest_is_loadable_and_unique() -> None:
    """The checked-in benchmark set must survive the CLI's own loader."""
    seeds = load_seed_manifest(PINNED_MANIFEST)

    assert len(seeds) == 100
    assert len(set(seeds)) == len(seeds)


def test_pinned_manifest_matches_its_documented_provenance() -> None:
    """Tripwire: the pinned file is the builder's 100-scenario draw.

    If the generator or the fixed seed set changes intentionally, regenerate the
    manifest deliberately - silently diverging would leave the file's `source`
    column, and every report compared against it, describing something untrue.
    """
    seeds = load_seed_manifest(PINNED_MANIFEST)

    assert seeds == build_evaluation_seed_manifest(
        [int(seed) for seed in Config.SCENARIO_EVAL_SEEDS],
        len(seeds),
        rng_seed=Config.SEED,
    )


def test_pinned_manifest_source_column_labels_the_fixed_seeds() -> None:
    """The in-band provenance column must agree with the configured seed set."""
    with PINNED_MANIFEST.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fixed = {int(seed) for seed in Config.SCENARIO_EVAL_SEEDS}
    labelled = {int(row["seed"]) for row in rows if row["source"] == "config"}

    assert labelled == fixed
    assert {row["source"] for row in rows} == {"config", "topup"}


def test_authorizer_refuses_reserved_seeds_without_development_gate() -> None:
    authorizer = SeedAuthorizer(reserved=frozenset({7}))
    authorizer.authorize([1, 2])
    with pytest.raises(ValueError, match="benchmark access gate"):
        authorizer.authorize([1, 7])


def test_authorizer_routes_only_reserved_claims_through_development_gate() -> None:
    claims = []
    authorizer = SeedAuthorizer(reserved=frozenset({7}), development_gate=claims.append)
    authorizer.authorize((1, 2))
    authorizer.authorize((7, 3))
    assert claims == [[7, 3]]
