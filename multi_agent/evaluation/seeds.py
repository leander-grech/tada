"""Evaluation seed identities: manifests, operational streams and reservation access.

``ATCEnv`` grows an evaluation manifest lazily: it walks the configured fixed
seeds, then draws additional ones from a seeded RNG until its randomize quota is
spent.  That is fine for a single pass, but a retry harness has to replay an
identical scenario set on every round, so it needs the whole list up front.

``build_evaluation_seed_manifest`` produces exactly the sequence the environment
would have drawn.  ``test/evaluation_test/seeds_test.py`` pins that equivalence
against the real environment so the two cannot drift apart silently.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from training.reservations import effective_reserved_seeds
from training.seed_stream import (
    ALGORITHM,
    SEED_MAX,
    ScenarioSeedStream,
    content_digest,
    nonnegative_int,
)

# Matches the draw range in ``ATCEnv._next_evaluation_scenario_seed``.
_SEED_MAX = 2**31 - 1


def build_evaluation_seed_manifest(
    fixed_seeds: Sequence[int],
    total: int,
    *,
    rng_seed: int,
) -> list[int]:
    """Return *total* evaluation seeds: the fixed manifest plus drawn extras.

    Args:
        fixed_seeds: The configured evaluation seeds, used in order first.
        total: Desired manifest length. Values at or below the fixed count
            truncate; larger values are topped up with drawn seeds.
        rng_seed: Seed for the draw RNG. The environment uses
            ``Config.SEED + cantor_pair(worker_index, vector_index)``, which is
            ``Config.SEED`` for the local evaluation runner.

    Returns:
        Seeds in the order the environment would visit them. Drawn seeds never
        collide with the fixed set, matching the environment's rejection rule,
        which does not deduplicate the drawn seeds against each other.
    """
    fixed = [int(seed) for seed in fixed_seeds]
    if total <= len(fixed):
        return fixed[:total]

    rng = random.Random(rng_seed)
    fixed_set = set(fixed)
    seeds = list(fixed)
    while len(seeds) < total:
        candidate = rng.randint(0, _SEED_MAX)
        while candidate in fixed_set:
            candidate = rng.randint(0, _SEED_MAX)
        seeds.append(candidate)
    return seeds


def build_reservation_manifest(
    validation_seeds: Sequence[int],
    *,
    root: int,
    heldout_count: int,
    replacement_count: int,
    stratum: dict,
    purpose: str,
) -> dict:
    """Freeze a disjoint primary stratum and ordered technical-failure reserves.

    This draws identities only, never worlds. Replacement eligibility is technical
    generation failure, never difficulty or policy quality.
    """
    validation = list(validation_seeds)
    for seed in validation:
        if nonnegative_int(seed, "validation seed") > SEED_MAX:
            raise ValueError("validation seed exceeds the native seed range")
    if len(validation) != len(set(validation)):
        raise ValueError("validation manifest must be duplicate-free")
    if purpose not in {"campaign", "fixture"}:
        raise ValueError("reservation purpose must be campaign or fixture")
    nonnegative_int(heldout_count, "heldout_count")
    nonnegative_int(replacement_count, "replacement_count")
    stream = ScenarioSeedStream(root, (0, 0, 0), frozenset(validation))
    seen = set(validation)
    candidates = []
    while len(candidates) < heldout_count + replacement_count:
        seed = stream.next_seed()
        if seed not in seen:
            candidates.append(seed)
            seen.add(seed)
    records = [
        {"seed": seed, "stratum": stratum["id"], "status": "unmaterialized"} for seed in candidates
    ]
    manifest = {
        "version": 1,
        "purpose": purpose,
        "strata": [dict(stratum)],
        "validation": [{"seed": seed, "stratum": "published-validation"} for seed in validation],
        "heldout": records[:heldout_count],
        "replacements": records[heldout_count:],
        "replacement_policy": "ordered-technical-generation-failure-only",
        "generation": {
            "algorithm": ALGORITHM,
            "root": root,
            "stream_id": [0, 0, 0],
            "cursor": stream.cursor,
            "reserved_rejections": stream.rejected,
            "duplicate_rejections": stream.cursor - stream.rejected - len(candidates),
        },
    }
    manifest["generation_sha256"] = content_digest(manifest["generation"])
    manifest["manifest_sha256"] = content_digest(manifest)
    return manifest


def operational_seeds(run, count: int) -> list[int]:
    """Draw *count* operational-evaluation seeds disjoint from every reservation."""
    stream = ScenarioSeedStream(
        run.root_seed,
        (0, 0, 0),
        effective_reserved_seeds(run.reserved_manifest),
        domain="operational_evaluation",
    )
    return [stream.next_seed() for _ in range(count)]


@dataclass(frozen=True)
class SeedAuthorizer:
    """The single gate for evaluating reserved seed identities.

    Disjoint operational seeds pass freely; reserved seeds pass only through an
    explicit development gate (the benchmark reader) and are refused otherwise.
    """

    reserved: frozenset[int]
    development_gate: Callable[[list[int]], None] | None = None

    def authorize(self, seeds: Sequence[int]) -> None:
        if not set(seeds) & self.reserved:
            return
        if self.development_gate is None:
            raise ValueError("reserved evaluation seeds require separate benchmark access gate")
        self.development_gate(list(seeds))
