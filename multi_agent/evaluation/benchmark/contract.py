"""Frozen comparison compatibility is distinct from each method's provenance."""

from dataclasses import dataclass
from typing import Any

from common.atomic_io import atomic_json
from common.hashing import FrozenDict, content_digest, nonnegative_int
from common.jsonio import read_json


@dataclass(frozen=True)
class BenchmarkContract:
    compatibility: Any
    method: Any

    def __post_init__(self):
        if self.compatibility.get("threshold_s") != 60:
            raise ValueError("benchmark grading requires inclusive absolute 60 seconds")
        if self.compatibility.get("verification_cap_s", 0) <= 0:
            raise ValueError("benchmark requires a positive shared simulated verification cap")
        object.__setattr__(self, "compatibility", FrozenDict(self.compatibility))
        object.__setattr__(self, "method", FrozenDict(self.method))
        content_digest(self.compatibility)
        content_digest(self.method)

    @property
    def compatibility_sha256(self):
        return content_digest(self.compatibility)

    def require_compatible(self, other):
        if self.compatibility_sha256 != other.compatibility_sha256:
            raise ValueError("incompatible benchmark protocols")

    def to_dict(self):
        return {"compatibility": dict(self.compatibility), "method": dict(self.method)}


def attempt_rng_id(benchmark_id: str, world_id: str, attempt: int, domain: str) -> int:
    """Portable RNG key independent of scheduling, shards and method architecture."""
    nonnegative_int(attempt, "attempt")
    if not all(isinstance(value, str) and value for value in (benchmark_id, world_id, domain)):
        raise ValueError("benchmark, world and RNG domain must be nonempty strings")
    return int(
        content_digest(["benchmark-rng-v1", benchmark_id, world_id, attempt, domain])[:8], 16
    )


def attempt_rng_ids(benchmark_id, world_id, ordinal):
    return {
        domain: attempt_rng_id(benchmark_id, world_id, ordinal, domain)
        for domain in ("policy", "frame")
    }


def attempt_key(benchmark_id, world_id, mode, ordinal, checkpoint_sha256):
    """The immutable physical key shared by producer, reader and opening receipts."""
    nonnegative_int(ordinal, "attempt ordinal")
    if mode not in {"deterministic", "stochastic"} or (mode == "deterministic" and ordinal != 0):
        raise ValueError("unsupported canonical attempt protocol")
    if not all(
        isinstance(value, str) and value for value in (benchmark_id, world_id, checkpoint_sha256)
    ):
        raise ValueError("benchmark, world and checkpoint identities are required")
    identity = [benchmark_id, world_id, mode, ordinal, checkpoint_sha256]
    return content_digest(identity)


def inference_protocol(*, k, explore):
    if type(explore) is not bool or type(k) is not int or k <= 0 or (not explore and k != 1):
        raise ValueError("primary is one deterministic attempt; supplementary runs all positive K")
    return {
        "mode": "stochastic" if explore else "deterministic",
        "k": k,
        "label": f"stochastic-pass@{k}" if explore else "deterministic-primary",
        "early_stop": False,
    }


def validate_inference_protocol(protocol):
    if protocol.get("mode") not in {"deterministic", "stochastic"}:
        raise ValueError("unsupported canonical inference mode")
    expected = inference_protocol(k=protocol.get("k"), explore=protocol["mode"] == "stochastic")
    if dict(protocol) != expected:
        raise ValueError("noncanonical inference protocol")
    return expected


def procedure_contract(
    role: str, *, campaign="bgy", aircraft_count=10, generator_variant=None
) -> BenchmarkContract:
    """Resolve native and baseline identities before materializing any world."""
    from pathlib import Path

    from evaluation.benchmark.use_cases import (
        DENSITIES,
        procedure_inputs,
        procedure_root,
        procedure_selection,
    )
    from training.frozen_settings import (
        read_current_initial_preprocessing_settings,
        read_current_physics_settings,
    )
    from training.reservations import (
        CONTRACT_ROOT,
        validate_reservation_manifest,
        validate_reserved_conditions,
    )
    from training.run_config import scenario_config_digest
    from training.runtime_gate import verify_native_artifact

    if role not in {"development", "heldout", "replacements"}:
        raise ValueError("unknown benchmark role")
    if type(aircraft_count) is not int or aircraft_count not in (
        DENSITIES if role == "development" else {10}
    ):
        raise ValueError("unsupported campaign density or role")
    reservation, conditions = procedure_inputs(campaign)
    validate_reservation_manifest(reservation)
    validate_reserved_conditions(conditions, reservation)
    verify_native_artifact(conditions)
    if conditions["physics"] != read_current_physics_settings():
        raise ValueError("benchmark physics differs from frozen baseline")
    settings = read_current_initial_preprocessing_settings()
    if any(conditions["benchmark"][key] != value for key, value in settings.items()):
        raise ValueError("benchmark initial preprocessing differs from frozen baseline")
    config = scenario_config_digest(Path(procedure_root(campaign)))
    if config != conditions["benchmark"]["stratum"]["scenario_config_sha256"]:
        raise ValueError("benchmark world configuration changed")
    protocol = read_json(CONTRACT_ROOT / "benchmark.json")
    from config.config import Config
    from outcomes.evidence import OUTCOME_VERSION
    from outcomes.native_contract import native_contract_supported

    if not native_contract_supported():
        raise ValueError("unscorable benchmark native timing capability")
    variant = {}
    if generator_variant is not None:
        from evaluation.benchmark.generator_variant import generator_variant as resolve_variant

        if role != "development":
            raise ValueError("generator variants require development evaluation")
        variant = {"generator_variant": resolve_variant(campaign, generator_variant)}
    return BenchmarkContract(
        {
            **protocol,
            **variant,
            **(
                {"campaign_selection": extension}
                if (extension := procedure_selection(campaign, aircraft_count))
                else {}
            ),
            "role": role,
            "conditions": conditions,
            "reservation_sha256": reservation["manifest_sha256"],
            "native_timing_contract": "owned-0.2.80-exact-or-bounded-no-event-v1",
            "scoring_reference_sha256": content_digest({"outcome": OUTCOME_VERSION}),
            "control": {"seconds_per_policy_step": Config.TIME_BETWEEN_ACTIONS},
            "strata": [
                {
                    **reservation["strata"][0],
                    "aircraft_count": aircraft_count,
                    "count": 100 if role == "development" else 200,
                }
            ],
        },
        {},
    )


def validate_manifest(manifest):
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if claimed != content_digest(body):
        raise ValueError("benchmark manifest hash mismatch")
    contract = BenchmarkContract(**body["contract"])
    worlds = body["worlds"]
    seeds = [world["seed"] for world in worlds]
    ids = [world["world_id"] for world in worlds]
    if not worlds or len(set(seeds)) != len(seeds) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate benchmark worlds")
    if body["compatibility_sha256"] != contract.compatibility_sha256:
        raise ValueError("benchmark protocol identity changed")
    if body.get("sample_count", len(worlds)) != len(worlds) or (
        body.get("complete_role")
        and len(worlds) != sum(s["count"] for s in contract.compatibility["strata"])
    ):
        raise ValueError("missing benchmark worlds or changed fixed sample count")
    for world in worlds:
        payload = dict(world)
        identity = payload.pop("world_id")
        if identity != content_digest(payload):
            raise ValueError("benchmark world identity changed")
    return manifest


def materialize_manifest(contract, seeds: list[int], output):
    """Generate immutable inputs only; never step a policy or call a verifier."""
    from pathlib import Path

    from evaluation.benchmark.use_cases import procedure_inputs, selection
    from evaluation.benchmark.supplied import supplied_seeds
    from evaluation.benchmark.worlds import MaterializingEnv, environment_options

    campaign, count = selection(contract.compatibility)
    from evaluation.benchmark.generator_variant import variant_name

    current = procedure_contract(
        contract.compatibility["role"],
        campaign=campaign,
        aircraft_count=count,
        generator_variant=variant_name(contract.compatibility),
    )
    current.require_compatible(contract)
    reservation, _ = procedure_inputs(campaign)
    group = (
        "validation"
        if contract.compatibility["role"] == "development"
        else contract.compatibility["role"]
    )
    allowed = [r["seed"] for r in reservation[group]]
    # Development may also follow the supplied cohort's own order; authorization happens on reset.
    orders = [allowed] + ([list(supplied_seeds())] if group == "validation" else [])
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or all(seeds != [s for s in order if s in set(seeds)] for order in orders)
    ):
        raise ValueError("materialization requires ordered duplicate-free frozen role seeds")
    env = MaterializingEnv(environment_options(contract, seeds))
    worlds = []
    try:
        for index, _seed in enumerate(seeds):
            env._eval_episode_index = index - 1
            env._scenario_finished = True
            env.reset()
            worlds.append(env.benchmark_world)
    finally:
        env.close()
    body = {
        "version": 1,
        "contract": contract.to_dict(),
        "compatibility_sha256": contract.compatibility_sha256,
        "worlds": worlds,
        "input_only": True,
        "sample_count": len(worlds),
        "complete_role": seeds == allowed,
    }
    body["manifest_sha256"] = content_digest(body)
    validate_manifest(body)
    output = Path(output)
    atomic_json(output, body, exclusive=True)
    return output
