"""Authenticate diagnostic density without changing the saved training contract."""

from pathlib import Path

from common.hashing import content_digest
from evaluation.benchmark.use_cases import selection
from evaluation.benchmark.contract import BenchmarkContract, procedure_contract
from training.run_config import scenario_config_digest


def benchmark_aircraft_count(payload, compatibility):
    campaign, count = selection(compatibility)
    from evaluation.benchmark.generator_variant import variant_name

    procedure_contract(
        "development",
        campaign=campaign,
        aircraft_count=count,
        generator_variant=variant_name(compatibility),
    ).require_compatible(BenchmarkContract(compatibility, {}))
    conditions = compatibility["conditions"]
    if payload["reserved_manifest"]["manifest_sha256"] != compatibility[
        "reservation_sha256"
    ] or content_digest(payload["reserved_conditions"]) != content_digest(conditions):
        raise ValueError("checkpoint benchmark reservation/conditions mismatch")
    baseline = conditions["benchmark"]
    scenario = payload["scenario"]
    saved_count = scenario["aircraft_count"]
    allowed_counts = {baseline["stratum"]["aircraft_count"], 20}
    if type(saved_count) is not int or saved_count not in allowed_counts:
        raise ValueError("checkpoint benchmark world mismatch: aircraft_count")
    expected = {
        "airport": baseline["stratum"]["airport"],
        "difficulty": baseline["stratum"]["difficulty"],
        "episode_steps": baseline["episode_steps"],
        "start_offset_max_s": baseline["start_offset_max_s"],
        "config_sha256": baseline["stratum"]["scenario_config_sha256"],
        "noop_verification_max_seconds": compatibility["verification_cap_s"],
        "verification": "native_generation",
    }
    for key, value in expected.items():
        if scenario[key] != value:
            raise ValueError(f"checkpoint benchmark world mismatch: {key}")
    if scenario_config_digest(Path(scenario["config_root"])) != expected["config_sha256"]:
        raise ValueError("checkpoint benchmark configuration root mismatch")
    return count
