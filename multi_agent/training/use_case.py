"""Validation use cases: each names its frozen reservation, conditions and generator snapshot.

The resolved run keeps explicit scenario airport/difficulty/config fields and reservation
sections; a use case is only a name for one consistent set of them. ``scenario.use_case`` in a
run input expands to that set before validation and is never stored, so an absent field keeps
the explicit scenario values and their semantic digest.
"""

from dataclasses import dataclass
from pathlib import Path

from common.hashing import content_digest
from common.jsonio import read_json
from training.reservations import (
    CONTRACT_ROOT,
    effective_reserved_seeds,
    validate_reservation_manifest,
    validate_reserved_conditions,
)

SCENARIO_FIELDS = ("airport", "difficulty", "config_sha256")


@dataclass(frozen=True)
class UseCase:
    name: str
    reservation_path: Path
    conditions_path: Path
    config_root: Path

    def inputs(self) -> tuple[dict, dict]:
        reservation = read_json(self.reservation_path)
        conditions = read_json(self.conditions_path)
        validate_reservation_manifest(reservation)
        validate_reserved_conditions(conditions, reservation)
        return reservation, conditions

    def scenario_fields(self, conditions: dict | None = None) -> dict:
        stratum = (conditions or self.inputs()[1])["benchmark"]["stratum"]
        return {
            "airport": stratum["airport"],
            "difficulty": stratum["difficulty"],
            "config_sha256": stratum["scenario_config_sha256"],
        }


USE_CASES = {
    "mxp": UseCase(
        "mxp",
        CONTRACT_ROOT / "reserved-mxp-v1.json",
        CONTRACT_ROOT / "conditions-mxp-v2.json",
        CONTRACT_ROOT.parent / "simulation_config_mxp",
    ),
    "bgy": UseCase(
        "bgy",
        CONTRACT_ROOT / "bgy/reserved-v1.json",
        CONTRACT_ROOT / "bgy/conditions-v2.json",
        CONTRACT_ROOT.parent / "simulation_config_bgy",
    ),
}


def use_case(name: str) -> UseCase:
    if name not in USE_CASES:
        raise ValueError(f"unknown use case {name!r}; choose from {sorted(USE_CASES)}")
    return USE_CASES[name]


def use_case_of(scenario) -> str:
    """The registered use case whose frozen stratum the scenario reproduces."""
    actual = {key: scenario[key] for key in SCENARIO_FIELDS}
    for name, case in USE_CASES.items():
        if case.scenario_fields() == actual:
            return name
    raise ValueError("scenario matches no registered use case")


def expand_use_case(data: dict) -> dict:
    """Replace an input ``scenario.use_case`` with its explicit absolute fields."""
    scenario = data.get("scenario")
    if not isinstance(scenario, dict) or "use_case" not in scenario:
        return data
    case = use_case(scenario.pop("use_case"))
    scenario.update(case.scenario_fields(), config_root=str(case.config_root))
    data["reserved_manifest"] = str(case.reservation_path)
    data["reserved_conditions"] = str(case.conditions_path)
    return data


def _invariant_conditions(conditions) -> dict:
    """Everything a stream or learner depends on, i.e. all but the stratum and its hash chain."""
    benchmark = {key: value for key, value in conditions["benchmark"].items() if key != "stratum"}
    return {
        "physics": conditions["physics"],
        "native_artifact": conditions["native_artifact"],
        "purpose": conditions["purpose"],
        "version": conditions["version"],
        "benchmark": benchmark,
    }


def require_use_case_switch(saved, requested) -> None:
    """A switch keeps every excluded seed, the native artifact, physics and preprocessing."""
    if effective_reserved_seeds(saved.reserved_manifest) != effective_reserved_seeds(
        requested.reserved_manifest
    ) or content_digest(dict(saved.reserved_manifest["generation"])) != content_digest(
        dict(requested.reserved_manifest["generation"])
    ):
        raise ValueError("use case switch changes reserved seeds or their generation")
    if content_digest(_invariant_conditions(saved.reserved_conditions)) != content_digest(
        _invariant_conditions(requested.reserved_conditions)
    ):
        raise ValueError("use case switch changes physics, native artifact or preprocessing")


def with_use_case(run, name: str):
    """The same run bound to another use case's scenario fields and reservation sections."""
    from training.run_config import reseal

    case = use_case(name)
    reservation, conditions = case.inputs()
    scenario = {
        **dict(run.scenario),
        **case.scenario_fields(conditions),
        "config_root": str(case.config_root),
    }
    rebound = reseal(
        run, scenario=scenario, reserved_manifest=reservation, reserved_conditions=conditions
    )
    require_use_case_switch(run, rebound)
    return rebound
