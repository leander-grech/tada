"""Immutable resolved experiment contracts; no process-global Config transport.

Ownership is explicit: env owns scenario/threshold/seed verification; model owns
features, dimensions and preprocessing; reward owns coefficients and coverage;
metrics owns outcomes; lifecycle owns optimizer, resources and cadences. A
projection attests its own bytes plus the shared semantic identity. Consumers
must validate it and report the subset they actually consume.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from common.hashing import FrozenDict, content_digest, file_sha256, freeze_json
from common.jsonio import read_json
from training import versions
from training.config_defaults import FEATURE_DEFAULTS, REWARD_DEFAULTS
from training.config_schema import (
    TABLE,
    TOP,
    TOP_LEVEL_KEYS,
    check,
    exact_keys,
    positive,
    validate_cross,
    validate_section,
)
from training.reservations import (
    validate_reservation_manifest as validate_reservation_manifest,
)
from training.reservations import (
    validate_reserved_conditions as validate_reserved_conditions,
)


def _options(data: dict, defaults: dict, label: str) -> dict:
    if not isinstance(data, dict) or set(data) - set(defaults):
        raise ValueError(f"unknown {label} option")
    result = dict(defaults)
    for key, default in defaults.items():
        value = data.get(key, default)
        if isinstance(default, dict):
            value = _options(value, default, f"{label}.{key}")
        elif isinstance(default, str):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label}.{key} must be a non-empty string")
        else:
            positive(value, f"{label}.{key}", zero=label == "reward", integer=type(default) is int)
        result[key] = value
    return result


def scenario_config_digest(root: Path) -> str:
    """Hash resolved scenario/physics input bytes, including configuration links."""
    root = root.resolve(strict=True)
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError("scenario configuration is empty")
    return content_digest(files)


def validate_campaign_stratum(scenario, manifest, conditions):
    """Campaign runs train on the reserved benchmark airport, difficulty and scenario inputs.

    A different aircraft_count is allowed only with the benchmark episode shape.
    """
    if manifest["purpose"] != "campaign":
        return
    benchmark = conditions["benchmark"]
    stratum = benchmark["stratum"]
    expected = {
        "airport": stratum["airport"],
        "difficulty": stratum["difficulty"],
        "config_sha256": stratum["scenario_config_sha256"],
    }
    if scenario["aircraft_count"] != stratum["aircraft_count"]:
        expected |= {
            "episode_steps": benchmark["episode_steps"],
            "start_offset_max_s": benchmark["start_offset_max_s"],
        }
    for key, value in expected.items():
        if scenario[key] != value:
            raise ValueError(f"campaign primary stratum mismatch: {key}")


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Fully resolved values, independent of mutable global Config."""

    run_id: str
    phase_id: str
    root_seed: int
    reserved_manifest: FrozenDict
    reserved_conditions: FrozenDict
    topology: FrozenDict
    scenario: FrozenDict
    thresholds: FrozenDict
    features: FrozenDict
    reward: FrozenDict
    optimizer: FrozenDict
    resources: FrozenDict
    cadences: FrozenDict
    versions: FrozenDict
    semantic_sections: FrozenDict
    semantic_digest: str
    execution: FrozenDict
    version: int = 1

    def __post_init__(self) -> None:
        validate_campaign_stratum(self.scenario, self.reserved_manifest, self.reserved_conditions)
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, freeze_json(getattr(self, name)))

    @property
    def reserved_seeds(self) -> frozenset[int]:
        return validate_reservation_manifest(self.reserved_manifest)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def _payload(self, role: str, **values) -> FrozenDict:
        payload = {
            "role": role,
            "semantic_digest": self.semantic_digest,
            **values,
        }
        return FrozenDict(payload)

    def env_payload(self) -> FrozenDict:
        return self._payload(
            "env",
            root_seed=self.root_seed,
            topology=self.topology,
            scenario=self.scenario,
            thresholds=self.thresholds,
            features=self.features,
            reward=self.reward,
            reserved_manifest=self.reserved_manifest,
            reserved_conditions=self.reserved_conditions,
        )

    def model_payload(self) -> FrozenDict:
        return self._payload("model", features=self.features)

    def reward_payload(self) -> FrozenDict:
        return self._payload("reward", reward=self.reward, thresholds=self.thresholds)

    def metrics_payload(self) -> FrozenDict:
        return self._payload("metrics", thresholds=self.thresholds)


def reseal(run: ResolvedRunConfig, **sections: Any) -> ResolvedRunConfig:
    """Replace config sections and recompute the section digests and semantic digest."""
    digests = dict(
        run.semantic_sections,
        **{key: content_digest(semantic_section(key, value)) for key, value in sections.items()},
    )
    return replace(
        run, **sections, semantic_sections=digests, semantic_digest=content_digest(digests)
    )


_OPTIONAL_SCENARIO_FIELDS = frozenset(
    {
        "min_aircraft_count",
        "max_initial_deviation_s",
        "failure_replay_fraction",
        "forecast_min_lookahead_s",
        "custom_action_mask",
        "generator_variant",
    }
)
_DISABLED = (0, "")
_OPTIONAL_REWARD_FIELDS = frozenset({"unresolved_failure_cost", "conflict_noop_cost"})
# Disabled saved action-shield fields are normalized without changing semantic identity.
_RETIRED = {"scenario": ("action_shield", "none"), "reward": ("shield_veto_cost", 0)}


def without_retired(config: Mapping[str, Any]) -> dict[str, Any]:
    """A saved run config without retired fields; a run that enabled one is unsupported."""
    config = dict(config)
    for section, (name, disabled) in _RETIRED.items():
        fields = dict(config[section])
        if fields.pop(name, disabled) != disabled:
            raise ValueError(f"saved run uses the retired {section}.{name}")
        config[section] = fields
    return config


def semantic_section(key: str, value: Any) -> Any:
    """A section's semantic content: scenario.config_root is a host path, never identity.

    Disabled optional fields (min_aircraft_count, max_initial_deviation_s, failure_replay_fraction,
    forecast_min_lookahead_s, unresolved_failure_cost, conflict_noop_cost 0; custom_action_mask
    and generator_variant "") are identical to configs that predate them.
    """
    if key == "scenario":
        return {
            name: item
            for name, item in value.items()
            if name != "config_root"
            and not (name in _OPTIONAL_SCENARIO_FIELDS and item in _DISABLED)
        }
    if key == "reward":
        return {
            name: item
            for name, item in value.items()
            if not (name in _OPTIONAL_REWARD_FIELDS and item == 0)
        }
    return value


def load_run_config(path: Path) -> ResolvedRunConfig:
    """Read, validate and freeze a JSON run and its content-addressed reservation."""
    from training.config_overlay import read_run_input
    from training.use_case import expand_use_case

    data, path = read_run_input(Path(path))
    expand_use_case(data)
    execution = data.setdefault("execution", {})
    if not isinstance(execution, dict) or set(execution) - {
        "stop_iterations",
        "output_root",
        "development_manifest",
    }:
        raise ValueError("unsupported execution controls")
    execution = {
        "stop_iterations": None,
        "output_root": "runs",
        **execution,
    }
    if execution["stop_iterations"] is not None:
        positive(execution["stop_iterations"], "execution.stop_iterations", integer=True)
    if not isinstance(execution["output_root"], str) or not execution["output_root"]:
        raise ValueError("execution.output_root must be a non-empty path")
    execution["output_root"] = str((path.parent / execution["output_root"]).resolve())
    if "development_manifest" in execution:
        if (
            not isinstance(execution["development_manifest"], str)
            or not execution["development_manifest"]
        ):
            raise ValueError("execution.development_manifest requires an explicit path")
        execution["development_manifest"] = str(
            (path.parent / execution["development_manifest"]).resolve()
        )
    data["execution"] = execution
    exact_keys(data, TOP_LEVEL_KEYS, "run config")
    for key, rule in TOP.items():
        check(rule, data[key], key)
    manifest = read_json(path.parent / data["reserved_manifest"])
    validate_reservation_manifest(manifest)
    data["reserved_manifest"] = manifest
    conditions = read_json(path.parent / data["reserved_conditions"])
    validate_reserved_conditions(conditions, manifest)
    data["reserved_conditions"] = conditions
    for name in TABLE:
        data[name] = validate_section(name, data[name])
    validate_campaign_stratum(data["scenario"], manifest, conditions)
    data["features"] = _options(data["features"], FEATURE_DEFAULTS, "features")
    data["reward"] = _options(data["reward"], REWARD_DEFAULTS, "reward")
    validate_cross(data)
    data["versions"] = versions.current()
    data["semantic_sections"] = {
        key: content_digest(semantic_section(key, value))
        for key, value in data.items()
        if key not in {"run_id", "phase_id", "execution"}
    }
    data["semantic_digest"] = content_digest(data["semantic_sections"])
    scenario = data["scenario"]
    scenario["config_root"] = str((path.parent / scenario["config_root"]).resolve())
    return ResolvedRunConfig(**{key: freeze_json(value) for key, value in data.items()})
