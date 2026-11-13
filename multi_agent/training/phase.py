"""One phase mechanism: a whitelisted set of config changes, validated and recorded as lineage."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from common.hashing import content_digest
from common.sealed import consume
from training.checkpoint_boundary import CheckpointBoundaryError
from training.config_schema import TABLE, check, validate_cross
from training.run_config import ResolvedRunConfig, reseal, semantic_section, without_retired
from training.use_case import (
    SCENARIO_FIELDS,
    require_use_case_switch,
    use_case,
    use_case_of,
    with_use_case,
)

# A phase restores all learner state. Value-normalizer statistics are deliberately inherited,
# also across aircraft_count changes; they re-adapt.
PHASE_FIELDS: dict[str, tuple[str, str, type]] = {
    "lr": ("optimizer", "learning_rate", float),
    "entropy": ("optimizer", "entropy_coeff", float),
    "train_batch_size": ("optimizer", "train_batch_size", int),
    "minibatch_size": ("optimizer", "minibatch_size", int),
    "aircraft_count": ("scenario", "aircraft_count", int),
    "min_aircraft_count": ("scenario", "min_aircraft_count", int),
    "max_initial_deviation_s": ("scenario", "max_initial_deviation_s", int),
    "failure_replay_fraction": ("scenario", "failure_replay_fraction", float),
    "forecast_min_lookahead_s": ("scenario", "forecast_min_lookahead_s", int),
    "action_mask": ("scenario", "custom_action_mask", str),
    "generator_variant": ("scenario", "generator_variant", str),
    "training_success_s": ("thresholds", "training_success_s", float),
    "unresolved_failure_cost": ("reward", "unresolved_failure_cost", float),
    "conflict_noop_cost": ("reward", "conflict_noop_cost", float),
    "conflict_weight": ("reward", "max_for_zero_infringement", float),
}
# Rebinds the scenario stratum fields and both reservation sections to another registered use
# case with the same excluded seeds, physics, native artifact and preprocessing; observation and
# model shapes are use-case independent.
USE_CASE = "use_case"
_USE_CASE_FREE = {("scenario", key) for key in SCENARIO_FIELDS} | {
    ("reserved_manifest", None),
    ("reserved_conditions", None),
}
# Rollout-inference placement may move with any phase (resource_profile validates the new one);
# the learner, its device and the runner topology that keys the seed streams stay fixed.
INFERENCE_PLACEMENT = ("inference_device", "gpus_per_runner", "inference_gpu_uuid")
LEARNER_KEYS = {"learning_rate", "entropy_coeff"}
_UNSEALED = {"run_id", "phase_id", "execution", "semantic_sections", "semantic_digest"}


def apply_sets(
    run: ResolvedRunConfig, sets: Sequence[str]
) -> tuple[ResolvedRunConfig, dict[str, list]]:
    updates: dict[str, dict[str, Any]] = {}
    change: dict[str, list] = {}
    parsed = []
    for item in sets:
        alias, separator, raw = item.partition("=")
        if not separator or alias in {name for name, _ in parsed}:
            raise ValueError(f"--set requires one KEY=VALUE per field, got {item!r}")
        if alias != USE_CASE and alias not in PHASE_FIELDS:
            raise ValueError(
                f"--set {alias!r} is not phase-changeable; allowed: "
                f"{sorted([*PHASE_FIELDS, USE_CASE])}"
            )
        parsed.append((alias, raw))
    for alias, raw in parsed:
        if alias == USE_CASE:
            change[alias] = [use_case_of(run.scenario), use_case(raw).name]
            run = with_use_case(run, change[alias][1])
    for alias, raw in parsed:
        if alias == USE_CASE:
            continue
        section, key, cast = PHASE_FIELDS[alias]
        value = cast(raw)
        if section in TABLE:
            check(TABLE[section][key], value, f"{section}.{key}")
        base = updates.setdefault(section, dict(getattr(run, section)))
        change[alias] = [base[key], value]
        base[key] = value
    if not updates:
        return run, change
    # reseal re-runs __post_init__ (campaign stratum check, freezing); combined fields recheck here.
    phased = reseal(run, **updates)
    validate_cross(phased.to_dict())
    if "optimizer" in updates:
        from training.learner_checkpoint import ATCPPOLearner

        ATCPPOLearner._validate_overrides(
            {key: phased.optimizer[key] for key in LEARNER_KEYS & set(updates["optimizer"])}
        )
    return phased, change


def saved_run(config: Mapping[str, Any]) -> ResolvedRunConfig:
    """Rebuild a checkpoint's config, rejecting tampered section digests or role payloads."""
    config = without_retired(config)
    run = ResolvedRunConfig(**config)
    hashes = {
        key: content_digest(semantic_section(key, value))
        for key, value in config.items()
        if key not in _UNSEALED
    }
    if hashes != run.semantic_sections or content_digest(hashes) != run.semantic_digest:
        raise CheckpointBoundaryError("invalid checkpoint semantic contract")
    for role, payload in (
        ("env", run.env_payload()),
        ("model", run.model_payload()),
        ("reward", run.reward_payload()),
        ("metrics", run.metrics_payload()),
    ):
        consume(payload, run.semantic_digest, role=role)
    return run


def _comparable(run: ResolvedRunConfig, section: str, ignored: set[str]) -> Any:
    value = semantic_section(section, getattr(run, section))
    if not isinstance(value, Mapping):
        return value
    return {key: item for key, item in value.items() if key not in ignored}


@dataclass(frozen=True)
class PhaseTransition:
    source_digest: str
    target_digest: str
    changed: dict[str, list]

    @property
    def overrides(self) -> dict[str, float]:
        """Learner hyperparameters the restored optimizer must adopt."""
        return {
            key: new
            for alias, (_, new) in self.changed.items()
            if alias in PHASE_FIELDS and (key := PHASE_FIELDS[alias][1]) in LEARNER_KEYS
        }

    @property
    def moves_inference(self) -> bool:
        return "resources.inference_device" in self.changed

    @classmethod
    def between(
        cls,
        saved: ResolvedRunConfig,
        requested: ResolvedRunConfig,
        changes: dict[str, list],
        *,
        mode: str = "phase",
    ) -> "PhaseTransition":
        """resume: identical config. phase: same run, new phase name, only whitelisted fields differ."""
        if mode == "resume":
            same = (saved.run_id, saved.phase_id, saved.semantic_digest) == (
                requested.run_id,
                requested.phase_id,
                requested.semantic_digest,
            )
            if changes or not same:
                raise CheckpointBoundaryError("resume contract mismatch")
            return cls(saved.semantic_digest, saved.semantic_digest, {})
        if mode != "phase":
            raise CheckpointBoundaryError("checkpoint mode must be resume or phase")
        if not changes or saved.run_id != requested.run_id or saved.phase_id == requested.phase_id:
            raise CheckpointBoundaryError("phase requires explicit changes and a new phase name")
        free = {PHASE_FIELDS[alias][:2] for alias in changes if alias != USE_CASE}
        free |= {("resources", key) for key in INFERENCE_PLACEMENT}
        if USE_CASE in changes:
            free |= _USE_CASE_FREE
        for section in saved.semantic_sections:
            if (section, None) in free:
                continue
            ignored = {key for name, key in free if name == section}
            old, new = (
                _comparable(saved, section, ignored),
                _comparable(requested, section, ignored),
            )
            if old != new:
                keys = (
                    sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
                    if isinstance(old, Mapping)
                    else []
                )
                raise CheckpointBoundaryError(
                    f"phase changes {section!r} beyond the whitelisted fields: {keys}"
                )
        if USE_CASE in changes:
            if use_case_of(requested.scenario) != changes[USE_CASE][1]:
                raise CheckpointBoundaryError(
                    "phase change use_case differs from the requested config"
                )
            try:
                require_use_case_switch(saved, requested)
            except ValueError as error:
                raise CheckpointBoundaryError(str(error)) from error
        for alias, (_, new) in changes.items():
            if alias == USE_CASE:
                continue
            section, key, _ = PHASE_FIELDS[alias]
            if getattr(requested, section)[key] != new:
                raise CheckpointBoundaryError(
                    f"phase change {alias} differs from the requested config"
                )
        placement = {
            f"resources.{key}": [saved.resources.get(key), requested.resources.get(key)]
            for key in INFERENCE_PLACEMENT
            if saved.resources.get(key) != requested.resources.get(key)
        }
        return cls(saved.semantic_digest, requested.semantic_digest, {**changes, **placement})

    def validate_stream_envelope(
        self, saved: Mapping[str, Any], current: Mapping[str, Any]
    ) -> None:
        """Validate this known envelope and its invariant stream before transport."""
        if (
            set(saved) != set(current)
            or saved["semantic_digest"] != self.source_digest
            or current["semantic_digest"] != self.target_digest
        ):
            raise CheckpointBoundaryError("phase stream contract mismatch")
        # A use-case switch rebinds the reservation documents; between() proved their excluded
        # seeds identical, and the stream's reserved-seed hash is compared below.
        rebound = {"reservation_sha256", "conditions_sha256"} if USE_CASE in self.changed else set()
        for key in {"version", "reservation_sha256", "conditions_sha256"} - rebound:
            if saved[key] != current[key]:
                raise CheckpointBoundaryError("phase stream reservation mismatch")
        stable = set(current["stream"]) - {"cursor", "rejected"}
        if set(saved["stream"]) != set(current["stream"]) or any(
            saved["stream"][key] != current["stream"][key] for key in stable
        ):
            raise CheckpointBoundaryError("phase stream coordinates mismatch")


def extend_lineage(
    parent: list[dict],
    run: ResolvedRunConfig,
    *,
    parent_sha: str,
    lifetime_iterations: int,
    changes: dict,
) -> list[dict]:
    return [
        *parent,
        {
            "phase_id": run.phase_id,
            "parent_checkpoint_sha256": parent_sha,
            "lifetime_iterations": lifetime_iterations,
            "changes": changes,
        },
    ]
