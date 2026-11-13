"""Declarative run_config validation: one rule table, one cross-field list, one walker."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from actions.action_table import NUM_ACTIONS
from common.device import DEVICES
from common.hashing import is_sha256
from training.generator_variant import GENERATOR_VARIANTS

_PORTABLE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
REQUIRED = object()


def exact_keys(data: dict, expected: set[str], label: str) -> None:
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"{label} requires exactly {sorted(expected)}")


def positive(value, label: str, *, zero=False, integer=False) -> None:
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(
            f"{label} must be a {'non-negative' if zero else 'positive'} finite {'integer' if integer else 'number'}"
        )


def require_sha256(value: object, label: str) -> None:
    if not is_sha256(value):
        raise ValueError(f"{label} must be a SHA-256 hash")


_NUMERIC = {
    "int_pos": {"integer": True},
    "int_nonneg": {"integer": True, "zero": True},
    "num_pos": {},
    "num_nonneg": {"zero": True},
}
_VALID: dict[str, Callable[[Any, Any], bool]] = {
    "str": lambda value, _: isinstance(value, str) and bool(value),
    "ident": lambda value, _: (
        isinstance(value, str) and value not in {"", ".", ".."} and _PORTABLE.issuperset(value)
    ),
    "enum": lambda value, members: value in members,
    "const": lambda value, expected: type(value) is type(expected) and value == expected,
    # One 0/1 character per action with no-command allowed; "" keeps the airport mask from Config.
    "action_mask": lambda value, _: (
        value == ""
        or (
            isinstance(value, str)
            and len(value) == NUM_ACTIONS
            and set(value) <= {"0", "1"}
            and value[0] == "1"
        )
    ),
    "gpu_uuid": lambda value, _: (
        value is None or (isinstance(value, str) and value.startswith("GPU-"))
    ),
}


@dataclass(frozen=True)
class Rule:
    kind: str  # a _NUMERIC or _VALID key, or "sha256"
    arg: Any = None  # enum members or the const value
    default: Any = REQUIRED


def check(rule: Rule, value: Any, name: str) -> Any:
    if rule.kind in _NUMERIC:
        positive(value, name, **_NUMERIC[rule.kind])
    elif rule.kind == "sha256":
        require_sha256(value, name)
    elif not _VALID[rule.kind](value, rule.arg):
        expected = rule.kind if rule.arg is None else f"{rule.kind} {rule.arg!r}"
        raise ValueError(f"{name} is invalid: {value!r} (expected {expected})")
    return value


R = Rule
TABLE: dict[str, dict[str, Rule]] = {
    "topology": {
        "phase_slot": R("int_nonneg"),
        "env_runners": R("int_nonneg"),
        "envs_per_runner": R("int_pos"),
    },
    "scenario": {
        "airport": R("str"),
        "difficulty": R("str"),
        "aircraft_count": R("int_pos"),
        # Training draws each scenario's count uniformly from [min, aircraft_count]; 0 keeps it fixed.
        "min_aircraft_count": R("int_nonneg", default=0),
        # Training rejects scenarios with any pre-autosolve |deviation| above this; 0 keeps all.
        "max_initial_deviation_s": R("int_nonneg", default=0),
        # Share of training resets that replay a recently failed training seed; 0 disables.
        "failure_replay_fraction": R("num_nonneg", default=0.0),
        # Minimum planning-forecast lookahead near the episode end, so the policy sees conflicts
        # the no-command verification will meet after the horizon; 0 forecasts to the end only.
        "forecast_min_lookahead_s": R("int_nonneg", default=0),
        # Replaces Config.CUSTOM_ACTION_MASK_BY_AIRPORT for the scenario airport, binding the
        # mask to the run and its checkpoints; "" keeps the Config mask.
        "custom_action_mask": R("action_mask", default=""),
        # Loads an authenticated one-setting derivative of the frozen MXP configuration
        # (training.generator_variant); the stratum keeps the parent hash. "" loads config_root.
        "generator_variant": R("enum", GENERATOR_VARIANTS, default=""),
        "config_root": R("str"),
        "config_sha256": R("sha256"),
        "episode_steps": R("int_pos"),
        "start_offset_max_s": R("int_nonneg"),
        "verification": R("const", "native_generation"),
        "noop_verification_max_seconds": R("int_pos", default=14400),
    },
    "thresholds": {
        "training_success_s": R("num_pos"),
        "operational_success_s": R("num_pos"),
        "lockout_s": R("num_pos"),
        "auto_solve_s": R("num_pos"),
    },
    "optimizer": {
        "learning_rate": R("num_pos"),
        "gamma": R("num_pos"),
        "gae_lambda": R("num_pos"),
        "clip_param": R("num_pos"),
        "vf_clip_param": R("num_pos"),
        "vf_loss_coeff": R("num_pos"),
        "entropy_coeff": R("num_nonneg"),
        "grad_clip": R("num_pos"),
        "train_batch_size": R("int_pos"),
        "minibatch_size": R("int_pos"),
        "epochs": R("int_pos"),
    },
    "resources": {
        "backend": R("enum", DEVICES),
        "learner_device": R("enum", DEVICES),
        "num_learners": R("int_pos"),
        "cpus_per_runner": R("num_pos"),
        "cpus_per_learner": R("num_pos"),
        "gpus_per_learner": R("num_nonneg"),
        "inference_device": R("enum", DEVICES, default="cpu"),
        "gpus_per_runner": R("num_nonneg", default=0.0),
        "learner_gpu_uuid": R("gpu_uuid", default=None),
        "inference_gpu_uuid": R("gpu_uuid", default=None),
        "rollout_backend": R(
            "enum", ("scalar", "native_cpu_batch", "native_cuda_batch"), default="scalar"
        ),
    },
    "cadences": {
        "checkpoint_iterations": R("int_pos"),
        "evaluation_iterations": R("int_pos"),
    },
}
TOP = {
    "version": R("const", 1),
    "run_id": R("ident"),
    "phase_id": R("ident"),
    "root_seed": R("int_nonneg"),
}
# Validated elsewhere: reservation documents, `_options` for features/reward, execution controls.
OTHER_KEYS = {"execution", "reserved_manifest", "reserved_conditions", "features", "reward"}
TOP_LEVEL_KEYS = set(TOP) | set(TABLE) | OTHER_KEYS


def validate_section(name: str, section: Mapping[str, Any]) -> dict[str, Any]:
    """Defaults fill only keys whose rule has one; unknown or missing keys fail."""
    rules = TABLE[name]
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} must be an object")
    section = dict(section)
    for key, rule in rules.items():
        if rule.default is not REQUIRED:
            section.setdefault(key, rule.default)
    exact_keys(section, set(rules), name)
    return {key: check(rule, section[key], f"{name}.{key}") for key, rule in rules.items()}


# (message, predicate over the fully resolved config dict); each mirrors one former
# imperative check in load_run_config.
CROSS: list[tuple[str, Callable[[Mapping[str, Any]], bool]]] = [
    (
        "thresholds.operational_success_s must be 60.0: operational success means every arrival target completed "
        "with absolute arrival error <=60 seconds (inclusive) and no realized separation loss or exit",
        lambda c: c["thresholds"]["operational_success_s"] == 60.0,
    ),
    (
        "root_seed exceeds Ray global/worker seed range for this topology",
        # Ray 2.53 initializes NumPy with root + worker index (+1M for eval).
        lambda c: c["root_seed"] <= 2**32 - 1 - max(c["topology"]["env_runners"], 1_000_001),
    ),
    (
        "features.knee_s must not exceed cap_s",
        lambda c: c["features"]["knee_s"] <= c["features"]["cap_s"],
    ),
    (
        "features.architecture.neighbor_heads must divide neighbor_model",
        lambda c: (
            c["features"]["architecture"]["neighbor_model"]
            % c["features"]["architecture"]["neighbor_heads"]
            == 0
        ),
    ),
    (
        "unsupported features.deviation_encoding or features.neighbor_selection",
        lambda c: (
            c["features"]["deviation_encoding"] in {"linear", "signed_log"}
            and c["features"]["neighbor_selection"] == "risk"
        ),
    ),
    (
        # One live failure charge uses the original target-count/baseline scale;
        # ordinary global/acting allocation divides it, never multiplies by actors.
        "reward.count_baseline and reward.realized_failure_cost must be positive",
        lambda c: c["reward"]["count_baseline"] > 0 and c["reward"]["realized_failure_cost"] > 0,
    ),
    (
        "scenario.min_aircraft_count must not exceed aircraft_count",
        lambda c: c["scenario"]["min_aircraft_count"] <= c["scenario"]["aircraft_count"],
    ),
    (
        "scenario.failure_replay_fraction must be at most one",
        lambda c: c["scenario"]["failure_replay_fraction"] <= 1,
    ),
    (
        "reward.global_share must be at most one",
        lambda c: c["reward"]["global_share"] <= 1,
    ),
    (
        "unsupported reward.count_scaling or reward.coverage_accounting",
        lambda c: (
            c["reward"]["count_scaling"] == "initial_count_over_baseline"
            and c["reward"]["coverage_accounting"] == "fixed_target_potential_v2"
        ),
    ),
    (
        "optimizer.minibatch_size must divide train_batch_size",
        lambda c: (
            c["optimizer"]["minibatch_size"] <= c["optimizer"]["train_batch_size"]
            and c["optimizer"]["train_batch_size"] % c["optimizer"]["minibatch_size"] == 0
        ),
    ),
    (
        "recurrent optimizer.minibatch_size must be at least features.architecture.max_seq_len",
        lambda c: c["optimizer"]["minibatch_size"] >= c["features"]["architecture"]["max_seq_len"],
    ),
    (
        "optimizer.gamma and optimizer.gae_lambda must be at most one",
        lambda c: c["optimizer"]["gamma"] <= 1 and c["optimizer"]["gae_lambda"] <= 1,
    ),
    (
        "resources.learner_device, resources.backend and gpus_per_learner disagree",
        lambda c: (
            (c["resources"]["learner_device"] == "cuda") == (c["resources"]["gpus_per_learner"] > 0)
            and (c["resources"]["learner_device"] != "cuda" or c["resources"]["backend"] == "cuda")
        ),
    ),
]


def validate_cross(config: Mapping[str, Any]) -> None:
    for message, holds in CROSS:
        if not holds(config):
            raise ValueError(message)
