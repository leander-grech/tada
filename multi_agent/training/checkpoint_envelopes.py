"""Shared nonmutating checkpoint envelope contracts, independent of live actors."""

from outcomes import vocabulary
from training.checkpoint_boundary import CheckpointBoundaryError, validate_receipts
from training.seed_stream import ScenarioSeedStream

LOCAL_FACADE_ROLES = frozenset({"training", "operational_evaluation"})


def validate_stream_envelope(state):
    if (
        not isinstance(state, dict)
        or set(state)
        != {"version", "semantic_digest", "reservation_sha256", "conditions_sha256", "stream"}
        or type(state["version"]) is not int
        or state["version"] != 1
    ):
        raise CheckpointBoundaryError("unknown scenario stream envelope")
    for key in ("semantic_digest", "reservation_sha256", "conditions_sha256"):
        value = state[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise CheckpointBoundaryError("invalid scenario stream identity")
    ScenarioSeedStream.validate_state_structure(state["stream"])


def validate_environment_envelope(state):
    if (
        not isinstance(state, dict)
        or set(state) != {"version", "boundary", "stream", "audit"}
        or type(state["version"]) is not int
        or state["version"] != 1
        or state["boundary"] not in ("pristine", "closed")
    ):
        raise ValueError("unknown scenario boundary sidecar")
    validate_stream_envelope(state["stream"])
    audit = state["audit"]
    if state["boundary"] == "pristine":
        if audit is not None or state["stream"]["stream"]["cursor"] != 0:
            raise ValueError("pristine scenario has consumed draws")
    elif (
        not isinstance(audit, dict)
        or set(audit)
        != {
            "seed",
            "last_end_reason",
            "physically_complete",
            "world_sha256",
            "planning_world_sha256",
            "policy_steps",
        }
        or audit["last_end_reason"] not in vocabulary.ALL_END_REASONS
    ):
        raise ValueError("completed scenario audit is missing or unknown")
    if audit is not None:
        if (
            type(audit["physically_complete"]) is not bool
            or type(audit["policy_steps"]) is not int
            or audit["policy_steps"] <= 0
            or type(audit["seed"]) is not int
            or not 0 <= audit["seed"] < 2**31
        ):
            raise ValueError("invalid completed scenario audit values")
        for name in ("world_sha256", "planning_world_sha256"):
            value = audit[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("invalid completed scenario audit hash")


def validate_runner_envelope(state, *, identity, has_environment):
    if (
        not isinstance(state, dict)
        or set(state)
        != {
            "version",
            "identity",
            "emitted",
            "environment",
            "metrics",
            "connectors",
            "weights_seq_no",
        }
        or type(state["version"]) is not int
        or state["version"] != 1
        or state["identity"] != identity
    ):
        raise CheckpointBoundaryError("unknown or incompatible runner sidecar")
    validate_receipts({state["identity"]: state["emitted"]})
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

    if (
        type(state["weights_seq_no"]) is not int
        or state["weights_seq_no"] < 0
        or not isinstance(state["metrics"], dict)
        or set(state["metrics"]) != {"stats"}
    ):
        raise CheckpointBoundaryError("invalid runner metrics/weight sidecar")
    MetricsLogger().set_state(state["metrics"])
    if state["connectors"] != {"env_to_module": {}, "module_to_env": {}}:
        raise CheckpointBoundaryError("unsupported saved connector state")
    if has_environment:
        validate_environment_envelope(state["environment"])
    elif state["environment"] is not None:
        raise CheckpointBoundaryError("local facade cannot restore an environment")


def validate_local_facades(local):
    if not isinstance(local, dict) or set(local) != LOCAL_FACADE_ROLES:
        raise CheckpointBoundaryError("missing or unknown local runner facade")
    for role, state in local.items():
        validate_runner_envelope(state, identity=f"{role}/0", has_environment=False)
