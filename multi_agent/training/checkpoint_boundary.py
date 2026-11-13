"""Exact completed-scenario sidecars, not independently published checkpoints.

Call only on a quiescent owned Algorithm. Restore full RLlib/learner state first,
then this sidecar while environments remain guarded, then activate explicitly.
"""

from copy import deepcopy
from functools import partial

import ray

from training.checkpoint_process import (
    capture_process_rng as capture_process_rng,
)
from training.checkpoint_process import (
    restore_process_rng as restore_process_rng,
)
from training.checkpoint_process import (
    validate_process_rng,
)

RECEIPT_KEY = "atc_checkpoint_receipt_v1"
BOUNDARY_VERSION = 1


class CheckpointBoundaryError(ValueError):
    """The runtime cannot prove an exact, fully learned scenario boundary."""


def validate_receipts(receipts):
    if not isinstance(receipts, dict):
        raise CheckpointBoundaryError("invalid learned receipts")
    for identity, state in receipts.items():
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(state, dict)
            or set(state) != {"sequence", "env_steps"}
        ):
            raise CheckpointBoundaryError("invalid learned receipt identity")
        if any(type(v) is not int or v < 0 for v in state.values()):
            raise CheckpointBoundaryError("invalid learned receipt counters")


def advance_learned_receipts(previous, episodes):
    """Acknowledge complete receipt sets once per successful learner update."""
    validate_receipts(previous)
    result = deepcopy(previous)
    by_identity = {}
    for episode in episodes:
        receipt = episode.custom_data.get(RECEIPT_KEY)
        if receipt is None:
            continue  # Direct learner diagnostics do not claim owned-runner provenance.
        if not isinstance(receipt, dict) or set(receipt) != {
            "identity",
            "sequence",
            "env_steps",
            "episode_id",
            "start",
            "end",
        }:
            raise CheckpointBoundaryError("unknown trajectory receipt")
        if (
            receipt["episode_id"] != episode.id_
            or receipt["start"] != episode.env_t_started
            or receipt["end"] != episode.env_t
            or receipt["env_steps"] != len(episode)
        ):
            raise CheckpointBoundaryError("receipt does not cover the transported episode chunk")
        validate_receipts({receipt["identity"]: {k: receipt[k] for k in ("sequence", "env_steps")}})
        by_identity.setdefault(receipt["identity"], []).append(receipt)
    for identity, receipts in by_identity.items():
        cursor = result.setdefault(identity, {"sequence": 0, "env_steps": 0})
        for receipt in sorted(receipts, key=lambda item: item["sequence"]):
            if receipt["sequence"] != cursor["sequence"] + 1:
                raise CheckpointBoundaryError(
                    "noncontiguous or duplicate learned trajectory receipt"
                )
            cursor["sequence"] += 1
            cursor["env_steps"] += receipt["env_steps"]
    return result


def _manager_idle(manager):
    if manager is not None and manager.num_outstanding_async_reqs():
        raise CheckpointBoundaryError("in-flight asynchronous work is not checkpoint safe")


def _drain_runner_sync(manager):
    """Await only Ray's authenticated weight/connector set_state acknowledgements."""
    pending = manager.num_outstanding_async_reqs()
    if pending != manager.num_outstanding_async_reqs(tag="set_state"):
        raise CheckpointBoundaryError("in-flight sampling/unknown work is not checkpoint safe")
    if pending:
        for result in manager.fetch_ready_async_reqs(tags="set_state", timeout_seconds=None):
            if not result.ok:
                raise result.get()
            if result.get() is not None:
                raise CheckpointBoundaryError("unexpected asynchronous set_state result")
    _manager_idle(manager)


def _supported_algorithm(algorithm, run):
    from training.algorithm import ATCPPO
    from training.runtime_gate import verify_algorithm_selection

    if type(algorithm) is not ATCPPO or ray.__version__ != "2.53.0":
        raise CheckpointBoundaryError("unsupported synchronous Algorithm/runtime")
    cfg = algorithm.config
    if (
        cfg.num_learners != 1
        or run.resources["num_learners"] != 1
        or cfg.num_envs_per_env_runner != 1
        or cfg.num_env_runners != run.topology["env_runners"]
        or cfg.batch_mode != "complete_episodes"
        or cfg.sample_timeout_s is not None
        or cfg.evaluation_parallel_to_training
        or algorithm.learner_group.is_local
    ):
        raise CheckpointBoundaryError("unsupported synchronous checkpoint topology")
    verify_algorithm_selection(algorithm, run)
    _manager_idle(algorithm.learner_group._worker_manager)
    for group in (algorithm.env_runner_group, algorithm.eval_env_runner_group):
        if group is None:
            raise CheckpointBoundaryError("missing checkpoint runner group")
        _drain_runner_sync(group._worker_manager)


def _capture_runner(runner):
    return {"runner": runner.capture_boundary_state(), "rng": capture_process_rng()}


def _capture_learner(learner):
    return {"effective": learner._effective_state(), "rng": capture_process_rng()}


def _learner_results(group, fn):
    values = []
    for result in group.foreach_learner(fn, healthy_only=False):
        if not result.ok:
            raise result.get()
        values.append(result.get())
    if len(values) != 1:
        raise CheckpointBoundaryError("missing or extra learner process sidecar")
    return values[0]


def _groups(algorithm):
    return (
        ("training", algorithm.env_runner_group),
        ("operational_evaluation", algorithm.eval_env_runner_group),
    )


def capture_boundary_state(algorithm, run) -> dict:
    """Capture pickle-safe per-process state after all samples have been learned."""
    _supported_algorithm(algorithm, run)
    processes = {}
    local = {}
    for role, group in _groups(algorithm):
        reports = group.foreach_env_runner(
            _capture_runner, local_env_runner=False, healthy_only=False
        )
        expected = run.topology["env_runners"] if role == "training" else 1
        if len(reports) != expected:
            raise CheckpointBoundaryError("missing runner process sidecar")
        for report in reports:
            identity = report["runner"]["identity"]
            if identity in processes or not identity.startswith(role + "/"):
                raise CheckpointBoundaryError("duplicate or incompatible runner identity")
            processes[identity] = report
        if group.local_env_runner is not None:
            local[role] = group.local_env_runner.capture_boundary_state()
    learner = _learner_results(algorithm.learner_group, _capture_learner)
    learned = learner["effective"]["learned"]
    expected_learned = {
        identity: item["runner"]["emitted"]
        for identity, item in processes.items()
        if identity.startswith("training/")
    }
    for identity, counters in expected_learned.items():
        if learned.get(identity, {"sequence": 0, "env_steps": 0}) != counters:
            raise CheckpointBoundaryError("sampled trajectory has not completed learning")
    if set(learned) - set(expected_learned):
        raise CheckpointBoundaryError("learner contains unknown runner receipts")
    processes["learner/0"] = learner
    from training.checkpoint_envelopes import validate_local_facades

    validate_local_facades(local)
    processes["driver"] = {"rng": capture_process_rng(), "local_runners": local}
    return {
        "version": BOUNDARY_VERSION,
        "semantic_digest": run.semantic_digest,
        "processes": processes,
    }


def _restore_runner(runner, *, processes, validate_only, transition=None):
    identity = runner.checkpoint_identity()
    try:
        saved = processes[identity]
        if not isinstance(saved, dict) or set(saved) != {"runner", "rng"}:
            raise CheckpointBoundaryError("unknown runner process sidecar")
        rng = saved["rng"]
        if transition is not None and transition.moves_inference:
            # The saved CUDA generator belonged to the runner's former inference device.
            rng = {**rng, "torch_cuda": None}
        validate_process_rng(rng)
        runner.restore_boundary_state(
            saved["runner"], validate_only=validate_only, transition=transition
        )
    except Exception as error:
        if validate_only:
            # Input rejection is not an actor failure: Ray otherwise marks the
            # healthy runner unavailable when a remote callback raises.
            return error
        raise
    if not validate_only:
        restore_process_rng(rng)
    return identity


def _restore_learner(learner, *, saved, validate_only, preflight=False, transition=None):
    expected = deepcopy(saved.get("effective"))
    if transition is not None and isinstance(expected, dict):
        for values in expected["modules"].values():
            values.update(transition.overrides)
    if set(saved) != {"effective", "rng"} or (
        not preflight and learner._effective_state() != expected
    ):
        raise CheckpointBoundaryError("restore complete learner state before boundary sidecar")
    validate_process_rng(saved["rng"])
    if not validate_only:
        restore_process_rng(saved["rng"])


def restore_boundary_state(algorithm, state, run, *, validate_only=False, transition=None) -> None:
    """Validate all processes before restoring guarded, same-contract streams/RNG."""
    _supported_algorithm(algorithm, run)
    expected_digest = run.semantic_digest if transition is None else transition.source_digest
    if transition is not None and transition.target_digest != run.semantic_digest:
        raise CheckpointBoundaryError("phase target contract mismatch")
    if (
        not isinstance(state, dict)
        or set(state) != {"version", "semantic_digest", "processes"}
        or state["version"] != BOUNDARY_VERSION
        or state["semantic_digest"] != expected_digest
    ):
        raise CheckpointBoundaryError("incompatible boundary sidecar envelope")
    expected = {"driver", "learner/0", "operational_evaluation/1"} | {
        f"training/{i}" for i in range(1, run.topology["env_runners"] + 1)
    }
    processes = state["processes"]
    if set(processes) != expected or set(processes["driver"]) != {"rng", "local_runners"}:
        raise CheckpointBoundaryError("missing or unknown process sidecar")
    learned = processes["learner/0"].get("effective", {}).get("learned")
    validate_receipts(learned)
    for identity in expected - {"driver", "learner/0", "operational_evaluation/1"}:
        emitted = processes[identity].get("runner", {}).get("emitted")
        validate_receipts({identity: emitted})
        if learned.get(identity, {"sequence": 0, "env_steps": 0}) != emitted:
            raise CheckpointBoundaryError("saved sampled/learned receipt mismatch")
    if set(learned) - {name for name in expected if name.startswith("training/")}:
        raise CheckpointBoundaryError("unknown saved learned runner identity")
    driver = processes["driver"]
    local = {
        role: group.local_env_runner
        for role, group in _groups(algorithm)
        if group.local_env_runner is not None
    }
    from training.checkpoint_envelopes import validate_local_facades

    validate_local_facades(driver["local_runners"])
    if set(driver["local_runners"]) != set(local):
        raise CheckpointBoundaryError("local runner facade mismatch")
    validate_process_rng(driver["rng"])
    for preflight in (True,) if validate_only else (True, False):
        for role, group in _groups(algorithm):
            actual = group.foreach_env_runner(
                partial(
                    _restore_runner,
                    processes=processes,
                    validate_only=preflight,
                    transition=transition,
                ),
                local_env_runner=False,
                healthy_only=False,
            )
            for result in actual:
                if isinstance(result, Exception):
                    raise result
            if set(actual) != {name for name in expected if name.startswith(role + "/")}:
                raise CheckpointBoundaryError("actual runner identity mismatch")
        for role, runner in local.items():
            runner.restore_boundary_state(
                driver["local_runners"][role], validate_only=preflight, transition=transition
            )
        _learner_results(
            algorithm.learner_group,
            partial(
                _restore_learner,
                saved=processes["learner/0"],
                validate_only=preflight,
                preflight=validate_only,
                transition=transition,
            ),
        )
    if not validate_only:
        restore_process_rng(driver["rng"])
