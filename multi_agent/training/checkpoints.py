"""Complete public checkpoint bundles and fresh guarded continuation construction."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.hashing import canonical_json
from common.jsonio import read_json
from training.checkpoint_boundary import (
    CheckpointBoundaryError,
    capture_boundary_state,
    restore_boundary_state,
)
from training.checkpoint_bundle import publish_bundle
from training.checkpoint_manifest import (
    MANIFEST,
    build_manifest,
    checkpoint_digest,
    current_identity,
    require_compatible,
    require_tokens,
    verify_files,
)
from training.checkpoint_preflight import (
    module_payload_layout,
    preflight_checkpoint,
    validate_current_model_payload,
    validate_payload_structure,
)
from training.phase import INFERENCE_PLACEMENT, PhaseTransition, extend_lineage, saved_run
from training.ppo_config import build_algorithm
from training.run_config import ResolvedRunConfig, without_retired
from training.runtime_gate import activate_algorithm

__all__ = [
    "CheckpointBoundaryError",
    "LoadedCheckpoint",
    "capture_checkpoint",
    "inspect_checkpoint",
    "restore_checkpoint",
    "load_checkpoint_algorithm",
]
_REQUIRED = {MANIFEST, "algorithm.pkl", "boundary.pkl", "counters.json"}


def _counters(algorithm, boundary):
    effective = boundary["processes"]["learner/0"]["effective"]
    return {
        "version": 1,
        "lifetime_iterations": algorithm.training_iteration,
        "lifetime_env_steps": sum(item["env_steps"] for item in effective["learned"].values()),
        "lifetime_updates": effective["updates"],
        "optimizer_work": effective.get("optimizer_work", {}),
        "phase_start_iterations": getattr(algorithm, "_checkpoint_phase_start", 0),
        "curriculum": {"version": 1, "stage": None},
    }


def capture_checkpoint(algorithm, destination: Path, run, *, advertise_latest=True) -> Path:
    identity = current_identity(run)
    boundary = capture_boundary_state(algorithm, run)
    state = algorithm.get_state()
    counters = _counters(algorithm, boundary)

    def write(staging):
        for name, value in (("algorithm.pkl", state), ("boundary.pkl", boundary)):
            (staging / name).write_bytes(pickle.dumps(value, protocol=5))
        (staging / "counters.json").write_text(canonical_json(counters) + "\n")
        return build_manifest(
            run,
            provenance=current_provenance(identity),
            lineage=getattr(algorithm, "_checkpoint_lineage", []),
            execution_segment=getattr(algorithm, "_checkpoint_execution_segment", None),
            module_layout=module_payload_layout(state),
            identity=identity,
        )

    return publish_bundle(Path(destination), write, advertise_latest=advertise_latest)


def current_provenance(identity) -> dict[str, Any]:
    return {"wheel_sha256": identity["native"]["wheel_sha256"]}


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    manifest: dict[str, Any]
    counters: dict[str, Any]
    transition: PhaseTransition | None = None

    @property
    def run(self) -> ResolvedRunConfig:
        run = ResolvedRunConfig(**without_retired(self.manifest["run_config"]))
        if self.manifest.get("kind") == "inference_weights_v1":
            from training.inference_bundle import resolve_scenario

            return resolve_scenario(run)
        return run


def inspect_checkpoint(
    checkpoint, run=None, *, mode="resume", changes=None, scope="resume"
) -> LoadedCheckpoint:
    """The only checkpoint reader.

    scope "resume" (mode resume|phase) requires identical tokens, runtime, native
    artifact and catalog plus an authorized config transition; scope "weights"
    (inference) requires only identical tokens and file integrity.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink():
        raise CheckpointBoundaryError("checkpoint symlinks are unsupported")
    try:
        manifest = read_json(checkpoint / MANIFEST)
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CheckpointBoundaryError("unreadable or incomplete checkpoint bundle") from error
    if manifest.get("kind") == "inference_weights_v1":
        from training.inference_bundle import inspect_inference_bundle

        if scope != "weights":
            raise CheckpointBoundaryError("inference-only bundles cannot resume training")
        if run is not None or mode != "resume" or changes:
            raise CheckpointBoundaryError("weights scope reads the checkpoint's own config")
        inspect_inference_bundle(checkpoint, manifest)
        return LoadedCheckpoint(checkpoint, manifest, {})
    verify_files(checkpoint, files)
    if {path.name for path in checkpoint.iterdir()} != _REQUIRED:
        raise CheckpointBoundaryError("unknown or missing complete bundle sidecars")
    if scope == "weights":
        if run is not None or mode != "resume" or changes:
            raise CheckpointBoundaryError("weights scope reads the checkpoint's own config")
        require_tokens(manifest)
        run = saved = saved_run(manifest["run_config"])
        transition = None
    elif scope == "resume":
        if run is None:
            raise CheckpointBoundaryError("resume scope requires the requested run")
        require_compatible(
            manifest,
            current_identity(run),
            movable=INFERENCE_PLACEMENT if mode == "phase" else (),
        )
        saved = saved_run(manifest["run_config"])
        transition = PhaseTransition.between(saved, run, changes or {}, mode=mode)
    else:
        raise CheckpointBoundaryError("checkpoint scope must be resume or weights")
    counters = read_json(checkpoint / "counters.json")
    if (
        set(counters)
        != {
            "version",
            "lifetime_iterations",
            "lifetime_env_steps",
            "lifetime_updates",
            "optimizer_work",
            "phase_start_iterations",
            "curriculum",
        }
        or counters["version"] != 1
    ):
        raise CheckpointBoundaryError("unknown canonical counters")
    for key in (
        "lifetime_iterations",
        "lifetime_env_steps",
        "lifetime_updates",
        "phase_start_iterations",
    ):
        if type(counters[key]) is not int or counters[key] < 0:
            raise CheckpointBoundaryError("invalid canonical counters")
    if counters["phase_start_iterations"] > counters["lifetime_iterations"] or counters[
        "curriculum"
    ] != {"version": 1, "stage": None}:
        raise CheckpointBoundaryError("unsupported phase/curriculum counters")
    state, boundary = read_checkpoint_payload(checkpoint)
    # The payload was produced under the saved contract; a phase may rebind reservations.
    validate_payload_structure(state, boundary, saved)
    validate_current_model_payload(state, run)
    if module_payload_layout(state) != manifest["module_layout"]:
        raise CheckpointBoundaryError("model payload differs from captured layout")
    if boundary["semantic_digest"] != manifest["run_config"]["semantic_digest"]:
        raise CheckpointBoundaryError("boundary semantic contract mismatch")
    if counters["lifetime_iterations"] != state["training_iteration"]:
        raise CheckpointBoundaryError("canonical Algorithm counter mismatch")
    effective = boundary["processes"]["learner/0"]["effective"]
    if (
        counters["lifetime_updates"] != effective["updates"]
        or counters["lifetime_env_steps"]
        != sum(v["env_steps"] for v in effective["learned"].values())
        or counters["optimizer_work"] != effective.get("optimizer_work", {})
    ):
        raise CheckpointBoundaryError("canonical learner counter mismatch")
    return LoadedCheckpoint(checkpoint, manifest, counters, transition)


def read_checkpoint_payload(checkpoint):
    manifest = read_json(Path(checkpoint) / MANIFEST)
    if manifest.get("kind") == "inference_weights_v1":
        from training.inference_bundle import read_inference_payload

        return read_inference_payload(checkpoint, manifest)
    try:
        return tuple(
            pickle.loads((Path(checkpoint) / name).read_bytes())
            for name in ("algorithm.pkl", "boundary.pkl")
        )  # noqa: S301
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as error:
        raise CheckpointBoundaryError("unreadable checkpoint state payload") from error


def _guarded(runner):
    if runner.env is None:
        return True
    return all(
        not wrapped.unwrapped._runtime_activated
        and not wrapped.unwrapped._has_reset_once
        and wrapped.unwrapped.simulator.world is None
        for wrapped in runner.env.envs
    )


def restore_checkpoint(algorithm, loaded: LoadedCheckpoint, run) -> None:
    """Apply only to a new owned guarded instance; callers discard it on failure."""
    checkpoint, manifest, counters = loaded.path, loaded.manifest, loaded.counters
    if algorithm.training_iteration != 0 or any(
        not all(group.foreach_env_runner(_guarded, healthy_only=False))
        for group in (algorithm.env_runner_group, algorithm.eval_env_runner_group)
    ):
        raise CheckpointBoundaryError("restore requires a new unactivated owned Algorithm")
    if loaded.transition is None or loaded.transition.target_digest != run.semantic_digest:
        raise CheckpointBoundaryError("restore requires a bundle inspected for this run")
    transition = loaded.transition if loaded.transition.changed else None
    # Only trusted local bundles are supported: checksums establish integrity,
    # not a security boundary for Python/RLlib pickle deserialization.
    state, boundary = read_checkpoint_payload(checkpoint)
    preflight_checkpoint(algorithm, state, boundary, run, transition=transition)
    algorithm.set_state(state)
    if transition is not None:
        values = transition.overrides
        for result in algorithm.learner_group.foreach_learner(
            lambda learner: learner.apply_phase_overrides(values), healthy_only=False
        ):
            if not result.ok:
                raise result.get()
    restore_boundary_state(algorithm, boundary, run, transition=transition)
    algorithm._checkpoint_phase_start = (
        counters["lifetime_iterations"]
        if transition is not None
        else counters["phase_start_iterations"]
    )
    algorithm._checkpoint_lineage = (
        extend_lineage(
            manifest["lineage"],
            run,
            parent_sha=checkpoint_digest(checkpoint),
            lifetime_iterations=counters["lifetime_iterations"],
            changes=transition.changed,
        )
        if transition is not None
        else manifest["lineage"]
    )
    activate_algorithm(algorithm, run)


def load_checkpoint_algorithm(loaded: LoadedCheckpoint, run, *, scenario_artifact_dir=None):
    """Construct from an inspected bundle; discard a failed new Algorithm entirely."""
    algorithm = build_algorithm(
        run, defer_activation=True, scenario_artifact_dir=scenario_artifact_dir
    )
    try:
        restore_checkpoint(algorithm, loaded, run)
        return algorithm
    except BaseException:
        algorithm.stop()
        raise
