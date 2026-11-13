"""Validated fresh fixed runs with explicit canonical and execution identities."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import ray
from ray import tune

from atc_env.sector_catalog import ensure_sector_catalog_up_to_date
from common.hashing import canonical_json
from training import process
from training.mlflow_tune import (
    build_tune_mlflow_callbacks,
    mark_running_mlflow_runs_killed,
    public_logging_settings,
    snapshot_logging_settings,
)
from training.run_config import load_run_config
from training.runtime_gate import verify_native_conditions
from training.trainable import SingleRunTrainable
from training.tune_run_config import build_trainer_run_config

MODES = {  # mode: (checkpoint_required, sets_required)
    "fresh": (False, False),
    "resume": (True, False),
    "phase": (True, True),
}


@dataclass(frozen=True)
class RunRequest:
    mode: str
    config_path: Path
    checkpoint: Path | None
    sets: tuple[str, ...] = ()

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unsupported mode {self.mode!r}; choose from {sorted(MODES)}")
        needs_checkpoint, needs_sets = MODES[self.mode]
        if (self.checkpoint is not None) != needs_checkpoint:
            raise ValueError(
                f"{self.mode} mode {'requires' if needs_checkpoint else 'forbids'} a checkpoint"
            )
        if bool(self.sets) != needs_sets:
            raise ValueError(f"{self.mode} mode {'requires' if needs_sets else 'forbids'} --set")


def run_training(
    request: RunRequest, *, validate_development: Callable[[Any, str], None] | None = None
) -> Path:
    """Run one fresh trial; return its evidence directory, never a checkpoint bundle.

    ``validate_development`` is injected by the composition root (``main.py``)
    because training may not import the evaluation layer.
    """
    from training.phase import apply_sets

    run, changes = apply_sets(load_run_config(request.config_path), request.sets)
    # Absolute: Tune runs trainables from per-trial working directories.
    checkpoint = request.checkpoint.resolve() if request.checkpoint is not None else None
    parent = None
    if checkpoint is not None:
        from training.checkpoints import inspect_checkpoint

        parent = inspect_checkpoint(checkpoint, run, mode=request.mode, changes=changes).manifest
    from training.resource_profile import resolve_resource_profile, runtime_inventory

    resolve_resource_profile(run, runtime_inventory())
    if run.execution.get("development_manifest"):
        if validate_development is None:
            raise ValueError("a development manifest requires an injected development validator")
        validate_development(run, run.execution["development_manifest"])
    verify_native_conditions(run)
    if request.mode == "fresh":
        ensure_sector_catalog_up_to_date()
    segment_id = "segment-" + uuid4().hex
    segment_dir = Path(run.execution["output_root"]) / run.run_id / run.phase_id / segment_id
    segment_dir.mkdir(parents=True, exist_ok=False)
    logging = snapshot_logging_settings()
    segment = {
        "run_id": run.run_id,
        "phase_id": run.phase_id,
        "segment_id": segment_id,
        "parent_run_id": parent["run_config"]["run_id"] if parent else None,
        "parent_phase_id": parent["run_config"]["phase_id"] if parent else None,
        "parent_segment_id": (parent["execution_segment"] or {}).get("segment_id")
        if parent
        else None,
        "mode": request.mode,
        "resource_bundles": SingleRunTrainable.default_resource_request({"run": run}).bundles,
        "controls": run.execution,
        "scenario_artifact_dir": str(segment_dir / "scenarios"),
        "logging": public_logging_settings(logging),
        "checkpoint_contract": "complete-bundle-v1",
        "parent_checkpoint": str(checkpoint) if checkpoint else None,
    }
    (segment_dir / "execution.json").write_text(canonical_json(segment) + "\n")
    process.heartbeat("training-startup", segment_id=segment_id)
    callbacks = build_tune_mlflow_callbacks(logging, segment=segment)
    owned_ray = not ray.is_initialized()
    try:
        tuner = tune.Tuner(
            SingleRunTrainable,
            param_space={
                "run": run,
                "segment_dir": str(segment_dir),
                "segment": segment,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "mode": request.mode,
                "changes": changes,
            },
            tune_config=tune.TuneConfig(num_samples=1, reuse_actors=False),
            run_config=build_trainer_run_config(run, segment_id, callbacks=callbacks),
        )
        results = tuner.fit()
        if len(results) != 1:
            raise RuntimeError("fixed execution must produce exactly one trial")
        if results[0].error is not None:
            raise RuntimeError("fixed execution failed") from results[0].error
        return segment_dir
    except BaseException:
        mark_running_mlflow_runs_killed(callbacks)
        raise
    finally:
        if owned_ray and ray.is_initialized():
            ray.shutdown()
