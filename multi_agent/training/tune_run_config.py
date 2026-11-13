"""Tune orchestration for one fixed execution segment."""

from pathlib import Path

from ray import tune

from training.run_config import ResolvedRunConfig


def build_trainer_run_config(
    run: ResolvedRunConfig, segment_id: str, *, callbacks=None
) -> tune.RunConfig:
    iterations = run.execution["stop_iterations"]
    return tune.RunConfig(
        storage_path=str(Path(run.execution["output_root"]) / run.run_id / run.phase_id / "tune"),
        name=segment_id,
        stop={"training_iteration": iterations} if iterations is not None else None,
        failure_config=tune.FailureConfig(max_failures=0),
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=run.cadences["checkpoint_iterations"], checkpoint_at_end=True
        ),
        callbacks=list(callbacks or []),
        verbose=0,
    )
