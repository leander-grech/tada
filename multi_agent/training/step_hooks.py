"""Per-iteration hooks: heartbeat, nonfinite stop, inline development evaluation."""

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from training import process
from training.checkpoint_manifest import checkpoint_digest

_LEARNER_HEALTH_TERMS = ("loss", "kl", "entropy", "grad")


def nonfinite_paths(value, path=""):
    """Dotted paths of every nonfinite float leaf."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite_paths(v, f"{path}.{k}".strip("."))]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in nonfinite_paths(v, f"{path}.{i}")]
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def json_safe(value):
    """JSON-serializable copy with nonfinite floats as None and opaque objects as text."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def unhealthy_learner_paths(result):
    """Nonfinite learner loss/KL/entropy/gradient values; missing metrics stay diagnostic."""
    return [
        path
        for path in nonfinite_paths(result)
        if path.startswith("learners.")
        and any(term in path.lower().rsplit(".", 1)[-1] for term in _LEARNER_HEALTH_TERMS)
    ]


class StepHooks:
    def __init__(self, run, segment_dir, algorithm, *, env_steps=0):
        self.run, self.segment_dir, self.algorithm = run, Path(segment_dir), algorithm
        # Lifetime env steps at the previous iteration; None when unknown (restored segment).
        self._env_steps = env_steps

    def before_step(self):
        process.heartbeat(
            "training-iteration", lifetime_iterations=self.algorithm.training_iteration
        )

    def after_step(self, result, *, elapsed_seconds, evaluation_seconds):
        memory = psutil.virtual_memory()
        result["active_seconds"] = max(0, elapsed_seconds - evaluation_seconds)
        result["evaluation_seconds"] = evaluation_seconds
        result["wall_seconds"] = elapsed_seconds
        result["resource_health"] = {
            "available_memory_bytes": memory.available,
            "process_rss_bytes": psutil.Process().memory_info().rss,
        }
        result["config_sha256"] = self.run.semantic_digest
        current_steps = result["lifetime_env_steps"]
        result["throughput_env_steps_per_active_second"] = (
            (current_steps - self._env_steps) / result["active_seconds"]
            if self._env_steps is not None and result["active_seconds"] > 0
            else None
        )
        self._env_steps = current_steps
        result["checkpoint_lineage"] = getattr(
            self.algorithm, "_checkpoint_execution_segment", None
        )
        unhealthy = unhealthy_learner_paths(result)
        if unhealthy:
            process.request_stop(f"nonfinite learner diagnostics: {unhealthy[:5]}")
        if not process.stop_requested() and self._evaluation_due(result):
            seconds = self._evaluate(Path(self.run.execution["development_manifest"]))
            result["evaluation_seconds"] += seconds
            result["wall_seconds"] += seconds
        if process.stop_requested():
            result["done"] = True
        process.heartbeat(
            "safe-iteration-boundary", lifetime_iterations=result["lifetime_iterations"]
        )
        return result

    def _evaluation_due(self, result):
        return (
            bool(self.run.execution.get("development_manifest"))
            and result["lifetime_iterations"] % self.run.cadences["evaluation_iterations"] == 0
        )

    def _evaluate(self, manifest):
        from training.checkpoints import capture_checkpoint

        iteration = self.algorithm.training_iteration
        checkpoint = capture_checkpoint(
            self.algorithm,
            self.segment_dir / "development-checkpoints" / f"checkpoint_{iteration:06d}",
            self.run,
        )
        digest = checkpoint_digest(checkpoint)
        output = self.segment_dir / "evaluations" / digest
        process.heartbeat("development-evaluation", checkpoint_sha256=digest)
        started = time.monotonic()
        with (self.segment_dir / f"evaluation-{digest}.log").open("w") as log:
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.checkpoint_evaluation",
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "RAY_ADDRESS": "local"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if child.returncode != 0:
            process.request_stop(f"development evaluation child failed: {child.returncode}")
        return time.monotonic() - started
