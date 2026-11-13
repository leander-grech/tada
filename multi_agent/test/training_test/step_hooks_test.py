"""Behavioral tests for per-iteration step hooks."""

import json
from types import SimpleNamespace

import pytest

from training import process, step_hooks
from training.step_hooks import StepHooks


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    monkeypatch.setenv(process.RUN_DIR_ENV, str(tmp_path))
    return tmp_path


def hooks(tmp_path, manifest=None, every=2):
    run = SimpleNamespace(
        semantic_digest="d",
        execution={"development_manifest": manifest},
        cadences={"evaluation_iterations": every},
    )
    return StepHooks(run, tmp_path, SimpleNamespace(training_iteration=1))


def result(iteration=1, steps=100, **extra):
    return {"lifetime_iterations": iteration, "lifetime_env_steps": steps, **extra}


def test_throughput_excludes_evaluation_time(run_root):
    out = hooks(run_root).after_step(result(), elapsed_seconds=12, evaluation_seconds=2)
    assert out["active_seconds"] == 10
    assert out["throughput_env_steps_per_active_second"] == 10
    assert "done" not in out
    assert (
        json.loads((run_root / "heartbeat.json").read_text())["stage"] == "safe-iteration-boundary"
    )


def test_nonfinite_learner_loss_requests_stop(run_root):
    bad = result(learners={"p": {"total_loss": float("nan")}})
    assert hooks(run_root).after_step(bad, elapsed_seconds=1, evaluation_seconds=0)["done"]
    assert process.stop_requested()


def test_nonfinite_episode_metric_does_not_stop(run_root):
    odd = result(env_runners={"episode_return_mean": float("nan")})
    assert "done" not in hooks(run_root).after_step(odd, elapsed_seconds=1, evaluation_seconds=0)


def test_operator_stop_marks_result_done(run_root):
    process.request_stop("operator")
    assert hooks(run_root).after_step(result(), elapsed_seconds=1, evaluation_seconds=0)["done"]


def test_development_evaluation_follows_iteration_cadence(run_root, monkeypatch):
    calls = []
    monkeypatch.setattr(
        StepHooks, "_evaluate", lambda self, manifest: calls.append(manifest) or 3.0
    )
    subject = hooks(run_root, manifest="m.json", every=2)
    outs = [
        subject.after_step(result(i, 100 * i), elapsed_seconds=5, evaluation_seconds=0)
        for i in (1, 2, 3, 4)
    ]
    out = outs[-1]
    assert len(calls) == 2
    assert out["evaluation_seconds"] == 3.0 and out["wall_seconds"] == 8.0


def test_failed_development_child_requests_stop(run_root, monkeypatch):
    monkeypatch.setattr("training.checkpoints.capture_checkpoint", lambda *a: run_root / "ckpt")
    monkeypatch.setattr(step_hooks, "checkpoint_digest", lambda path: "abc")
    monkeypatch.setattr(step_hooks.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7))
    out = hooks(run_root, manifest="m.json", every=1).after_step(
        result(), elapsed_seconds=1, evaluation_seconds=0
    )
    assert out["done"] and "7" in process.read_json(run_root / "stop.json")["reason"]
