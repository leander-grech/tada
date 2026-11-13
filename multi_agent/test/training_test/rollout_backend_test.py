"""Absent native batch capabilities fail without sampling or global mutation."""

from copy import deepcopy
from dataclasses import replace

import pytest

from training.resource_profile import resolve_resource_profile, validate_rollout_capability


def test_missing_native_batch_api_fails_before_sampling(native_run, monkeypatch):
    from atc_env.environment import ATCEnv
    from training.ppo_config import build_algorithm

    generated = []
    monkeypatch.setattr(ATCEnv, "reset", lambda *a, **k: generated.append(True))
    run = replace(
        native_run, resources=dict(native_run.resources, rollout_backend="native_cpu_batch")
    )
    with pytest.raises(ValueError, match="native_cpu_batch.*unsupported"):
        build_algorithm(run)
    assert not generated


def test_unavailable_backend_does_not_mutate_config(native_run):
    runtime = {"gpus": [], "rollout_backends": ["scalar"]}
    before = deepcopy(runtime)
    run = replace(
        native_run, resources=dict(native_run.resources, rollout_backend="native_cuda_batch")
    )
    saved = run.to_dict()
    with pytest.raises(ValueError, match="native_cuda_batch.*unsupported"):
        resolve_resource_profile(run, runtime)
    assert runtime == before
    assert run.to_dict() == saved
    validate_rollout_capability("scalar", runtime)
    with pytest.raises(ValueError, match="unsupported"):
        validate_rollout_capability("cuda", runtime)
