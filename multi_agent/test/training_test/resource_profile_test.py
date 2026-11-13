"""Requested resources are distinct from observed actor placement."""

from dataclasses import replace

import pytest

from training.resource_profile import (
    ActorDeviceReport,
    ResourceProfile,
    resolve_resource_profile,
    validate_actor_devices,
    validate_resource_profile,
)


def inventory():
    return {"gpus": [{"uuid": "GPU-A", "free_mib": 12000}, {"uuid": "GPU-B", "free_mib": 12000}]}


def test_dual_gpu_profile_requires_distinct_verified_devices():
    profile = ResourceProfile("cuda", "cuda", 1, 1, 1, 0.5, "GPU-A", "GPU-B")
    validate_resource_profile(profile, inventory())
    reports = (
        ActorDeviceReport("learner", "L", (0,), ("GPU-A",)),
        ActorDeviceReport("training", "T", (0,), ("GPU-B",)),
        ActorDeviceReport("operational_evaluation", "E", (0,), ("GPU-B",)),
    )
    validate_actor_devices(profile, reports)
    for bad in (
        replace(profile, inference_gpu_uuid="GPU-A"),
        replace(profile, inference_gpu_uuid="GPU-X"),
        replace(profile, gpus_per_runner=0),
        replace(profile, gpus_per_runner=0.75),
    ):
        with pytest.raises(ValueError):
            validate_resource_profile(bad, inventory())
    for bad in (
        reports[:-1],
        reports + (reports[0],),
        (replace(reports[0], physical_uuids=("GPU-B",)), *reports[1:]),
    ):
        with pytest.raises(ValueError):
            validate_actor_devices(profile, bad)


def test_gpu_ports_keep_fixed_run_and_reserved_streams(native_run):
    original = native_run.to_dict()
    resources = dict(
        native_run.resources,
        learner_device="cuda",
        gpus_per_learner=1,
        learner_gpu_uuid="GPU-A",
        inference_device="cpu",
        inference_gpu_uuid=None,
        gpus_per_runner=0,
        rollout_backend="scalar",
    )
    run = replace(native_run, resources=resources)
    profile = resolve_resource_profile(run, inventory())
    from training.ppo_config import build_ppo_config

    config = build_ppo_config(run)
    assert config.num_env_runners is not None and config.num_env_runners > 0
    assert not config.create_env_on_local_worker
    assert profile.learner_count == 1
    assert run.reserved_manifest == native_run.reserved_manifest
    assert run.env_payload() == native_run.env_payload()
    assert native_run.to_dict() == original
    with pytest.raises(ValueError, match="one learner"):
        resolve_resource_profile(
            replace(run, resources=dict(resources, num_learners=2)), inventory()
        )


def test_cpu_reports_reject_cuda_visibility():
    profile = ResourceProfile("cpu", "cpu", 1, 1, 1, 0, None, None)
    reports = tuple(
        ActorDeviceReport(role, role, (), ())
        for role in ("learner", "training", "operational_evaluation")
    )
    validate_actor_devices(profile, reports)
    with pytest.raises(ValueError, match="CPU"):
        validate_actor_devices(
            profile,
            (replace(reports[0], visible_indices=(0,), physical_uuids=("GPU-A",)), *reports[1:]),
        )
