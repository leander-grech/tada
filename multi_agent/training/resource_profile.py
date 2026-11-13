"""Immutable requested resources and independently observed physical placement.

The selected external native artifact has no authenticated batch engine. Torch
CUDA inference does not imply CUDA simulator capability.
"""

import subprocess
from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from common.device import DEVICES
from training.run_config import ResolvedRunConfig


@dataclass(frozen=True)
class ResourceProfile:
    learner_device: str
    inference_device: str
    learner_count: int
    env_runners: int
    envs_per_runner: int
    gpus_per_runner: float
    learner_gpu_uuid: str | None
    inference_gpu_uuid: str | None


@dataclass(frozen=True)
class ActorDeviceReport:
    role: str
    actor_id: str
    visible_indices: tuple[int, ...]
    physical_uuids: tuple[str, ...]


def runtime_inventory() -> dict:
    """Read physical inventory without creating a CUDA context in the driver."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {"gpus": [], "rollout_backends": ["scalar"]}
    gpus = []
    for line in result.stdout.splitlines():
        index, uuid, name, total, free = (value.strip() for value in line.split(","))
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "total_mib": int(total),
                "free_mib": int(free),
            }
        )
    return {"gpus": gpus, "rollout_backends": ["scalar"]}


def validate_rollout_capability(backend: str, runtime: dict) -> None:
    """No unverified native engine may be activated, including via supplied flags."""
    if backend != "scalar":
        raise ValueError(
            f"{backend} rollout backend unsupported: selected native artifact has no authenticated batch engine"
        )


def validate_resource_profile(profile: ResourceProfile, runtime: dict) -> None:
    if profile.learner_count != 1 or profile.envs_per_runner != 1:
        raise ValueError("resource profile requires one learner and one environment per runner")
    if type(profile.env_runners) is not int or profile.env_runners < 1:
        raise ValueError("resource profile requires remote environment runners")
    if (
        isinstance(profile.gpus_per_runner, bool)
        or not isfinite(profile.gpus_per_runner)
        or profile.gpus_per_runner < 0
    ):
        raise ValueError("invalid runner GPU reservation")
    inventory = {gpu["uuid"]: gpu for gpu in runtime.get("gpus", [])}
    for device, uuid in (
        (profile.learner_device, profile.learner_gpu_uuid),
        (profile.inference_device, profile.inference_gpu_uuid),
    ):
        if device not in DEVICES:
            raise ValueError("device must be cpu or cuda")
        if device == "cpu" and uuid is not None:
            raise ValueError("CPU profile cannot request a GPU UUID")
        if device == "cuda" and (uuid is None or uuid not in inventory):
            raise ValueError("CUDA profile requires an available physical GPU UUID")
    if profile.inference_device == "cpu" and profile.gpus_per_runner != 0:
        raise ValueError("CPU inference requires zero GPU reservation")
    if profile.inference_device == "cuda":
        if profile.gpus_per_runner <= 0:
            raise ValueError("CUDA inference requires a positive GPU reservation")
        # The operational evaluator is a simultaneous allocated runner as well.
        if (profile.env_runners + 1) * profile.gpus_per_runner > 1:
            raise ValueError("inference GPU is overcommitted including evaluation runner")
        if profile.learner_gpu_uuid == profile.inference_gpu_uuid:
            raise ValueError("dual GPU profile requires distinct physical devices")


def resolve_resource_profile(run: ResolvedRunConfig, runtime: dict) -> ResourceProfile:
    """Resolve automatic UUID choices against this inventory, without mutation."""
    resources = run.resources
    validate_rollout_capability(resources.get("rollout_backend", "scalar"), runtime)
    learner = resources["learner_device"]
    inference = resources.get("inference_device", "cpu")
    if inference == "cuda" and resources["backend"] != "cuda":
        raise ValueError("CUDA inference requires the selected CUDA runtime")
    if learner == "cuda" and resources["gpus_per_learner"] != 1:
        raise ValueError("CUDA learner requires exactly one GPU reservation")
    if learner == "cpu" and resources["gpus_per_learner"] != 0:
        raise ValueError("CPU learner requires zero GPU reservation")
    uuids = [gpu["uuid"] for gpu in runtime.get("gpus", [])]
    learner_uuid = resources.get("learner_gpu_uuid")
    if learner == "cuda" and learner_uuid is None and uuids:
        learner_uuid = uuids[0]
    inference_uuid = resources.get("inference_gpu_uuid")
    if inference == "cuda" and inference_uuid is None:
        inference_uuid = next((uuid for uuid in uuids if uuid != learner_uuid), None)
    profile = ResourceProfile(
        learner,
        inference,
        resources["num_learners"],
        run.topology["env_runners"],
        run.topology["envs_per_runner"],
        resources.get("gpus_per_runner", 0),
        learner_uuid,
        inference_uuid,
    )
    validate_resource_profile(profile, runtime)
    return profile


def validate_actor_devices(
    profile: ResourceProfile, reports: tuple[ActorDeviceReport, ...]
) -> None:
    expected = {
        "learner": profile.learner_count,
        "training": profile.env_runners,
        "operational_evaluation": 1,
    }
    if len({report.actor_id for report in reports}) != len(reports) or any(
        not report.actor_id for report in reports
    ):
        raise ValueError("missing or duplicate actor identity")
    if set(report.role for report in reports) != set(expected) or any(
        sum(report.role == role for report in reports) != count for role, count in expected.items()
    ):
        raise ValueError("missing or unexpected actor device reports")
    for report in reports:
        learner = report.role == "learner"
        device = profile.learner_device if learner else profile.inference_device
        uuid = profile.learner_gpu_uuid if learner else profile.inference_gpu_uuid
        if device == "cpu":
            if report.visible_indices or report.physical_uuids:
                raise ValueError("CPU actor exposes CUDA devices")
        elif report.visible_indices != (0,) or report.physical_uuids != (uuid,):
            raise ValueError(
                f"actual {report.role} physical UUID {report.physical_uuids} differs from requested assignment {uuid}"
            )


def observe_actor_device(role: str, actor_id: str, device) -> ActorDeviceReport:
    """Read the actual Torch device UUID, not a Ray fraction or driver label."""
    import torch

    if device.type == "cpu":
        if torch.cuda.is_initialized():
            raise ValueError("CPU actor unexpectedly initialized CUDA")
        return ActorDeviceReport(role, actor_id, (), ())
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise ValueError("CUDA actor requires one visible physical device")
    uuid = "GPU-" + str(
        UUID(str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-"))
    )
    return ActorDeviceReport(role, actor_id, (device.index or 0,), (uuid,))
