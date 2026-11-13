"""Explicit inference-only resources and observed module placement."""

import math
import os
from dataclasses import dataclass

from common.device import DEVICES


@dataclass(frozen=True)
class InferenceResources:
    device: str
    workers: int
    gpus_per_runner: float

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError("inference device must be cpu or cuda")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("inference workers must be a nonnegative integer")
        if (
            isinstance(self.gpus_per_runner, bool)
            or not math.isfinite(self.gpus_per_runner)
            or not 0 <= self.gpus_per_runner <= 1
        ):
            raise ValueError("invalid inference GPU reservation")
        if self.device == "cpu" and self.gpus_per_runner != 0:
            raise ValueError("CPU inference requires zero GPU reservation")
        if self.device == "cuda" and (self.workers < 1 or self.gpus_per_runner <= 0):
            raise ValueError("CUDA inference requires remote workers and positive GPU reservation")

    @property
    def ray_gpus(self):
        if self.device == "cpu":
            return 0
        # Ray cannot split one actor's fraction across physical devices.
        slots_per_gpu = math.floor(1 / self.gpus_per_runner)
        return math.ceil(self.workers / slots_per_gpu)

    def validate_capacity(self, *, cuda_available, visible_gpus):
        if self.device == "cuda":
            if not cuda_available:
                raise ValueError("CUDA inference requested but CUDA is unavailable")
            if self.ray_gpus > visible_gpus:
                raise ValueError("insufficient visible GPU capacity for inference workers")

    def preflight(self):
        if self.device == "cuda":
            import torch

            self.validate_capacity(
                cuda_available=torch.cuda.is_available(), visible_gpus=torch.cuda.device_count()
            )


def observe_placement(runner):
    """Observe tensors, connector device and physical UUID in the owning process."""
    from training.resource_profile import observe_actor_device

    module = runner.module["shared_policy"]
    observed = observe_actor_device("inference", str(os.getpid()), runner._device)
    return {
        "pid": os.getpid(),
        "worker_index": runner.worker_index,
        "device": str(runner._device),
        "parameter_devices": sorted({str(p.device) for p in module.parameters()}),
        "buffer_devices": sorted({str(b.device) for b in module.buffers()}),
        "physical_uuid": observed.physical_uuids[0] if observed.physical_uuids else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def validate_placement(resources, reports):
    if len(reports) != max(1, resources.workers):
        raise ValueError("missing inference runner placement reports")
    for report in reports:
        device = report["device"]
        if (
            device.split(":")[0] != resources.device
            or not report["parameter_devices"]
            or any(
                value != device for value in report["parameter_devices"] + report["buffer_devices"]
            )
        ):
            raise ValueError("actual inference tensor placement differs from requested device")
        if bool(report["physical_uuid"]) != (resources.device == "cuda"):
            raise ValueError("inference physical GPU identity is missing or unexpected")
