"""Inference placement rejects unavailable capacity and false placement claims."""

import pytest

from evaluation.inference_resources import InferenceResources, validate_placement


def test_gpu_request_requires_remote_workers_and_sufficient_capacity():
    for workers, fraction in ((0, 1), (2, 0), (2, -0.1), (2, float("nan"))):
        with pytest.raises(ValueError):
            InferenceResources("cuda", workers, fraction)
    assert InferenceResources("cuda", 3, 0.6).ray_gpus == 3
    resources = InferenceResources("cuda", 3, 0.25)
    assert resources.ray_gpus == 1
    with pytest.raises(ValueError, match="CUDA"):
        resources.validate_capacity(cuda_available=False, visible_gpus=0)
    with pytest.raises(ValueError, match="capacity"):
        InferenceResources("cuda", 3, 0.5).validate_capacity(cuda_available=True, visible_gpus=1)
    resources.validate_capacity(cuda_available=True, visible_gpus=1)


def test_cpu_request_cannot_reserve_gpu_or_accept_cuda_placement():
    with pytest.raises(ValueError):
        InferenceResources("cpu", 1, 0.5)
    resources = InferenceResources("cpu", 0, 0)
    validate_placement(
        resources,
        [
            {
                "device": "cpu",
                "parameter_devices": ["cpu"],
                "buffer_devices": [],
                "physical_uuid": None,
            }
        ],
    )
    with pytest.raises(ValueError):
        validate_placement(
            resources,
            [
                {
                    "device": "cuda:0",
                    "parameter_devices": ["cuda:0"],
                    "buffer_devices": [],
                    "physical_uuid": "GPU-A",
                }
            ],
        )


def test_gpu_placement_rejects_partial_cpu_modules_and_missing_runners():
    resources = InferenceResources("cuda", 2, 0.5)
    report = {
        "device": "cuda:0",
        "parameter_devices": ["cuda:0"],
        "buffer_devices": ["cuda:0"],
        "physical_uuid": "GPU-A",
    }
    validate_placement(resources, [report, report])
    for reports in (
        [report],
        [report, {**report, "parameter_devices": ["cpu"]}],
        [report, {**report, "physical_uuid": None}],
    ):
        with pytest.raises(ValueError):
            validate_placement(resources, reports)


def test_cuda_unavailable_fails_before_checkpoint_or_output_access(tmp_path):
    import os
    import subprocess
    import sys

    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.benchmark.evaluate",
            "--checkpoint",
            "missing-checkpoint",
            "--manifest",
            "missing-manifest",
            "--output",
            str(output),
            "--inference-device",
            "cuda",
            "--workers",
            "1",
            "--gpus-per-runner",
            "1",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "CUDA inference requested but CUDA is unavailable" in result.stderr
    assert "FileNotFoundError" not in result.stderr
    assert not output.exists()
