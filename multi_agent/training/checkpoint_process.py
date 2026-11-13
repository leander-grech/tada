"""Pickle-safe process RNG state; no PID-based identities or implicit reseeding."""

import random

import numpy as np
import torch


def capture_process_rng() -> dict:
    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None,
    }


def validate_process_rng(state: dict) -> None:
    if (
        not isinstance(state, dict)
        or set(state) != {"version", "python", "numpy", "torch_cpu", "torch_cuda"}
        or state["version"] != 1
    ):
        raise ValueError("unknown process RNG sidecar")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState(0).set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        cuda = state["torch_cuda"]
        if cuda is not None:
            if not isinstance(cuda, list) or len(cuda) != torch.cuda.device_count():
                raise ValueError("CUDA RNG device count mismatch")
            for index, value in enumerate(cuda):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.uint8
                    or value.ndim != 1
                ):
                    raise ValueError("invalid CUDA RNG state")
                torch.Generator(device=f"cuda:{index}").set_state(value)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("invalid process RNG state") from exc


def restore_process_rng(state: dict) -> None:
    validate_process_rng(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
