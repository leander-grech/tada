"""Supported Adam restore preserving integer parameter IDs and tensor moments.

Adapted from 1a9b82735b02dc6cc41136d3136300cb5f120b49; unlike its post-load
normalization, conversion here keeps IDs intact before torch maps live parameters.
"""

from copy import deepcopy
from math import isfinite
from numbers import Real

import numpy as np
import torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor


def restore_adam(optimizer, state, device):
    """Load only the supported Adam envelope; never tensorize parameter IDs."""
    if type(optimizer) is not torch.optim.Adam or set(state) != {"state", "param_groups"}:
        raise ValueError("unsupported Adam checkpoint state")
    prepared = deepcopy(state)
    ids = []
    parameters = {}
    if len(prepared["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Adam parameter group count mismatch")
    bools = {"amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"}
    optional_bools = {"foreach", "fused"}
    scalars = {"lr", "eps", "weight_decay"}
    for group, live_group in zip(prepared["param_groups"], optimizer.param_groups, strict=True):
        if len(group["params"]) != len(live_group["params"]):
            raise ValueError("Adam parameter count mismatch")
        if set(group) - (bools | optional_bools | scalars | {"params", "betas", "initial_lr"}):
            raise ValueError("unsupported Adam parameter group")
        if any(type(item) is not int or item < 0 for item in group["params"]):
            raise ValueError("Adam parameter IDs must remain integers")
        ids.extend(group["params"])
        parameters.update(
            (identity, (parameter, group))
            for identity, parameter in zip(group["params"], live_group["params"], strict=True)
        )
        for key in bools | optional_bools:
            if key not in group or group[key] is None and key in optional_bools:
                continue
            value = group[key]
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.item()
            if type(value) is not bool:
                raise ValueError(f"unsupported Adam boolean {key}")
            group[key] = value
        for key in scalars | {"initial_lr"}:
            if key not in group:
                continue
            value = group[key]
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.item()
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"unsupported Adam scalar {key}")
            group[key] = float(value)
        betas = group["betas"]
        if len(betas) != 2:
            raise ValueError("unsupported Adam betas")
        values = [v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v for v in betas]
        if any(isinstance(v, bool) or not isinstance(v, Real) or not 0 <= v < 1 for v in values):
            raise ValueError("unsupported Adam betas")
        group["betas"] = tuple(float(v) for v in values)
        if group.get("fused") or group.get("differentiable"):
            raise ValueError("unsupported fused/differentiable Adam checkpoint")
        if group.get("capturable") and torch.device(device).type != "cuda":
            raise ValueError("capturable Adam checkpoint requires CUDA")
    if len(ids) != len(set(ids)) or set(prepared["state"]) - set(ids):
        raise ValueError("Adam state has duplicate or orphan parameter IDs")
    for identity, slots in prepared["state"].items():
        parameter, group = parameters[identity]
        if not isinstance(slots, dict):
            raise ValueError("unsupported Adam per-parameter state")
        if not slots:  # Adam initializes absent/empty per-parameter state on first gradient.
            continue
        required = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad"):
            required.add("max_exp_avg_sq")
        if set(slots) != required:
            raise ValueError("unsupported Adam moment state for parameter group")
        for name, value in slots.items():
            # Ray serializes zero-dimensional tensors as Python floats. Retain
            # this legitimate scalar path, but never coerce arbitrary lists/ints.
            if type(value) is float:
                tensor = torch.tensor(
                    value, dtype=parameter.dtype if name != "step" else torch.float64
                )
            elif isinstance(value, (torch.Tensor, np.ndarray, np.floating)):
                tensor = torch.as_tensor(value)
            else:
                raise ValueError(f"unsupported Adam {name} representation")
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"Adam {name} must contain finite floating values")
            if name == "step":
                if (
                    tensor.ndim != 0
                    or tensor.dtype not in (torch.float32, torch.float64)
                    or tensor.item() < 0
                ):
                    raise ValueError("Adam step must be a finite nonnegative scalar")
            elif tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(f"Adam {name} shape/dtype differs from live parameter")
            elif name != "exp_avg" and (tensor < 0).any():
                raise ValueError(f"Adam {name} must be nonnegative")
    prepared["state"] = convert_to_torch_tensor(prepared["state"], device=device)
    # Ray may narrow floating state during conversion; validate the values that
    # Torch will actually load as well as their serialized representation.
    if any(
        not torch.isfinite(value).all()
        for slots in prepared["state"].values()
        for value in slots.values()
    ):
        raise ValueError("Adam slot conversion produced nonfinite values")
    optimizer.load_state_dict(prepared)
