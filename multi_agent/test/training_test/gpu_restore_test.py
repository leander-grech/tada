"""Real CUDA two-update continuation against the uninterrupted Adam reference."""

import os
import subprocess
import sys

import pytest


def test_cuda_adam_restore_preserves_first_update_and_moments():
    import torch.version

    if not torch.version.cuda:
        pytest.skip("selected CUDA runtime required")
    script = r"""
import copy
import torch
from ray.rllib.utils.numpy import convert_to_numpy
from network.optimizer_state import restore_adam
for capturable in (False, True):
    reference = torch.nn.Parameter(torch.tensor([1., 2., 3.], device="cuda"))
    optimizer = torch.optim.Adam([reference], lr=.00073123456789, betas=(.83,.97), foreach=True, capturable=capturable)
    def update(parameter, opt, gradient):
        opt.zero_grad()
        loss = (parameter * gradient).sum()
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        assert torch.isfinite(parameter).all()
    for gradient in (1., 2.):
        update(reference, optimizer, gradient)
    restored = torch.nn.Parameter(reference.detach().clone())
    resumed = torch.optim.Adam([restored], lr=.01, foreach=True, capturable=capturable)
    saved = copy.deepcopy(convert_to_numpy(optimizer.state_dict()))
    restore_adam(resumed, saved, "cuda")
    for key in ("step", "exp_avg", "exp_avg_sq"):
        actual, expected = resumed.state[restored][key], optimizer.state[reference][key]
        if key == "step":
            actual, expected = actual.cpu(), expected.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for gradient in (3., 4.):
        update(reference, optimizer, gradient)
        update(restored, resumed, gradient)
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            actual, expected = resumed.state[restored][key], optimizer.state[reference][key]
            if key == "step":
                actual, expected = actual.cpu(), expected.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
