"""Averaged real bundles pass the weights-only reader and carry the elementwise mean."""

import os
import subprocess
import sys


def test_average_of_two_real_checkpoints(native_run, config_path, tmp_path):
    script = r"""
import json, sys
from pathlib import Path
import numpy as np
import ray
from training.run_config import load_run_config
from training.ppo_config import build_algorithm
from training.checkpoints import capture_checkpoint, inspect_checkpoint, read_checkpoint_payload
from training.checkpoint_average import average_checkpoints
from training.checkpoint_boundary import CheckpointBoundaryError
run = load_run_config(sys.argv[1])
root = Path(sys.argv[2])
ray.init(num_cpus=4, num_gpus=0, include_dashboard=False)
a = build_algorithm(run)
try:
    a.train(); first = capture_checkpoint(a, root / "first", run)
    a.train(); second = capture_checkpoint(a, root / "second", run)
finally:
    a.stop(); ray.shutdown()
averaged = average_checkpoints([first, second], root / "averaged")
inspect_checkpoint(averaged, scope="weights")
modules = [read_checkpoint_payload(p)[0]["learner_group"]["learner"]["rl_module"]["shared_policy"]
           for p in (first, second, averaged)]
left, right, mean = modules
changed = 0
for name, value in mean.items():
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        np.testing.assert_allclose(value, (left[name] + right[name]) / 2, rtol=1e-6, atol=1e-7)
        changed += not np.array_equal(left[name], right[name])
    else:
        assert value == right[name], name
assert changed, "training did not move any weight"
for name in ("boundary.pkl", "counters.json"):
    assert (averaged / name).read_bytes() == (second / name).read_bytes()
manifest = lambda p: {k: v for k, v in json.loads((p / "manifest.json").read_text()).items() if k != "files"}
assert manifest(averaged) == manifest(second)
(averaged / "algorithm.pkl").write_bytes(b"tampered")
try: inspect_checkpoint(averaged, scope="weights")
except CheckpointBoundaryError as error: assert "checksum" in str(error)
else: raise AssertionError("tampered averaged bundle was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
