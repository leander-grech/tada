"""The actual guarded remote topology captures/restores foundation sidecars."""

import os
import subprocess
import sys


def test_actual_actor_boundary_roundtrip(native_run, config_path):
    script = r"""
import copy
import sys
import ray
import random
import numpy as np
import torch
from training.run_config import load_run_config
from training.ppo_config import build_algorithm
from training.runtime_gate import activate_algorithm
from training.checkpoint_boundary import capture_boundary_state, restore_boundary_state
from common.runtime_log import runtime_print

run = load_run_config(sys.argv[1])
ray.init(num_cpus=4, include_dashboard=False)
def draw(_actor=None):
    return (random.random(), float(np.random.random()), torch.rand(3).tolist())

def draws(algorithm):
    result = {}
    for role, group in (("training", algorithm.env_runner_group), ("evaluation", algorithm.eval_env_runner_group)):
        result[role] = group.foreach_env_runner(draw, local_env_runner=False)
    result["learner"] = [item.get() for item in algorithm.learner_group.foreach_learner(draw)]
    result["driver"] = draw()
    return result

a = b = c = None
try:
    a = build_algorithm(run)
    a.train()
    boundary = copy.deepcopy(capture_boundary_state(a, run))
    assert boundary['processes']['operational_evaluation/1']['runner']['environment']['boundary'] == 'pristine'
    expected_draws = draws(a)
    training = boundary['processes']['training/1']['runner']
    assert training['emitted']['env_steps'] > 0
    assert boundary['processes']['learner/0']['effective']['learned']['training/1'] == training['emitted']
    state = a.get_state()
    a.stop()
    a = None
    b = build_algorithm(run, defer_activation=True)
    b.set_state(state)
    healthy = b.env_runner_group.healthy_env_runner_ids()
    for field in ("missing_process", "digest", "receipt", "rng", "environment", "outer_envelope"):
        bad = copy.deepcopy(boundary)
        if field == "missing_process":
            del bad["processes"]["training/1"]
        elif field == "digest":
            bad["semantic_digest"] = "0" * 64
        elif field == "receipt":
            bad["processes"]["training/1"]["runner"]["emitted"]["sequence"] += 1
        elif field == "rng":
            bad["processes"]["training/1"]["rng"]["unknown"] = True
        elif field == "outer_envelope":
            bad["processes"]["training/1"]["unknown"] = True
        else:
            bad["processes"]["training/1"]["runner"]["environment"]["unknown"] = True
        try:
            restore_boundary_state(b, bad, run)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted malformed " + field)
        assert b.env_runner_group.healthy_env_runner_ids() == healthy, field
    assert b.env_runner_group.foreach_env_runner(lambda r: r.env.envs[0].unwrapped.get_training_stream_state()["stream"]["cursor"], local_env_runner=False) == [0]
    restore_boundary_state(b, boundary, run)
    assert draws(b) == expected_draws
    before = b.env_runner_group.foreach_env_runner(lambda r: (r.env.envs[0].unwrapped._has_reset_once, r.env.envs[0].unwrapped._runtime_activated), local_env_runner=False)
    assert before == [(False, False)]
    activate_algorithm(b, run)
    b.train()
    after = capture_boundary_state(b, run)
    assert after['processes']['training/1']['runner']['emitted']['env_steps'] > training['emitted']['env_steps']
    assert after['processes']['learner/0']['effective']['updates'] > boundary['processes']['learner/0']['effective']['updates']
    assert after['processes']['operational_evaluation/1']['runner']['environment']['boundary'] == 'closed'
    after_state = b.get_state()
    b.stop()
    b = None
    c = build_algorithm(run, defer_activation=True)
    c.set_state(after_state)
    restore_boundary_state(c, after, run)
    activate_algorithm(c, run)
    assert capture_boundary_state(c, run)['processes']['operational_evaluation/1']['runner']['emitted'] == after['processes']['operational_evaluation/1']['runner']['emitted']
    c.env_runner_group.foreach_env_runner(lambda r: (r.sample(), r.get_metrics()), local_env_runner=False)
    try:
        capture_boundary_state(c, run)
    except ValueError as error:
        assert "not completed learning" in str(error)
    else:
        raise AssertionError("accepted unlearned trajectory")
    runtime_print('Checkpoint', 'actual_actor_restore', sampled=after['processes']['training/1']['runner']['emitted'], learned=after['processes']['learner/0']['effective']['learned'], rng_equal=True, restored_after_evaluation=True)
finally:
    if a is not None:
        a.stop()
    if b is not None:
        b.stop()
    if c is not None:
        c.stop()
    ray.shutdown()
"""
    # CPU actor diagnostic only: CUDA acceptance is the separate actual Adam test.
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout
