"""Real native/RLlib weights-only inference, transport and tamper preflight."""

import os
import subprocess
import sys


def test_authenticated_native_reader(native_run, config_path, tmp_path):
    import json

    data = json.loads(config_path.read_text())
    data["scenario"].update(
        aircraft_count=10, episode_steps=128, noop_verification_max_seconds=6000
    )
    data["optimizer"].update(train_batch_size=128, minibatch_size=32)
    config_path.write_text(json.dumps(data))
    script = r"""
import json, pickle, sys, shutil
from pathlib import Path
import numpy as np
import ray
from training.run_config import load_run_config
from training.ppo_config import build_algorithm
from training.checkpoints import capture_checkpoint, inspect_checkpoint
from training.checkpoint_bundle import publish_bundle
from evaluation.checkpoint_adapter import inspect_inference_checkpoint
from evaluation.loading import build_eval_algo
from inference.engine import InferenceEngine
from atc_env.completed_scenario import CompletedScenarioResult
from common.runtime_log import runtime_print
run = load_run_config(sys.argv[1])
root = Path(sys.argv[2])
ray.init(num_cpus=4, num_gpus=0, include_dashboard=False)
a = b = None
try:
    a = build_algorithm(run)
    a.train()
    path = capture_checkpoint(a, root / "complete", run)
    def observations(runner):
        env = runner.env.envs[0].unwrapped
        obs, _ = env.reset()
        return env.simulator.world.json(), obs
    training_world, training_obs = a.env_runner_group.foreach_env_runner(observations, local_env_runner=False)[0]
    a.stop(); a = None
    inspect_inference_checkpoint(path)
    historical = root / "historical"
    historical.mkdir()
    saved_state = pickle.loads((path / "algorithm.pkl").read_bytes())
    (historical / "weights.pkl").write_bytes(pickle.dumps(saved_state["learner_group"]["learner"]["rl_module"]))
    try: build_eval_algo(historical, eval_env_config={}, callbacks=[], timestep_budget=None)
    except ValueError as error: assert "incomplete checkpoint bundle" in str(error)
    else: raise AssertionError("metadata-free weights became current eligible")
    # A resealed, well-formed inner hash must fail every reader BEFORE construction.
    def write(staging):
        for file in path.iterdir():
            if file.name != "manifest.json":
                shutil.copyfile(file, staging / file.name)
        sidecar = pickle.loads((staging / "boundary.pkl").read_bytes())
        sidecar["processes"]["training/1"]["runner"]["environment"]["stream"]["stream"]["reserved_manifest_hash"] = "0" * 64
        (staging / "boundary.pkl").write_bytes(pickle.dumps(sidecar))
        body = json.loads((path / "manifest.json").read_text())
        del body["files"]
        return body
    bad = publish_bundle(root / "bad-hash", write)
    from unittest.mock import patch, Mock
    pristine = Mock()
    def no_construction(*args, **kwargs):
        raise AssertionError("reader constructed before preflight")
    construction_guard = patch("training.checkpoints.build_algorithm", no_construction)
    inference_guard = patch("ray.rllib.algorithms.ppo.PPOConfig.build_algo", no_construction)
    construction_guard.start(); inference_guard.start()
    for reader in (lambda: inspect_checkpoint(bad, run),
                   lambda: build_eval_algo(bad, eval_env_config={}, callbacks=[], timestep_budget=None)):
        try: reader()
        except ValueError as error:
            assert "stream identity" in str(error), str(error)
        else: raise AssertionError("accepted inconsistent reserved set")
    construction_guard.stop(); inference_guard.stop()
    assert not pristine.mock_calls, pristine.mock_calls
    from config.config import Config
    Config.DEVIATION_ENCODING = "linear"
    Config.NEIGHBOR_SELECTION = "invalid"
    Config.MAX_WAYPOINTS = run.features["dimensions"]["max_waypoints"]
    Config.MAX_NEIGHBORS = run.features["dimensions"]["max_neighbors"]
    Config.GRU_HIDDEN_SIZE = run.features["architecture"]["gru_hidden"]
    Config.MLP_HIDDEN = run.features["architecture"]["mlp_hidden"]
    b = build_eval_algo(path, eval_env_config={"evaluation_mode": True}, callbacks=[], timestep_budget=None)
    env = b.eval_env_runner.env.envs[0].unwrapped
    assert env.consumed_semantic_digest == run.semantic_digest
    assert env.feature_settings.deviation_encoding == "signed_log"
    assert env.feature_settings.neighbor_selection == "risk"
    assert env.success_time_dev_s == 60
    module = b.get_module("shared_policy")
    assert module.consumed_semantic_digest == run.semantic_digest
    from flight_simulator import World
    env.set_next_world(World.from_json(training_world))
    eval_obs, _ = env.reset()
    def bytes_tree(value):
        if isinstance(value, dict): return {k: bytes_tree(v) for k, v in value.items()}
        if isinstance(value, np.ndarray): return (str(value.dtype), value.shape, value.tobytes())
        return value
    assert bytes_tree(training_obs) == bytes_tree(eval_obs), "same-world observation bytes differ"
    from evaluation.sampling import run_completed_scenario
    from training.reservations import effective_reserved_seeds
    assert 1123251507 in effective_reserved_seeds(run.reserved_manifest)
    diagnostic_seed = 1717171717
    assert diagnostic_seed not in effective_reserved_seeds(run.reserved_manifest)
    env._next_attempt_id = "owned-disjoint-continuation"
    env._next_attempt_seed = diagnostic_seed
    b.eval_env_runner._needs_initial_reset = True
    original_step = env.step
    terminal_worlds = []
    def capture_world(actions):
        result = original_step(actions)
        if env.attempt_id == "owned-disjoint-continuation" and env._scenario_finished:
            terminal_worlds.append(env.simulator.world.json())
        return result
    env.step = capture_world
    retained = run_completed_scenario(b.eval_env_runner, env.completed_collector,
                                     attempt_id="owned-disjoint-continuation", explore=True, max_sample_steps=130)
    assert retained.diagnostics["complete"], retained.diagnostics
    runtime_print("CompletedNative", "owned disjoint fragment continuation", steps=retained.diagnostics["steps"], result=retained.consumer_fields())
    env.step = original_step
    assert len(terminal_worlds) == 1 and retained.eligible
    (root / "world.json").write_text(terminal_worlds[0])
    result = env.last_completed_result
    assert result is not None
    before = json.dumps(result.to_dict(), sort_keys=True)
    env.reset()
    assert json.dumps(result.to_dict(), sort_keys=True) == before
    assert CompletedScenarioResult.from_dict(json.loads(before)) == result
    runtime_print("CompletedNative", "authenticated reader", provenance=b.inference_provenance,
                  result=result.consumer_fields())
    # Controlled caller-owned late-ETA world requires actual multi-step continuation.
    delayed = json.loads(training_world)
    for target in delayed["arrival_targets"]: target["eta_s"] += 20000
    env.set_next_world(World.from_json(json.dumps(delayed)))
    env._next_attempt_id = "multi-step-owned"
    b.eval_env_runner._needs_initial_reset = True
    reset_calls = []
    original_reset = env.reset
    def reset_once(**kwargs):
        reset_calls.append((env._next_attempt_id, env.attempt_id, env._scenario_finished))
        return original_reset(**kwargs)
    env.reset = reset_once
    continued = run_completed_scenario(b.eval_env_runner, env.completed_collector,
                                      attempt_id="multi-step-owned", explore=True, max_sample_steps=130)
    assert continued.diagnostics["complete"] and continued.diagnostics["steps"] > 1, continued.diagnostics
    assert reset_calls[0][0] == "multi-step-owned"
    assert all(finished for _, previous_id, finished in reset_calls[1:] if previous_id == "multi-step-owned"), "stochastic attempt reset before terminal capture"
    runtime_print("CompletedNative", "multi-step continuation", steps=continued.diagnostics["steps"], resets=len(reset_calls))
    # Remote actor transport is explicit, and carries the exact immutable record.
    b.stop(); b = None
    engine = InferenceEngine(path)
    b = engine._algo
    serving_env = engine._get_eval_env()
    serving_env.set_next_world(World.from_json(training_world))
    serving_obs, _ = serving_env.reset()
    assert bytes_tree(serving_obs) == bytes_tree(training_obs), "serving observation bytes differ"
    serving = engine.run_episode_with_metrics(World.from_json((root / "world.json").read_text()), seed=19, explore=True)
    assert serving["complete"] and serving["eligible"], serving
    assert serving["total_deviation"] is None or isinstance(serving["total_deviation"], (int, float))
    runtime_print("CompletedNative", "actual serving request", candidate=serving)
    b.stop(); b = None
    from evaluation.sampling import sample_completed_round
    seeds = [diagnostic_seed, diagnostic_seed]
    b = build_eval_algo(path, eval_env_config={"evaluation_seeds": seeds, "evaluation_seed_shards": 2},
                        callbacks=[], timestep_budget=None, num_env_runners=2)
    records = sample_completed_round(b, seeds=seeds, rng_seed=17)
    assert len(records) == 2 and all(isinstance(record, CompletedScenarioResult) for record in records)
    assert len({record.attempt_id for record in records}) == 2
    assert [record.diagnostics["seed"] for record in records] == seeds
    runtime_print("CompletedNative", "remote transport", attempts=[r.attempt_id for r in records],
                  declared_inference_topology=b.inference_provenance["inference_topology"])
finally:
    if a is not None: a.stop()
    if b is not None: b.stop()
    ray.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=300,
    )
    from common.runtime_log import runtime_print

    runtime_print("CompletedNative", "child output", stdout=result.stdout, stderr=result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
