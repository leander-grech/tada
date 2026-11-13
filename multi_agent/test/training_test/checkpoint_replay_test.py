"""Cold driver/actor process replay through the public complete bundle API."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_cold_process_bundle_replay(native_run, config_path, tmp_path, device):
    import torch.version

    if device == "cuda" and not torch.version.cuda:
        pytest.skip("CUDA profile requires the selected CUDA runtime")
    from common.hashing import content_digest
    from evaluation.seeds import build_reservation_manifest
    from training.seed_stream import ScenarioSeedStream

    data = json.loads(config_path.read_text())
    data["resources"].update(
        learner_device=device, gpus_per_learner=1.0 if device == "cuda" else 0.0
    )
    reserved_path = config_path.parent / data["reserved_manifest"]
    old = json.loads(reserved_path.read_text())
    collision = ScenarioSeedStream(data["root_seed"], (0, 1, 0), frozenset()).next_seed()
    reservation = build_reservation_manifest(
        [5, 9, collision],
        root=617,
        heldout_count=3,
        replacement_count=2,
        stratum=old["strata"][0],
        purpose="fixture",
    )
    reserved_path.write_text(json.dumps(reservation))
    conditions_path = config_path.parent / data["reserved_conditions"]
    conditions = json.loads(conditions_path.read_text())
    conditions["reservation_sha256"] = reservation["manifest_sha256"]
    conditions.pop("conditions_sha256")
    conditions["conditions_sha256"] = content_digest(conditions)
    conditions_path.write_text(json.dumps(conditions))
    data["topology"]["env_runners"] = 2
    data["scenario"]["episode_steps"] = 64
    data["scenario"]["aircraft_count"] = 3
    data["optimizer"].update(train_batch_size=128, minibatch_size=32)
    config_path.write_text(json.dumps(data))
    script = r"""
import sys
from pathlib import Path
import ray
from training.run_config import load_run_config
from training.ppo_config import build_algorithm
from training.checkpoints import capture_checkpoint, inspect_checkpoint, load_checkpoint_algorithm
from common.runtime_log import runtime_print
run = load_run_config(Path(sys.argv[1]))
root = Path(sys.argv[2])
mode = sys.argv[3]
ray.init(num_cpus=5, num_gpus=1 if run.resources["learner_device"] == "cuda" else 0, include_dashboard=False)
a = None
def observe_participation(runner):
    if runner.env is None:
        return
    env = runner.env.envs[0].unwrapped
    original = env.step
    env._cold_max_live = 0
    env._cold_acted_by_world = {}
    def step(actions):
        env._cold_max_live = max(env._cold_max_live, len(env.agents))
        world = env.simulator.initial_world.json()
        env._cold_acted_by_world.setdefault(world, set()).update(actions)
        return original(actions)
    env.step = step

def participation(runner):
    env = runner.env.envs[0].unwrapped
    return {"live": env._cold_max_live, "acted_same_world": max(map(len, env._cold_acted_by_world.values()), default=0)}

try:
    if mode == "baseline":
        a = build_algorithm(run)
        a.train()
        capture_checkpoint(a, root / "before-eval", run)
        a.train()
        capture_checkpoint(a, root / "parent", run)
    elif mode == "abrupt":
        import os
        a = load_checkpoint_algorithm(inspect_checkpoint(root / "parent", run), run)
        capture_checkpoint(a, root / "recovery-parent", run)
        import json
        import psutil
        from training.runtime_gate import verify_algorithm_selection
        proof = verify_algorithm_selection(a, run)
        owned = {process.pid: process.create_time() for process in psutil.Process().children(recursive=True)}
        actors = [item["pid"] for item in proof["actors"] if item["role"] == "learner" or item["worker_index"] > 0]
        assert set(actors) <= set(owned), (actors, owned)
        (root / "abrupt-processes.json").write_text(json.dumps({"owned": owned, "actors": actors}))
        runtime_print("Checkpoint", "abrupt exit after public save", driver=os.getpid(), actors=actors)
        sys.stdout.flush()
        os._exit(17)
    elif mode == "phase":
        from dataclasses import replace
        from training.phase import apply_sets
        run, changes = apply_sets(replace(run, phase_id="lower-lr"), [
            f"lr={run.optimizer['learning_rate'] / 3}",
            f"entropy={run.optimizer['entropy_coeff'] / 2}",
            f"aircraft_count={run.scenario['aircraft_count'] + 1}",
        ])
        a = load_checkpoint_algorithm(
            inspect_checkpoint(root / "parent", run, mode="phase", changes=changes), run
        )
        capture_checkpoint(a, root / "phase-parent", run)
        a.train()
        capture_checkpoint(a, root / "phase-first", run)
    else:
        a = load_checkpoint_algorithm(inspect_checkpoint(root / "recovery-parent", run), run)
        capture_checkpoint(a, root / "restored-parent", run)
    a.env_runner_group.foreach_env_runner(observe_participation, local_env_runner=False)
    a.train()
    simultaneous = a.env_runner_group.foreach_env_runner(participation, local_env_runner=False)
    assert any(item["live"] >= 2 and item["acted_same_world"] >= 2 for item in simultaneous), simultaneous
    runtime_print("Checkpoint", "actual overlapping aircraft", mode=mode, simultaneous=simultaneous)
    capture_checkpoint(a, root / mode, run)
    import pickle
    boundary = pickle.loads((root / mode / "boundary.pkl").read_bytes())
    runtime_print("Checkpoint", "cold_process", mode=mode, iteration=a.training_iteration, runners={key: value["runner"] for key, value in boundary["processes"].items() if key.startswith("training/")})
finally:
    if a is not None:
        a.stop()
    ray.shutdown()
"""
    for mode in ("baseline", "abrupt", "resumed", "phase"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(config_path), str(tmp_path), mode],
            env=dict(
                os.environ, CUDA_VISIBLE_DEVICES="0" if device == "cuda" else "", PYTHONHASHSEED="0"
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        (tmp_path / (mode + ".log")).write_text(result.stdout)
        assert result.returncode == (17 if mode == "abrupt" else 0), result.stdout
        if mode == "abrupt":
            import signal
            import time

            import psutil

            evidence = json.loads((tmp_path / "abrupt-processes.json").read_text())
            owned = {int(pid): started for pid, started in evidence["owned"].items()}

            def alive():
                remaining = []
                for pid, started in owned.items():
                    try:
                        process = psutil.Process(pid)
                        if (
                            process.create_time() == started
                            and process.status() != psutil.STATUS_ZOMBIE
                        ):
                            remaining.append(pid)
                    except psutil.NoSuchProcess:
                        pass
                return remaining

            deadline = time.monotonic() + 10
            while alive() and time.monotonic() < deadline:
                time.sleep(0.1)
            fenced = []
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for pid in alive():
                    try:
                        os.kill(pid, sig)
                        fenced.append(pid)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + 5
                while alive() and time.monotonic() < deadline:
                    time.sleep(0.1)
            assert not alive(), "owned abrupt driver descendants remain"
            (tmp_path / "abrupt-exit-proof.json").write_text(
                json.dumps(
                    {
                        "actors": evidence["actors"],
                        "owned": owned,
                        "fenced": fenced,
                        "remaining": alive(),
                    }
                )
            )

    import pickle

    import numpy as np

    def load(name, file):
        return pickle.loads((tmp_path / name / file).read_bytes())

    parent, replay_parent = (load(name, "algorithm.pkl") for name in ("parent", "restored-parent"))
    for module_id, state in parent["learner_group"]["learner"]["rl_module"].items():
        for name, value in state.items():
            np.testing.assert_array_equal(
                value,
                replay_parent["learner_group"]["learner"]["rl_module"][module_id][name],
                err_msg="restore/" + name,
            )
    baseline, restored = (load(name, "algorithm.pkl") for name in ("baseline", "resumed"))
    expected = baseline["learner_group"]["learner"]
    actual = restored["learner_group"]["learner"]
    assert expected["atc_effective_state"] == actual["atc_effective_state"]
    for module_id, state in expected["rl_module"].items():
        for name, value in state.items():
            if device == "cpu":
                np.testing.assert_array_equal(
                    value, actual["rl_module"][module_id][name], err_msg=name
                )
            else:
                np.testing.assert_allclose(
                    value, actual["rl_module"][module_id][name], rtol=1e-6, atol=1e-7, err_msg=name
                )
    assert json.loads((tmp_path / "baseline/counters.json").read_text()) == json.loads(
        (tmp_path / "resumed/counters.json").read_text()
    )
    left, right = (load(name, "boundary.pkl") for name in ("baseline", "resumed"))
    for identity in ("training/1", "training/2", "operational_evaluation/1"):
        assert (
            left["processes"][identity]["runner"]["environment"]
            == right["processes"][identity]["runner"]["environment"]
        )

    assert (
        load("parent", "boundary.pkl")["processes"]["training/1"]["runner"]["environment"][
            "stream"
        ]["stream"]["rejected"]
        >= 1
    )
    boundary = load("parent", "boundary.pkl")
    lengths = [
        boundary["processes"][identity]["runner"]["environment"]["audit"]["policy_steps"]
        for identity in ("training/1", "training/2")
    ]
    assert len(set(lengths)) == 2, lengths
    from common.runtime_log import runtime_print

    runtime_print(
        "Checkpoint",
        "native cold unequal proof",
        device=device,
        lengths=lengths,
        logs=str(tmp_path),
    )
    previous = parent["learner_group"]["learner"]
    phase_parent = load("phase-parent", "algorithm.pkl")["learner_group"]["learner"]
    for module_id, state in previous["rl_module"].items():
        for name, value in state.items():
            np.testing.assert_array_equal(value, phase_parent["rl_module"][module_id][name])
    for name in ("phase-parent", "phase-first", "phase"):
        learner = load(name, "algorithm.pkl")["learner_group"]["learner"]
        for values in learner["atc_effective_state"]["modules"].values():
            assert values == {
                "learning_rate": data["optimizer"]["learning_rate"] / 3,
                "entropy_coeff": data["optimizer"]["entropy_coeff"] / 2,
            }
        for optimizer_name, envelope in learner["optimizer"].items():
            for slot, values in envelope["state"]["state"].items():
                prior = previous["optimizer"][optimizer_name]["state"]["state"][slot]
                if name == "phase-parent":
                    for field in values:
                        np.testing.assert_array_equal(values[field], prior[field])
                else:
                    assert values["step"] > prior["step"]
        previous = learner
