"""Actual checkpoint-linked development input and outcome acceptance."""

import json
import os
import subprocess
import sys


def test_benchmark_native_checkpoint_development(native_run, config_path, tmp_path):
    from pathlib import Path

    conditions = json.loads(Path("config/contracts/bgy/conditions-v2.json").read_text())
    baseline = conditions["benchmark"]
    data = json.loads(config_path.read_text())
    data["reserved_manifest"] = str(Path("config/contracts/bgy/reserved-v1.json").resolve())
    data["reserved_conditions"] = str(
        Path("config/contracts/bgy/conditions-v2.json").resolve()
    )
    data["scenario"].update(
        aircraft_count=baseline["stratum"]["aircraft_count"],
        episode_steps=baseline["episode_steps"],
        start_offset_max_s=baseline["start_offset_max_s"],
        noop_verification_max_seconds=6000,
    )
    data["optimizer"].update(train_batch_size=128, minibatch_size=32)
    config_path.write_text(json.dumps(data))
    script = r"""
import json, sys
from pathlib import Path
import ray
from training.run_config import load_run_config
from training.ppo_config import build_algorithm
from training.checkpoints import capture_checkpoint
from evaluation.benchmark.contract import procedure_contract, materialize_manifest
from evaluation.benchmark.evaluate import run_development
from common.runtime_log import runtime_print
run = load_run_config(sys.argv[1])
root = Path(sys.argv[2])
ray.init(num_cpus=4, num_gpus=0, include_dashboard=False)
algorithm = None
try:
    algorithm = build_algorithm(run)
    algorithm.train()
    checkpoint = capture_checkpoint(algorithm, root / "complete", run)
finally:
    if algorithm is not None: algorithm.stop()
    ray.shutdown()
reservation = json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
seeds = [row["seed"] for row in reservation["validation"][:2]]
manifest = materialize_manifest(procedure_contract("development"), seeds, root / "development.json")
path = run_development(checkpoint, manifest, root / "evidence")
result = json.loads(path.read_text())
assert len(result["attempts"]) == 2
attempt = result["attempts"][0]
assert attempt["completed"]["diagnostics"]["complete"], attempt
assert attempt["completed"]["diagnostics"]["error"] is None, attempt
assert attempt["input_binding"]["world_id"] == attempt["world_id"]
assert attempt["completed"]["diagnostics"]["input_binding"]["world_id"] == attempt["world_id"]
from evaluation.benchmark.evidence import AttemptEvidence, validate_attempt_binding
from dataclasses import replace
canonical = AttemptEvidence.from_dict(attempt)
changed_target = replace(canonical.completed.ledger.targets[0], eta_s=canonical.completed.ledger.targets[0].eta_s + 1)
changed_ledger = replace(canonical.completed.ledger, targets=(changed_target,) + canonical.completed.ledger.targets[1:])
changed = replace(canonical, completed=replace(canonical.completed, ledger=changed_ledger))
try:
    validate_attempt_binding(changed, result["manifest"]["worlds"][0], result["manifest"])
except ValueError as exc:
    assert "authoritative targets" in str(exc)
else:
    raise AssertionError("altered authoritative target accepted")
parallel_path = run_development(checkpoint, manifest, root / "parallel-evidence", workers=2)
parallel = json.loads(parallel_path.read_text())
def physical(rows):
    return {a["world_id"]: {key: a["completed"][key] for key in ("actions", "ledger", "verification")} for a in rows}
assert physical(result["attempts"]) == physical(parallel["attempts"]), "worker schedule changed deterministic policy results"
from scripts.compare_runs import compare_results
comparison = compare_results(path, parallel_path, bootstrap_seed=17, solve_threshold_s=60)
assert comparison["matched"], comparison
runtime_print("BenchmarkNative", "engineering checkpoint and development proof", checkpoint=str(checkpoint), result=str(path), learned_iterations=1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=420,
    )
    from common.runtime_log import runtime_print

    runtime_print("BenchmarkNative", "child output", stdout=result.stdout, stderr=result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
