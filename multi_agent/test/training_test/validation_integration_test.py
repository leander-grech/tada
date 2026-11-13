"""Actual complete checkpoint -> development -> development evaluation linkage."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_checkpoint_development_evaluation_identity(native_run, config_path, tmp_path):
    data = json.loads(config_path.read_text())
    data["reserved_manifest"] = str(Path("config/contracts/bgy/reserved-v1.json").resolve())
    data["reserved_conditions"] = str(
        Path("config/contracts/bgy/conditions-v2.json").resolve()
    )
    baseline = json.loads(Path("config/contracts/bgy/conditions-v2.json").read_text())[
        "benchmark"
    ]
    data["scenario"].update(
        aircraft_count=baseline["stratum"]["aircraft_count"],
        episode_steps=baseline["episode_steps"],
        start_offset_max_s=baseline["start_offset_max_s"],
        noop_verification_max_seconds=6000,
    )
    data["optimizer"].update(train_batch_size=128, minibatch_size=32)
    config_path.write_text(json.dumps(data))
    script = r"""
import hashlib, json, sys
from pathlib import Path
import ray
from training.run_config import load_run_config
from training.ppo_config import build_algorithm
from training.checkpoints import capture_checkpoint, inspect_checkpoint
from evaluation.checkpoint_evaluation import evaluate_checkpoint
from evaluation.benchmark.contract import procedure_contract, materialize_manifest, BenchmarkContract
root=Path(sys.argv[2]); run=load_run_config(sys.argv[1])
ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False)
algorithm=None
try:
 algorithm=build_algorithm(run); algorithm.train()
 checkpoint=capture_checkpoint(algorithm,root/"complete",run)
finally:
 if algorithm is not None: algorithm.stop()
 ray.shutdown()
inspect_checkpoint(checkpoint,run)
contract=procedure_contract("development")
reservation=json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
manifest=materialize_manifest(contract,[w["seed"] for w in reservation["validation"][:1]],root/"development.json")
from evaluation.checkpoint_evaluation import validate_development_request
validate_development_request(run,manifest)
validate_development_request(load_run_config(Path("config/contracts/bgy/base.json")),manifest)
from dataclasses import replace
bad=replace(run,scenario={**run.scenario,"noop_verification_max_seconds":run.scenario["noop_verification_max_seconds"]+1})
try: validate_development_request(bad,manifest)
except ValueError: pass
else: raise AssertionError("wrong run world contract accepted")
try: validate_development_request(run,root/"missing.json")
except FileNotFoundError: pass
else: raise AssertionError("missing manifest accepted")
contract=BenchmarkContract(contract.compatibility,{"development_manifest":str(manifest)})
path=evaluate_checkpoint(checkpoint,contract,root/"evaluation")
evaluation=json.loads(path.read_text())
assert evaluation["coverage"]["status"]=="partial"
assert evaluation["coverage"]["completed"]==1
assert not evaluation["promotion_eligible"]
assert evaluation["checkpoint_sha256"]==hashlib.sha256((checkpoint/"manifest.json").read_bytes()).hexdigest()
assert evaluation["config_sha256"]==run.semantic_digest
assert evaluation["manifest_sha256"]==json.loads(manifest.read_text())["manifest_sha256"]
assert evaluate_checkpoint(checkpoint,contract,root/"evaluation")==path
import evaluation.benchmark.evaluate as evaluator
def failed(*args, **kwargs): raise RuntimeError("controlled development transport failure")
evaluator.run_development=failed
error_path=evaluate_checkpoint(checkpoint,contract,root/"error-evaluation")
error=json.loads(error_path.read_text())
assert error["coverage"]=={"expected":1,"campaign_required":100,"observed":0,"completed":0,"errors":0,"missing":1,"status":"error"}
assert error["error"]["type"]=="RuntimeError" and not error["promotion_eligible"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lifecycle_links_independent_cadences_and_resume(native_run, config_path, tmp_path):
    script = r"""
import hashlib,json,sys
from pathlib import Path
import ray
from training import lifecycle
from training.lifecycle import RunRequest
from evaluation.benchmark.contract import procedure_contract,materialize_manifest
from evaluation.checkpoint_evaluation import validate_development_request
root=Path(sys.argv[2]); config=Path(sys.argv[1])
contract=procedure_contract("development")
reservation=json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
manifest=materialize_manifest(contract,[reservation["validation"][0]["seed"]],root/"development.json")
data=json.loads(config.read_text()); baseline=contract.compatibility["conditions"]["benchmark"]
data["reserved_manifest"]=str(Path("config/contracts/bgy/reserved-v1.json").resolve())
data["reserved_conditions"]=str(Path("config/contracts/bgy/conditions-v2.json").resolve())
data["scenario"].update(aircraft_count=baseline["stratum"]["aircraft_count"],episode_steps=baseline["episode_steps"],start_offset_max_s=baseline["start_offset_max_s"],noop_verification_max_seconds=contract.compatibility["verification_cap_s"])
data["optimizer"].update(train_batch_size=128,minibatch_size=32)
data["execution"].update(stop_iterations=1,development_manifest=str(manifest))
data["cadences"].update(checkpoint_iterations=3,evaluation_iterations=1)
config.write_text(json.dumps(data))
lifecycle.snapshot_logging_settings=lambda:{"backend":"none"}
ray.init(address="local",num_cpus=4,num_gpus=0,include_dashboard=False)
try:
 first=lifecycle.run_training(RunRequest("fresh",config,None),validate_development=validate_development_request)
 checkpoint=first/"development-checkpoints"/"checkpoint_000001"
 assert (checkpoint/"manifest.json").is_file()
 digest=hashlib.sha256((checkpoint/"manifest.json").read_bytes()).hexdigest()
 initial=json.loads((first/"evaluations"/digest/"evaluation.json").read_text())
 assert initial["checkpoint_sha256"]==digest and initial["lifetime_iterations"]==1
 assert initial["coverage"]["completed"]==1
 assert initial["coverage"]["status"]=="partial" and not initial["promotion_eligible"]
 second=lifecycle.run_training(RunRequest("resume",config,checkpoint),validate_development=validate_development_request)
 assert second!=first
 resumed=second/"development-checkpoints"/"checkpoint_000002"
 digest=hashlib.sha256((resumed/"manifest.json").read_bytes()).hexdigest()
 final=json.loads((second/"evaluations"/digest/"evaluation.json").read_text())
 assert final["lifetime_iterations"]==2 and final["checkpoint_sha256"]!=initial["checkpoint_sha256"]
finally: ray.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=300,
    )
    (tmp_path / "lifecycle-review.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
