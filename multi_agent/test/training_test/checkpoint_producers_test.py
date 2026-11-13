"""Actual Tune segments use complete final-storage bundles and fresh continuation."""

import json
import os
import subprocess
import sys


def test_real_tune_periodic_end_and_resume_segments(native_run, config_path, tmp_path):
    data = json.loads(config_path.read_text())
    data["cadences"]["checkpoint_iterations"] = 1
    config_path.write_text(json.dumps(data))
    script = r"""
import json
import sys
from pathlib import Path
import ray
from training import lifecycle
from training.checkpoints import inspect_checkpoint
from common.runtime_log import runtime_print
root = Path(sys.argv[2])
lifecycle.snapshot_logging_settings = lambda: {"backend": "mlflow", "tracking_uri": "sqlite:///" + str(root / "mlflow.db"), "experiment_name": "checkpoint-producers", "tracking_token": None, "save_artifact": False, "log_params_on_trial_end": False}
ray.init(num_cpus=4, include_dashboard=False)
try:
    first = lifecycle.run_training(lifecycle.RunRequest("fresh", Path(sys.argv[1]), None))
    completed = sorted((root / "diagnostic").rglob("algorithm.pkl"))
    assert len(completed) >= 2, completed
    parents = []
    for marker in completed:
        inspect_checkpoint(marker.parent, scope="weights")
        counters = json.loads((marker.parent / "counters.json").read_text())
        parents.append((counters["lifetime_iterations"], marker.parent))
    _, checkpoint = max(parents)
    second = lifecycle.run_training(lifecycle.RunRequest("resume", Path(sys.argv[1]), checkpoint))
    assert first != second
    newest = list((root / "diagnostic").rglob("algorithm.pkl"))
    assert len(newest) > len(completed)
    iterations = [json.loads((marker.parent / "counters.json").read_text())["lifetime_iterations"] for marker in newest]
    assert max(iterations) == 4, iterations
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri="sqlite:///" + str(root / "mlflow.db"))
    experiment = client.get_experiment_by_name("checkpoint-producers")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2 and all(item.info.status == "FINISHED" for item in runs)
    histories = []
    for item in runs:
        history = client.get_metric_history(item.info.run_id, "lifetime_iterations")
        steps = client.get_metric_history(item.info.run_id, "lifetime_env_steps")
        assert [entry.step for entry in history] == [entry.value for entry in steps]
        histories.append([entry.value for entry in history])
    assert sorted(histories) == [[1, 2], [3, 4]], histories
    from training.trainable import SingleRunTrainable
    from training.run_config import load_run_config
    trial = SingleRunTrainable(config={"run": load_run_config(Path(sys.argv[1])), "segment_dir": str(root / "owned-stop"), "segment": {"segment_id": "stop-probe"}})
    try:
        trial.train()
        manual = trial.save_checkpoint(str(root / "manual-parent"))
    finally:
        trial.stop()
    trial = SingleRunTrainable(config={"run": load_run_config(Path(sys.argv[1])), "segment_dir": str(root / "owned-stop-continuation"), "segment": {"segment_id": "fresh-stop-continuation"}})
    try:
        trial.load_checkpoint(manual)
        trial.train()
    finally:
        trial.stop()
    stopped = root / "owned-stop-continuation" / "stop" / "checkpoint_000002"
    inspect_checkpoint(stopped, scope="weights")
    runtime_print("Checkpoint", "real Tune producers", first=str(first), second=str(second), completed=len(newest), iterations=iterations)
finally:
    ray.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(tmp_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    (tmp_path / "tune-producers.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout


def test_cooperative_stop_at_step_boundary_preserves_earlier_bundle(
    native_run, config_path, tmp_path
):
    script = r"""
import sys
from pathlib import Path
import ray
from training import process
from training.checkpoints import inspect_checkpoint
from training.run_config import load_run_config
from training.trainable import SingleRunTrainable
root = Path(sys.argv[2])
ray.init(num_cpus=4, include_dashboard=False)
try:
    trial = SingleRunTrainable(config={"run": load_run_config(Path(sys.argv[1])), "segment_dir": str(root / "segment"), "segment": {"segment_id": "cooperative-stop"}})
    try:
        trial.train()
        parent = Path(trial.save_checkpoint(str(root / "parent")))
        before = {path: path.read_bytes() for path in parent.rglob("*")}
        process.request_stop("operator")
        assert trial.train()["done"]
    finally:
        trial.stop()
    assert {path: path.read_bytes() for path in parent.rglob("*")} == before
    inspect_checkpoint(parent, scope="weights")
    inspect_checkpoint(root / "segment" / "stop" / "checkpoint_000002", scope="weights")
finally:
    ray.shutdown()
"""
    slot = tmp_path / "slot"
    slot.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(tmp_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0", TADA_RUN_DIR=str(slot)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
