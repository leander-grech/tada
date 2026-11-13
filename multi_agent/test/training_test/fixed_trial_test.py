"""CPU-only actual fixed Tune trial, native scenarios, and two PPO parameter updates."""

import json
import os
import subprocess
import sys
from pathlib import Path

from common.runtime_log import runtime_print


def test_fixed_trial_two_actual_updates(native_run, config_path):
    child = subprocess.run(
        [sys.executable, __file__, str(config_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=240,
    )
    runtime_print("FixedTrial", "CPU-only diagnostic", stdout=child.stdout, stderr=child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr


def _observe(learner, *_args):
    import hashlib

    import ray

    module = learner.module["shared_policy"]
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return {
        "parameters": digest.hexdigest(),
        "lr": [
            group["lr"]
            for optimizer in learner._optimizer_parameters
            for group in optimizer.param_groups
        ],
        "entropy": float(
            learner.entropy_coeff_schedulers_per_module["shared_policy"].get_current_value()
        ),
        "resources": ray.get_runtime_context().get_assigned_resources(),
    }


def _run(path):
    import math
    from unittest.mock import patch

    import ray
    from mlflow.tracking import MlflowClient
    from ray.tune.utils import flatten_dict

    from training import lifecycle
    from training.lifecycle import RunRequest
    from training.run_config import load_run_config
    from training.trainable import SingleRunTrainable

    run = load_run_config(path)
    uri = "sqlite:///" + str(path.parent / "fixture-mlflow.db")
    logging_settings = {
        "backend": "mlflow",
        "tracking_uri": uri,
        "experiment_name": "fixed-trial-fixture",
        "save_artifact": False,
        "log_params_on_trial_end": False,
        "tracking_token": None,
    }

    class ObservedTrainable(SingleRunTrainable):
        def setup(self, config):
            super().setup(config)
            self.observed = []
            self.capture()

        def capture(self):
            assert self.algorithm is not None
            assert self.algorithm.learner_group is not None
            values = [
                result.get() for result in self.algorithm.learner_group.foreach_learner(_observe)
            ]
            self.observed.append(values)
            (self.segment_dir / "learner-evidence.json").write_text(json.dumps(self.observed))

        def step(self):
            result = super().step()
            self.capture()
            losses = {
                key: value
                for key, value in flatten_dict(result).items()
                if key.startswith("learners/") and "loss" in key and isinstance(value, (int, float))
            }
            assert losses and all(math.isfinite(value) for value in losses.values()), losses
            (self.segment_dir / f"update-{len(self.observed) - 1}.json").write_text(
                json.dumps(
                    {
                        "losses": losses,
                        "lifetime_env_steps": result["lifetime_env_steps"],
                    }
                )
            )
            return result

    # This instrumented subclass is not an eligible production checkpoint
    # producer. Dedicated tests exercise the exact owned Trainable public paths.
    original_run_config = lifecycle.build_trainer_run_config

    def observed_run_config(*args, **kwargs):
        config = original_run_config(*args, **kwargs)
        assert config.checkpoint_config is not None
        config.checkpoint_config.checkpoint_frequency = 0
        config.checkpoint_config.checkpoint_at_end = False
        return config

    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
    )
    try:
        with (
            patch.object(lifecycle, "SingleRunTrainable", ObservedTrainable),
            patch.object(lifecycle, "build_trainer_run_config", observed_run_config),
            patch.object(lifecycle, "snapshot_logging_settings", return_value=logging_settings),
        ):
            directory = lifecycle.run_training(RunRequest("fresh", path, None))
        observed = json.loads((directory / "learner-evidence.json").read_text())
        assert len(observed) == 3
        assert len({item[0]["parameters"] for item in observed}) == 3
        assert all(item[0]["lr"] == [run.optimizer["learning_rate"]] for item in observed)
        assert all(item[0]["entropy"] == run.optimizer["entropy_coeff"] for item in observed)
        assert all(
            item[0]["resources"] == {"CPU": run.resources["cpus_per_learner"]} for item in observed
        )
        updates = [json.loads((directory / f"update-{i}.json").read_text()) for i in (1, 2)]
        assert 0 < updates[0]["lifetime_env_steps"] < updates[1]["lifetime_env_steps"]
        manifest = json.loads((directory / "manifest.json").read_text())
        from atc_env.sector_catalog import capture_sector_catalog_identity

        assert manifest["effective_inputs"]["sector_catalog"] == capture_sector_catalog_identity()
        proof = manifest["runtime"]["owned_learning"]
        assert proof["selection_verified"]
        for actor in proof["actors"]:
            if actor["role"] == "learner":
                assert actor["assigned_resources"] == {"CPU": run.resources["cpus_per_learner"]}
            elif actor["worker_index"] > 0:
                assert actor["assigned_resources"] == {"CPU": run.resources["cpus_per_runner"]}
                env = actor["environments"][0]
                assert env["scenario"]["aircraft_count"] == run.scenario["aircraft_count"]
                assert env["features"] == run.features and env["reward"] == run.reward
                expected_threshold = (
                    60 if env["evaluation_mode"] else run.thresholds["training_success_s"]
                )
                assert env["success_time_dev_s"] == expected_threshold
        segment = manifest["execution_segment"]
        assert len({segment["run_id"], segment["phase_id"], segment["segment_id"]}) == 3
        assert segment["scenario_artifact_dir"] == str(directory / "scenarios")
        assert segment["resource_bundles"] == [{"CPU": 1.0}] * 4
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name("fixed-trial-fixture")
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1 and runs[0].info.status == "FINISHED"
        for key in ("run_id", "phase_id", "segment_id"):
            assert runs[0].data.tags[key] == segment[key]
        history = client.get_metric_history(runs[0].info.run_id, "lifetime_env_steps")
        assert [(m.step, m.value) for m in history] == [
            (u["lifetime_env_steps"], u["lifetime_env_steps"]) for u in updates
        ]
        reserved = json.loads(Path("config/contracts/bgy/reserved-v1.json").read_text())
        excluded = {
            item["seed"]
            for role in ("validation", "heldout", "replacements")
            for item in reserved[role]
        }
        assert len(excluded) == 500
        journals = list((directory / "scenarios").rglob("*.jsonl"))
        events = [
            json.loads(line) for journal in journals for line in journal.read_text().splitlines()
        ]
        assert events and all(event["seed"] not in excluded for event in events)
        assert {event["stream"]["domain"] for event in events} == {
            "training",
            "operational_evaluation",
        }
        runtime_print(
            "FixedTrial",
            "evidence",
            directory=directory,
            updates=updates,
            learner=observed,
            segment=segment,
            journals=[str(j) for j in journals],
            events=events,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    _run(Path(sys.argv[1]))
