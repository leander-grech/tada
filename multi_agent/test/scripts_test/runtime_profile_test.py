"""Runtime profile guards, with explicit profile choices."""

import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class RuntimeProfileTests(unittest.TestCase):
    def test_invalid_backend_is_rejected_before_install(self):
        result = subprocess.run(
            ["sh", str(ROOT / "scripts/runtime.sh"), "setup", "--backend", "invalid"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertIn("backend must be cpu or cuda", result.stderr)
        self.assertNotEqual(result.returncode, 0)

    @unittest.skipUnless(os.environ.get("TADA_PROFILE_TEST"), "requires provisioned runtime")
    def test_repeated_invocations_preserve_selected_backend_native_and_patch(self):
        backend = os.environ["TADA_PROFILE_TEST"]
        for _ in range(2):
            subprocess.run(
                [
                    "sh",
                    str(ROOT / "scripts/runtime.sh"),
                    "runtime-check",
                    "--device",
                    "cpu",
                    "--expect-backend",
                    backend,
                ],
                env={**os.environ, "OPENBLAS_NUM_THREADS": "97"},
                check=True,
            )

    def test_malformed_supplied_archive_cannot_replace_selected_native(self):
        from common import native_runtime

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            state.mkdir()
            selected = state / "native.json"
            selected.write_text('{"selected": "preserve-me"}')
            archive = Path(temporary) / "invalid.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.whl", b"not a wheel")
            with patch.object(native_runtime, "STATE", state):
                with self.assertRaisesRegex(RuntimeError, "exactly one wheel"):
                    native_runtime.install_native(str(archive))
            self.assertEqual(selected.read_text(), '{"selected": "preserve-me"}')
            self.assertFalse((Path(temporary) / "escape.whl").exists())

    def test_changed_retained_artifact_fails_before_reinstall(self):
        from common import native_runtime

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            artifact = state / "input.zip"
            artifact.write_bytes(b"changed")
            selected = {
                "mode": "external",
                "supplied_artifact": str(artifact),
                "supplied_sha256": "unmatched-checksum",
            }
            record = state / "native.json"
            record.write_text(json.dumps(selected))
            with patch.object(native_runtime, "STATE", state):
                with self.assertRaisesRegex(RuntimeError, "artifact checksum changed"):
                    native_runtime.install_native()
            self.assertEqual(json.loads(record.read_text()), selected)

    @unittest.skipUnless(os.environ.get("TADA_RAY_PROFILE"), "starts actual Ray runner and learner")
    def test_ray_workers_keep_selected_native_patch_and_thread_policy(self):
        subprocess.run(
            [
                "sh",
                str(ROOT / "scripts/runtime.sh"),
                "run",
                "python",
                "-c",
                "from test.scripts_test.runtime_profile_test import probe_ray_workers; probe_ray_workers()",
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "97",
                "OMP_NUM_THREADS": "83",
                "CUDA_VISIBLE_DEVICES": "",
            },
            check=True,
        )


def probe_ray_workers():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from atc_env.environment import ATCEnv
    from common.runtime_log import runtime_print
    from config.config import Config
    from config.settings import settings_from_config
    from config.simulation_config_path import get_config_path
    from network.rlm import ATCRLM, unbound_model_config
    from training.learner_checkpoint import ATCPPOLearner

    def report(actor):
        import scipy.linalg
        import torch
        from threadpoolctl import threadpool_info

        scipy.linalg.norm([[1.0]])
        pools = threadpool_info()
        from common.native_runtime import verify_native

        native = verify_native()
        result = {
            "native": native,
            "classes": [cls.__name__ for cls in type(actor).__mro__],
            "class": type(actor).__name__,
            "pid": os.getpid(),
            "omp": os.environ.get("OMP_NUM_THREADS"),
            "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
            "torch": torch.get_num_threads(),
            "pools": pools,
        }
        assert result["omp"] == result["openblas"] == "1", result
        assert result["torch"] == 1, result
        assert all(p["num_threads"] == 1 for p in pools), result
        return result

    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        _temp_dir="/tmp/tada-runtime-ray",
    )
    algorithm = None
    try:
        config = (
            PPOConfig()
            .framework("torch")
            .environment(
                ATCEnv,
                env_config={
                    "simulation_config_path": get_config_path(),
                    "max_aircraft_count": 2,
                    "max_steps_per_episode": 8,
                },
            )
            .env_runners(num_env_runners=1, num_cpus_per_env_runner=1, rollout_fragment_length=8)
            .learners(
                learner_class=ATCPPOLearner,
                num_learners=1,
                num_cpus_per_learner=1,
                num_gpus_per_learner=0,
            )
            .resources(num_cpus_for_main_process=1)
            .multi_agent(
                policies={Config.POLICY_NAME},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: Config.POLICY_NAME,
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        Config.POLICY_NAME: RLModuleSpec(
                            module_class=ATCRLM,
                            model_config={
                                **unbound_model_config(settings_from_config().architecture),
                                "max_seq_len": 8,
                            },
                        )
                    }
                )
            )
            .training(train_batch_size_per_learner=16, minibatch_size=8, num_epochs=1)
        )
        algorithm = config.build_algo()
        reports = {
            "runner": algorithm.env_runner_group.foreach_env_runner(report),
            "learner": [item.get() for item in algorithm.learner_group.foreach_learner(report)],
        }
        runtime_print("RuntimeThreads", "verified", report=json.dumps(reports))
        with open(str(ROOT / "runs/runtime/ray-workers.json"), "w") as stream:
            json.dump(reports, stream, indent=2)
    finally:
        try:
            if algorithm is not None:
                algorithm.stop()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
