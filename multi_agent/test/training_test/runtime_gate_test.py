"""Native identity and real owned actor selection precede scenario generation."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from common.hashing import content_digest


def test_native_gate_checks_actual_artifact_not_provided_evidence(native_run):
    from training.runtime_gate import verify_native_conditions

    actual = verify_native_conditions(native_run)
    assert actual["version"] == native_run.reserved_conditions["native_artifact"]["version"]
    for field in ("version", "wheel_sha256", "extension_sha256"):
        conditions = dict(native_run.reserved_conditions)
        conditions["native_artifact"] = dict(conditions["native_artifact"], **{field: "bad"})
        conditions.pop("conditions_sha256")
        conditions["conditions_sha256"] = content_digest(conditions)
        with pytest.raises(ValueError, match="native.*mismatch"):
            verify_native_conditions(replace(native_run, reserved_conditions=conditions))


def test_builder_binds_payloads_constants_and_locks_reset_before_selection(native_run):
    from training.env import TrainingATCEnv
    from training.ppo_config import build_ppo_config

    cfg = build_ppo_config(native_run)
    assert cfg.lr == native_run.optimizer["learning_rate"]
    assert cfg.entropy_coeff == native_run.optimizer["entropy_coeff"]
    assert cfg.use_kl_loss is False
    eval_config = cfg.get_evaluation_config_object()
    assert eval_config is not None
    for evaluation in (False, True):
        env_config = eval_config.env_config if evaluation else cfg.env_config
        env = TrainingATCEnv(env_config)
        try:
            assert env.consumed_semantic_digest == native_run.semantic_digest
            assert env.feature_settings.deviation_encoding == "signed_log"
            assert env.feature_settings.neighbor_selection == "risk"
            assert env.max_aircraft_count == native_run.scenario["aircraft_count"]
            assert env.success_time_dev_s == (
                60 if evaluation else native_run.thresholds["training_success_s"]
            )
            assert env.reward_settings == native_run.reward
            assert env.simulator.world is None
            state = env.get_training_stream_state()
            with pytest.raises(RuntimeError, match="activation"):
                env.reset()
            assert env.get_training_stream_state() == state
            assert env.simulator.world is None
        finally:
            env.close()


def test_supported_model_spec_uses_explicit_constructor_kwargs(native_run, monkeypatch):
    import gymnasium as gym

    from models.observation_settings import ObservationSettings
    from models.observation_space import build_observation_space
    from network.rlm import ATCRLM
    from training.ppo_config import POLICY_ID, build_ppo_config

    calls = []
    original = ATCRLM.__init__

    def traced(self, *args, **kwargs):
        calls.append((args, set(kwargs)))
        original(self, *args, **kwargs)

    monkeypatch.setattr(ATCRLM, "__init__", traced)
    spec = build_ppo_config(native_run).rl_module_spec.rl_module_specs[POLICY_ID]
    spec.observation_space = build_observation_space(
        ObservationSettings.from_features(native_run.features)
    )
    spec.action_space = gym.spaces.Discrete(9)
    model = spec.build()
    assert model.consumed_semantic_digest == native_run.semantic_digest
    assert calls == [
        (
            (),
            {
                "observation_space",
                "action_space",
                "inference_only",
                "model_config",
                "catalog_class",
            },
        )
    ]


def test_bound_evaluation_margin_keeps_parent_training_window(native_run, config_path):
    import json

    from training.env import TrainingATCEnv
    from training.ppo_config import build_ppo_config
    from training.run_config import load_run_config

    data = json.loads(config_path.read_text())
    data["thresholds"]["training_success_s"] = 40.0
    config_path.write_text(json.dumps(data))
    cfg = build_ppo_config(load_run_config(config_path))
    eval_config = cfg.get_evaluation_config_object()
    assert eval_config is not None
    for env_config in (cfg.env_config, eval_config.env_config):
        env = TrainingATCEnv(env_config)
        try:
            assert env._compute_success_margin(
                deviation_proxies={"A": 10.0}, arrival_target_callsigns={"A"}
            ) == pytest.approx(0.75)
        finally:
            env.close()


def test_bound_generation_rejects_ignored_auto_solve_override(native_run, config_path):
    import json

    from training.env import TrainingATCEnv
    from training.ppo_config import build_ppo_config
    from training.run_config import load_run_config

    data = json.loads(config_path.read_text())
    data["thresholds"]["auto_solve_s"] += 13.0
    config_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="auto.solve|preprocessing"):
        TrainingATCEnv(build_ppo_config(load_run_config(config_path)).env_config)


def test_direct_bound_environment_rejects_native_mismatch(native_run, config_path):
    import json

    from training.env import TrainingATCEnv
    from training.run_config import load_run_config

    conditions_path = config_path.parent / "conditions.json"
    conditions = json.loads(conditions_path.read_text())
    conditions["native_artifact"]["wheel_sha256"] = "f" * 64
    conditions.pop("conditions_sha256")
    conditions["conditions_sha256"] = content_digest(conditions)
    conditions_path.write_text(json.dumps(conditions))
    run = load_run_config(config_path)
    with pytest.raises(ValueError, match="native.*mismatch"):
        TrainingATCEnv(
            {
                "resolved_env": run.env_payload(),
                "expected_semantic_digest": run.semantic_digest,
                "scenario_artifact_dir": str(config_path.parent / "never-generated"),
            }
        )


def test_bound_reset_rechecks_preprocessing_after_config_drift(native_run, monkeypatch):
    from config.config import Config
    from training.env import TrainingATCEnv

    env = TrainingATCEnv(
        {
            "resolved_env": native_run.env_payload(),
            "expected_semantic_digest": native_run.semantic_digest,
            "scenario_artifact_dir": native_run.execution["output_root"],
        }
    )
    try:
        monkeypatch.setattr(
            Config, "AUTO_SOLVE_TARGET_WINDOW_S", Config.AUTO_SOLVE_TARGET_WINDOW_S + 1
        )
        with pytest.raises(ValueError, match="preprocessing"):
            env.reset()
        assert env.simulator.world is None
        assert env.get_training_stream_state()["stream"]["cursor"] == 0
    finally:
        env.close()


def test_pristine_actor_cannot_release_unrestored_stream_progress(native_run):
    from types import SimpleNamespace

    from training.env import TrainingATCEnv
    from training.ppo_config import build_ppo_config
    from training.runtime_gate import _verify_environment_identity

    env = TrainingATCEnv(build_ppo_config(native_run).env_config)
    actor = SimpleNamespace(worker_index=0, config=SimpleNamespace(in_evaluation=False))
    try:
        assert env.get_training_stream_state()["stream"]["cursor"] == 0
        assert env._training_stream is not None
        env._training_stream.next_seed()
        with pytest.raises(ValueError, match="actual environment stream advanced"):
            _verify_environment_identity(env, actor, native_run, "training", 0)
        assert env.simulator.world is None
        assert not env._has_reset_once
        assert not env._runtime_activated
        assert env._scenario_artifact_path is not None
        assert not env._scenario_artifact_path.exists()
        assert env.get_training_stream_state()["stream"]["cursor"] == 1
    finally:
        env.close()


def test_supported_ray_matches_the_installed_runtime():
    import ray

    from training.runtime_gate import SUPPORTED_RAY

    assert SUPPORTED_RAY == ray.__version__


@pytest.mark.parametrize("seed,late", [("13", False), ("random", False), ("13", True)])
def test_effective_hash_policy_rejects_randomization_and_late_environment_mutation(seed, late):
    script = "import os\n"
    if late:
        script += "os.environ['PYTHONHASHSEED']='0'\n"
    script += "from training.runtime_gate import verify_hash_policy\nverify_hash_policy()"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONHASHSEED=seed),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "effective Python hash policy" in result.stderr


def test_hash_policy_is_real_process_stable():
    script = "import json\nfrom training.runtime_gate import verify_hash_policy\nfrom common.runtime_log import runtime_print\nruntime_print('Hash', 'proof', policy=verify_hash_policy(), order=list(set(['aircraft-a','aircraft-b','aircraft-c'])))"
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONHASHSEED="0"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]


def test_checkpoint_contract_rejects_changed_process_hash_policy(native_run):
    from training import versions
    from training.checkpoint_manifest import current_identity, require_compatible

    identity = current_identity(native_run)
    saved = json.loads(json.dumps({"format": 1, "versions": versions.current(), **identity}))
    assert "python_hash_policy" in saved["runtime"]
    saved["runtime"]["python_hash_policy"] = {"seed": "changed"}
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        require_compatible(saved, identity)


def test_phase_identity_moves_only_inference_placement(native_run):
    from training import versions
    from training.checkpoint_manifest import current_identity, require_compatible
    from training.phase import INFERENCE_PLACEMENT

    identity = current_identity(native_run)
    saved = json.loads(json.dumps({"format": 1, "versions": versions.current(), **identity}))
    profile = saved["runtime"]["resource_profile"]
    profile.update(inference_device="cuda", gpus_per_runner=0.075, inference_gpu_uuid="GPU-other")
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        require_compatible(saved, identity)
    require_compatible(saved, identity, movable=INFERENCE_PLACEMENT)
    profile["learner_device"] = "changed"
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        require_compatible(saved, identity, movable=INFERENCE_PLACEMENT)
