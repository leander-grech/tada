"""Public complete bundles must replace the temporary publication refusal."""


def test_tune_checkpoint_publishes_complete_bundle(tmp_path, native_run, monkeypatch):
    from training import trainable

    destination = tmp_path / "checkpoint"
    calls = []
    algorithm = object()

    def capture(actual, path, run, *, advertise_latest):
        assert not advertise_latest
        assert actual is algorithm
        assert run is native_run
        calls.append(path)
        return path

    monkeypatch.setattr(trainable, "capture_checkpoint", capture)
    trial = object.__new__(trainable.SingleRunTrainable)
    trial.algorithm = algorithm
    trial.run = native_run
    assert trial.save_checkpoint(str(destination)) == str(destination)
    assert calls == [destination]


def test_periodic_and_end_producers_are_enabled(native_run):
    from training.tune_run_config import build_trainer_run_config

    config = build_trainer_run_config(native_run, "bundle-segment")
    assert config.checkpoint_config is not None
    assert (
        config.checkpoint_config.checkpoint_frequency
        == native_run.cadences["checkpoint_iterations"]
    )
    assert config.checkpoint_config.checkpoint_at_end


def test_bundle_preflight_rejects_missing_model_before_learner_mutation():
    import copy
    import importlib
    import importlib.util
    import pickle

    import pytest
    from support.bootstrap import learner_config

    spec = importlib.util.find_spec("training.checkpoint_preflight")
    assert spec is not None, "component-aware preflight is missing"
    api = importlib.import_module("training.checkpoint_preflight")
    learner = learner_config().training(use_kl_loss=False).build_learner()
    saved = learner.get_state()
    bad = copy.deepcopy(saved)
    del bad["rl_module"]
    before = pickle.dumps(learner.get_state())
    with pytest.raises(ValueError):
        api.validate_learner_state(learner, bad)
    assert pickle.dumps(learner.get_state()) == before


def test_scalar_normalizer_restores_exact_float64():
    import gymnasium as gym
    import torch

    from atc_env.env_config import observation_settings_from_config
    from config.config import Config
    from config.settings import settings_from_config
    from models.observation_space import build_observation_space
    from network.rlm import ATCRLM, unbound_model_config

    module = ATCRLM(
        observation_space=build_observation_space(observation_settings_from_config()),
        action_space=gym.spaces.Discrete(Config.NUM_ACTIONS),
        model_config={
            **unbound_model_config(settings_from_config().architecture),
            "max_seq_len": 2,
        },
    )
    value = -0.011262123456789
    module.value_normalizer.mean_acc.fill_(value)
    saved = module.get_state()
    module.value_normalizer.mean_acc.zero_()
    module.set_state(saved)
    assert module.value_normalizer.mean_acc.dtype == torch.float64
    assert module.value_normalizer.mean_acc.item() == value


def _state_writer(path):
    (path / "state.pkl").write_bytes(b"state")
    return {"format": 1}


def _verify(root):
    from common.jsonio import read_json
    from training.checkpoint_manifest import verify_files

    verify_files(root, read_json(root / "manifest.json")["files"])


def test_semantic_and_artifact_compatibility_reject_before_construction(native_run, config_path):
    import copy
    import json

    import pytest

    from training.checkpoint_manifest import current_identity, require_compatible
    from training.phase import PhaseTransition, saved_run
    from training.run_config import load_run_config

    saved = native_run.to_dict()
    original = json.loads(config_path.read_text())
    changes = [
        ("root_seed", None, original["root_seed"] + 1),
        ("topology", "env_runners", 2),
        ("topology", "phase_slot", 1),
        ("scenario", "aircraft_count", 3),
        ("thresholds", "lockout_s", 23),
        ("features", "knee_s", 18),
        ("reward", "action_penalty_weight", 0.17),
        ("optimizer", "learning_rate", 0.007),
        ("optimizer", "train_batch_size", 16),
    ]
    for section, key, value in changes:
        data = copy.deepcopy(original)
        if key is None:
            data[section] = value
        else:
            data[section][key] = value
        config_path.write_text(json.dumps(data))
        requested = load_run_config(config_path)
        with pytest.raises(ValueError):
            PhaseTransition.between(saved_run(saved), requested, {}, mode="resume")
    config_path.write_text(json.dumps(original))
    identity = current_identity(native_run)
    manifest: dict = {"format": 1, "versions": native_run.versions, **copy.deepcopy(identity)}
    require_compatible(manifest, identity)
    for section in ("native", "catalog", "runtime"):
        changed = copy.deepcopy(manifest)
        changed[section]["unknown"] = "same semantic digest, substituted behavior/input"
        with pytest.raises(ValueError, match=section):
            require_compatible(changed, identity)


def test_tune_storage_uses_atomic_final_publication(tmp_path):
    from types import SimpleNamespace

    from pyarrow.fs import LocalFileSystem

    from training.checkpoint_bundle import publish_bundle
    from training.trainable import SingleRunTrainable

    source = tmp_path / "temporary"
    final = tmp_path / "storage" / "checkpoint_000001"
    publish_bundle(source, _state_writer)

    def unsupported_copy(*args):
        raise AssertionError("non-atomic stock storage copy was used")

    trial = object.__new__(SingleRunTrainable)
    trial._last_result = {"training_iteration": 1}
    trial._storage = SimpleNamespace(
        storage_filesystem=LocalFileSystem(),
        checkpoint_fs_path=str(final),
        _update_checkpoint_index=lambda metrics: None,
        _check_validation_file=lambda: None,
        persist_current_checkpoint=unsupported_copy,
        persist_artifacts=lambda **kwargs: None,
        sync_config=SimpleNamespace(sync_artifacts_on_checkpoint=False),
    )
    result = trial._report_class_trainable_checkpoint(str(source), str(source))
    assert result.checkpoint.path == str(final)
    _verify(final)
    assert (final.parent / "latest.json").exists()


def test_owned_stop_publishes_only_a_completed_unsaved_update(tmp_path, native_run, monkeypatch):
    from types import SimpleNamespace

    from training import trainable

    calls = []
    trial = object.__new__(trainable.SingleRunTrainable)
    trial.algorithm = SimpleNamespace(training_iteration=2, stop=lambda: calls.append("stopped"))
    trial.run = native_run
    trial.segment_dir = tmp_path
    trial._last_step_completed = True
    trial._checkpoint_saved_iteration = 1
    monkeypatch.setattr(
        trainable, "capture_checkpoint", lambda algorithm, path, run: calls.append(path)
    )
    trial.cleanup()
    assert calls == [tmp_path / "stop" / "checkpoint_000002", "stopped"]
    assert trial.algorithm is None
    trial.cleanup()
    assert len(calls) == 2
    trial.algorithm = SimpleNamespace(training_iteration=3, stop=lambda: calls.append("stopped"))
    trial._last_step_completed = False
    trial.cleanup()
    assert calls[-1] == "stopped" and len(calls) == 3


def test_actual_bundle_components_reject_before_new_algorithm(native_run, tmp_path, monkeypatch):
    import copy
    import json
    import pickle
    import shutil

    import pytest
    import ray

    from training import checkpoints
    from training.checkpoint_bundle import publish_bundle

    ray.init(num_cpus=4, include_dashboard=False)
    algorithm = None
    try:
        algorithm = checkpoints.build_algorithm(native_run)
        algorithm.train()
        parent = checkpoints.capture_checkpoint(algorithm, tmp_path / "parent", native_run)
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid payload reached Algorithm construction")

    monkeypatch.setattr(checkpoints, "build_algorithm", forbidden)
    original = pickle.loads((parent / "algorithm.pkl").read_bytes())
    boundary = pickle.loads((parent / "boundary.pkl").read_bytes())

    def remove_normalizer(state, processes):
        del state["learner_group"]["learner"]["rl_module"]["shared_policy"][
            "value_normalizer.mean_acc"
        ]

    def remove_weight(state, processes):
        module = state["learner_group"]["learner"]["rl_module"]["shared_policy"]
        del module[next(name for name in module if not name.startswith("value_normalizer."))]

    def wrong_rng(state, processes):
        processes["processes"]["training/1"]["rng"]["version"] = 999

    def wrong_boundary(state, processes):
        processes["version"] = 999

    def absent_optimizer(state, processes):
        state["learner_group"]["learner"]["optimizer"] = {}

    for index, damage in enumerate(
        (remove_normalizer, remove_weight, wrong_rng, wrong_boundary, absent_optimizer)
    ):
        state, processes = copy.deepcopy(original), copy.deepcopy(boundary)
        damage(state, processes)

        def writer(path):
            for source in parent.iterdir():
                if source.name != "manifest.json":
                    shutil.copyfile(source, path / source.name)
            body = json.loads((parent / "manifest.json").read_text())
            del body["files"]
            if index == 1:
                from training.checkpoint_preflight import module_payload_layout

                body["module_layout"] = module_payload_layout(state)
            (path / "algorithm.pkl").write_bytes(pickle.dumps(state))
            (path / "boundary.pkl").write_bytes(pickle.dumps(processes))
            return body

        invalid = publish_bundle(tmp_path / f"invalid-{index}", writer)
        with pytest.raises(ValueError):
            checkpoints.inspect_checkpoint(invalid, native_run)


def test_tune_failed_final_copy_preserves_prior_advertisement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pyarrow.fs import LocalFileSystem

    from training import checkpoint_tune
    from training.checkpoint_bundle import publish_bundle
    from training.trainable import SingleRunTrainable

    root = tmp_path / "storage"
    first = publish_bundle(root / "checkpoint_000000", _state_writer)
    latest = (root / "latest.json").read_bytes()
    source = publish_bundle(tmp_path / "temporary", _state_writer, advertise_latest=False)
    trial = object.__new__(SingleRunTrainable)
    trial._last_result = {"training_iteration": 2}
    trial._storage = SimpleNamespace(
        storage_filesystem=LocalFileSystem(),
        checkpoint_fs_path=str(root / "checkpoint_000001"),
        _update_checkpoint_index=lambda metrics: None,
        _check_validation_file=lambda: None,
        persist_artifacts=lambda **kwargs: None,
        sync_config=SimpleNamespace(sync_artifacts_on_checkpoint=False),
    )

    def failed(*args, **kwargs):
        raise OSError("injected final Tune copy failure")

    monkeypatch.setattr(checkpoint_tune.shutil, "copyfile", failed)
    import pytest

    with pytest.raises(OSError, match="injected final"):
        trial._report_class_trainable_checkpoint(str(source), str(source))
    assert (root / "latest.json").read_bytes() == latest
    assert not (root / "checkpoint_000001").exists()
    _verify(first)
    assert not (source.parent / "latest.json").exists()


def test_current_meta_model_preflight_preserves_rng_and_no_world(native_run, monkeypatch):
    import pickle
    import random

    import numpy as np
    import pytest
    import torch

    from atc_env.environment import ATCEnv
    from training.checkpoint_preflight import validate_current_model_payload

    monkeypatch.setattr(
        ATCEnv, "__init__", lambda *args, **kwargs: pytest.fail("preflight constructed environment")
    )
    before = pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()))
    cuda_before = torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="components"):
        validate_current_model_payload(
            {"learner_group": {"learner": {"rl_module": {}}}}, native_run
        )
    after = pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()))
    assert before == after
    assert torch.cuda.is_initialized() == cuda_before


def test_nested_process_envelopes_reject_before_factory_or_pristine_stop(
    native_run, tmp_path, monkeypatch
):
    import copy
    import json
    import pickle
    import shutil

    import ray

    from training import checkpoints
    from training.checkpoint_bundle import publish_bundle
    from training.trainable import SingleRunTrainable

    ray.init(num_cpus=4, include_dashboard=False)
    algorithm = None
    try:
        algorithm = checkpoints.build_algorithm(native_run)
        algorithm.train()
        parent = checkpoints.capture_checkpoint(algorithm, tmp_path / "parent", native_run)
        algorithm.stop()
        algorithm = checkpoints.build_algorithm(native_run)
        real_stop = algorithm.stop
        pristine = checkpoints.capture_boundary_state(algorithm, native_run)
        trial = object.__new__(SingleRunTrainable)
        trial.run = native_run
        trial.algorithm = algorithm
        boundary = pickle.loads((parent / "boundary.pkl").read_bytes())
        runner = ("processes", "training/1", "runner")
        env = (*runner, "environment")
        stream = (*env, "stream")
        local = ("processes", "driver", "local_runners")
        mutations = [
            (runner, "environment", "delete"),
            (env, "stream", "delete"),
            (stream, "stream", "delete"),
            (runner, "version", 999),
            (env, "version", 999),
            (stream, "version", 999),
            ((*stream, "stream"), "version", 999),
            (runner, "version", True),
            (env, "version", True),
            (stream, "version", True),
            ((*stream, "stream"), "version", True),
            (("processes", "training/1"), "runner", {}),
            (local, "training", "delete"),
            (local, "extra", {}),
            ((*local, "training"), "version", 999),
            ((*local, "training"), "version", True),
            ((*local, "training"), "environment", {}),
            (env, "unexpected", 1),
            (stream, "stream", None),
            (("processes", "driver"), "local_runners", None),
            ((*local, "training"), "unexpected", 1),
            (env, "boundary", []),
            (stream, "semantic_digest", "0" * 64),
            ((*stream, "stream"), "domain", "operational_evaluation"),
        ]
        calls = []

        def forbidden_factory(*args, **kwargs):
            calls.append("factory")
            raise AssertionError("malformed envelope reached Algorithm factory")

        def forbidden_stop():
            calls.append("stop")
            raise AssertionError("malformed envelope stopped pristine Algorithm")

        monkeypatch.setattr(checkpoints, "build_algorithm", forbidden_factory)
        monkeypatch.setattr(algorithm, "stop", forbidden_stop)
        violations = []
        for index, (path, key, value) in enumerate(mutations):
            damaged = copy.deepcopy(boundary)
            target = damaged
            for component in path:
                target = target[component]
            if value == "delete":
                del target[key]
            else:
                target[key] = value

            def writer(destination):
                for source in parent.iterdir():
                    if source.name != "manifest.json":
                        shutil.copyfile(source, destination / source.name)
                (destination / "boundary.pkl").write_bytes(pickle.dumps(damaged))
                body = json.loads((parent / "manifest.json").read_text())
                del body["files"]
                return body

            invalid = publish_bundle(tmp_path / f"invalid-{index}", writer)
            for reader in ("factory", "low-level"):
                trial.algorithm = algorithm
                try:
                    if reader == "factory":
                        checkpoints.load_checkpoint_algorithm(
                            checkpoints.inspect_checkpoint(invalid, native_run), native_run
                        )
                    else:
                        trial.load_checkpoint(str(invalid))
                except ValueError as error:
                    if not any(word in str(error) for word in ("runner", "scenario", "facade")):
                        violations.append((index, reader, repr(error)))
                except Exception as error:
                    violations.append((index, reader, repr(error)))
                else:
                    violations.append((index, reader, "accepted malformed envelope"))
        assert calls == [], (calls, violations)
        assert violations == []
        assert trial.algorithm is algorithm
        assert algorithm.training_iteration == 0
        after = checkpoints.capture_boundary_state(algorithm, native_run)
        assert (
            after["processes"]["learner/0"]["effective"]
            == pristine["processes"]["learner/0"]["effective"]
        )
        for identity in ("training/1", "operational_evaluation/1"):
            assert (
                after["processes"][identity]["runner"]["environment"]
                == pristine["processes"][identity]["runner"]["environment"]
            )
    finally:
        if algorithm is not None:
            if "real_stop" in locals():
                real_stop()
            else:
                algorithm.stop()
        ray.shutdown()
