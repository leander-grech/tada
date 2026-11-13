"""Fixed execution contracts and explicit intermediate checkpoint refusal."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("mode", "checkpoint", "sets"),
    [
        ("resume", None, ()),
        ("phase", None, ("lr=0.1",)),
        ("phase", Path("saved"), ()),
        ("fresh", Path("saved"), ()),
        ("fresh", None, ("lr=0.2",)),
        ("resume", Path("saved"), ("lr=0.1",)),
        ("other", None, ()),
    ],
)
def test_unsupported_request_fails_before_reading_config(mode, checkpoint, sets):
    from training.lifecycle import RunRequest

    with pytest.raises(ValueError, match=mode):
        RunRequest(mode, Path("missing.json"), checkpoint, sets)


def test_fixed_execution_publishes_at_cadence_and_end(native_run):
    from training.tune_run_config import build_trainer_run_config

    config = build_trainer_run_config(native_run, "segment-test", callbacks=[])
    assert config.checkpoint_config is not None
    assert config.failure_config is not None
    assert config.stop == {"training_iteration": 2}
    assert (
        config.checkpoint_config.checkpoint_frequency
        == native_run.cadences["checkpoint_iterations"]
    )
    assert config.checkpoint_config.checkpoint_at_end
    assert config.failure_config.max_failures == 0


def test_explicit_checkpoint_operations_fail_without_algorithm():
    from training.trainable import SingleRunTrainable

    trial = object.__new__(SingleRunTrainable)
    with pytest.raises(ValueError, match="unsupported.*contract"):
        trial.save_checkpoint("unused")
    with pytest.raises(ValueError, match="unsupported.*contract"):
        trial.load_checkpoint("unused")


def test_uncapped_execution_and_separate_segment_journals(config_path):
    from training.ppo_config import build_ppo_config
    from training.run_config import load_run_config
    from training.tune_run_config import build_trainer_run_config

    run = load_run_config(config_path)
    assert build_trainer_run_config(run, "uncapped", callbacks=[]).stop is None
    first = build_ppo_config(run, scenario_artifact_dir=Path("/tmp/segment-a/scenarios"))
    second = build_ppo_config(run, scenario_artifact_dir=Path("/tmp/segment-b/scenarios"))
    assert first.env_config["scenario_artifact_dir"] != second.env_config["scenario_artifact_dir"]
    assert (
        first.env_config["resolved_env"] == second.env_config["resolved_env"] == run.env_payload()
    )


def test_cli_refuses_unsupported_options_before_algorithm():
    from main import main

    with pytest.raises(SystemExit) as result:
        main(["train", "--config", "missing.json", "--mode", "resume"])
    assert result.value.code == 2


def test_pre_actor_evidence_keeps_connection_secrets_private(config_path, native_run, monkeypatch):
    import json
    from types import SimpleNamespace

    from training import lifecycle
    from training.provenance import write_run_manifest

    secrets = ["segment-user", "segment-password", "segment-query", "segment-token"]
    uri = (
        f"https://{secrets[0]}:{secrets[1]}@example.invalid/path?access_token={secrets[2]}&view=all"
    )
    monkeypatch.setattr(
        lifecycle,
        "snapshot_logging_settings",
        lambda: {
            "backend": "mlflow",
            "tracking_uri": uri,
            "tracking_token": secrets[3],
            "experiment_name": "synthetic",
        },
    )
    monkeypatch.setattr(lifecycle, "verify_native_conditions", lambda run: None)
    monkeypatch.setattr(lifecycle, "ensure_sector_catalog_up_to_date", lambda: None)
    captured = {}

    class BeforeActors:
        def __init__(self, trainable, *, param_space, run_config, **kwargs):
            captured.update(param_space)
            callback = run_config.callbacks[0]
            if callback.tracking_uri != uri or callback.tracking_token != secrets[3]:
                pytest.fail("fresh callback lost private connection", pytrace=False)

        def fit(self):
            return [SimpleNamespace(error=None)]

    monkeypatch.setattr(lifecycle.tune, "Tuner", BeforeActors)
    segment_dir = lifecycle.run_training(lifecycle.RunRequest("fresh", config_path, None))
    # Exercise the same manifest writer and execution segment consumed by Trainable.setup.
    write_run_manifest(
        segment_dir / "manifest.json", captured["run"], {}, execution_segment=captured["segment"]
    )
    evidence = (segment_dir / "execution.json").read_text() + (
        segment_dir / "manifest.json"
    ).read_text()
    evidence += json.dumps(captured["segment"])
    if any(secret in evidence for secret in secrets):
        pytest.fail("private connection leaked through execution evidence", pytrace=False)
    assert captured["segment"]["logging"]["tracking_uri"].endswith("?view=all")
