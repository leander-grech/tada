"""Tests for Tune MLflow callback factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config.config import Config
from training import mlflow_tune


def test_build_tune_mlflow_callbacks_returns_empty_for_tensorboard(monkeypatch):
    monkeypatch.setattr(Config, "LOGGING_BACKEND", "tensorboard")

    callbacks = mlflow_tune.build_tune_mlflow_callbacks()

    assert callbacks == []


def test_build_tune_mlflow_callbacks_builds_single_callback_with_config(monkeypatch):
    captured = {}

    def dummy_mlflow_logger_callback(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(Config, "LOGGING_BACKEND", "mlflow")
    monkeypatch.setattr(Config, "MLFLOW_TRACKING_URI", "file:./mlruns-test")
    monkeypatch.setattr(Config, "MLFLOW_EXPERIMENT_NAME", None)
    monkeypatch.setattr(Config, "EXPERIMENT_NAME", "fallback-experiment")
    monkeypatch.setattr(Config, "MLFLOW_TRACKING_TOKEN", "secret-token")
    monkeypatch.setattr(Config, "MLFLOW_SAVE_ARTIFACT", True)
    monkeypatch.setattr(Config, "MLFLOW_LOG_PARAMS_ON_TRIAL_END", False)
    monkeypatch.setattr(mlflow_tune, "BatchMLflowLoggerCallback", dummy_mlflow_logger_callback)

    callbacks = mlflow_tune.build_tune_mlflow_callbacks()

    assert len(callbacks) == 1
    assert captured["kwargs"] == {
        "tracking_uri": "file:./mlruns-test",
        "experiment_name": "fallback-experiment",
        "tracking_token": "secret-token",
        "save_artifact": True,
        "log_params_on_trial_end": False,
    }


@patch("training.mlflow_tune.MlflowClient")
def test_batch_callback_uses_log_batch_instead_of_individual_calls(mock_client_cls):
    """log_trial_result should call client.log_batch() instead of per-metric log_metric()."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    callback = mlflow_tune.BatchMLflowLoggerCallback(
        tracking_uri="file:./mlruns-test",
        experiment_name="test-experiment",
    )
    trial = MagicMock()
    callback._trial_runs = {}  # pylint: disable=protected-access
    callback._trial_runs[trial] = "run-123"  # pylint: disable=protected-access

    result = {
        "training_iteration": 5,
        "timesteps_total": 10000,
        "lifetime_env_steps": 10000,
        "env_runners": {
            "episode_return_mean": 42.0,
            "custom_metrics": {
                "episodes_end_success": 1.0,
            },
        },
        "non_numeric_field": "should_be_skipped",
    }

    callback.log_trial_result(iteration=5, trial=trial, result=result)

    mock_client.log_batch.assert_called_once()
    mock_client.log_metric.assert_not_called()

    call_args, call_kwargs = mock_client.log_batch.call_args
    logged_run_id = call_kwargs.get("run_id")
    if logged_run_id is None and call_args:
        logged_run_id = call_args[0]

    logged_metrics = call_kwargs.get("metrics")
    if logged_metrics is None and len(call_args) > 1:
        logged_metrics = call_args[1]

    assert logged_run_id == "run-123"
    assert logged_metrics is not None
    logged_values = {m.key: m.value for m in logged_metrics}
    assert logged_values == {
        "training_iteration": 5.0,
        "timesteps_total": 10000.0,
        "lifetime_env_steps": 10000.0,
        "env_runners/episode_return_mean": 42.0,
        "env_runners/custom_metrics/episodes_end_success": 1.0,
    }
    assert all(m.step == 10000 for m in logged_metrics)


def test_real_tuner_checkpoint_omits_private_callback_connection(tmp_path, monkeypatch):
    import pytest
    from ray import cloudpickle, tune
    from ray.tune.callback import CallbackList
    from ray.tune.impl.tuner_internal import TunerInternal

    secrets = ["audit-user", "audit-password", "audit-query", "audit-token"]
    uri = f"https://{secrets[0]}:{secrets[1]}@example.invalid/mlflow?token={secrets[2]}&view=all"
    callback = mlflow_tune.build_tune_mlflow_callbacks(
        {
            "backend": "mlflow",
            "tracking_uri": uri,
            "tracking_token": secrets[3],
            "experiment_name": "synthetic",
        }
    )[0]
    connected = []

    def connect(**kwargs):
        connected.append(kwargs["tracking_uri"] == uri and kwargs["tracking_token"] == secrets[3])
        # Mimic the installed utility retaining its client connection after setup.
        callback.mlflow_util.client = SimpleNamespace(tracking_uri=uri)

    monkeypatch.setattr(callback.mlflow_util, "setup_mlflow", connect)
    callback.setup()
    assert connected == [True]
    TunerInternal(
        trainable="__fake",
        run_config=tune.RunConfig(storage_path=str(tmp_path), name="privacy", callbacks=[callback]),
    )
    durable = tmp_path / "privacy" / "tuner.pkl"
    CallbackList([callback]).save_to_dir(str(durable.parent))
    if any(
        secret.encode() in path.read_bytes()
        for path in durable.parent.iterdir()
        if path.is_file()
        for secret in secrets
    ):
        pytest.fail("private connection leaked through durable callback state", pytrace=False)
    state = cloudpickle.loads(durable.read_bytes())
    restored = state["_run_config"].callbacks[0]
    for operation in (
        restored.setup,
        lambda: restored.log_trial_start(None),
        lambda: restored.log_trial_result(0, None, {}),
        lambda: restored.log_trial_end(None),
        restored.mark_active_trial_runs_killed,
    ):
        with pytest.raises(RuntimeError, match="restoration.*unsupported"):
            operation()
    # Serialization must not mutate the live authenticated connection.
    callback.setup()
    assert connected == [True, True]


def test_fresh_callback_logs_to_real_store_after_tuner_serialization(tmp_path, monkeypatch):
    from ray import tune
    from ray.tune.impl.tuner_internal import TunerInternal

    # Isolate MLflow's process environment from any operator credentials.
    for name in ("MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    uri = "sqlite:///" + str(tmp_path / "tracking.db")
    callback = mlflow_tune.build_tune_mlflow_callbacks(
        {
            "backend": "mlflow",
            "tracking_uri": uri,
            "experiment_name": "fresh-driver",
        }
    )[0]
    tuner = TunerInternal(
        trainable="__fake",
        run_config=tune.RunConfig(storage_path=str(tmp_path), name="fresh", callbacks=[callback]),
    )
    callbacks = tuner.get_run_config().callbacks
    assert callbacks is not None
    live = callbacks[0]
    assert live is callback
    live.setup()
    trial = MagicMock()
    trial.config = {"segment": {"logging": {"tracking_uri": uri}}}
    live.log_trial_start(trial)
    live.log_trial_result(1, trial, {"lifetime_env_steps": 7, "score": 2.5})
    live.log_trial_end(trial)
    client = mlflow_tune.MlflowClient(tracking_uri=uri)
    run = client.get_run(live._trial_runs[trial])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["score"] == 2.5
    assert client.get_metric_history(run.info.run_id, "score")[0].step == 7
