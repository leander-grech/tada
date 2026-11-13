"""Tune MLflow callback factory with batched metric logging."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mlflow.entities import Metric, RunStatus
from mlflow.tracking import MlflowClient
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.callback import Callback
from ray.tune.utils import flatten_dict

from config.config import Config, get_logging_backend

if TYPE_CHECKING:
    from ray.tune.experiment import Trial


_SECRET_QUERY_FIELDS = {
    "token",
    "access_token",
    "tracking_token",
    "api_key",
    "apikey",
    "key",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "auth",
    "authorization",
    "signature",
    "sig",
    "x-amz-credential",
    "x-amz-signature",
    "x-amz-security-token",
}


def public_logging_settings(settings: dict) -> dict:
    """Copy logging evidence without supported connection credentials.

    The original settings belong only to the live driver callback. URI fragments
    have no tracking endpoint semantics and are omitted from public evidence.
    """
    public = {key: value for key, value in settings.items() if key != "tracking_token"}
    for key in ("tracking_uri", "registry_uri"):
        uri = public.get(key)
        if not uri:
            continue
        try:
            parts = urlsplit(uri)
            public[key] = urlunsplit(
                (
                    parts.scheme,
                    parts.netloc.rsplit("@", 1)[-1],
                    parts.path,
                    urlencode(
                        [
                            (name, value)
                            for name, value in parse_qsl(parts.query, keep_blank_values=True)
                            if name.lower().replace("-", "_")
                            not in {field.replace("-", "_") for field in _SECRET_QUERY_FIELDS}
                        ]
                    ),
                    "",
                )
            )
        except ValueError:
            public[key] = "[invalid tracking URI omitted]"
    return public


class BatchMLflowLoggerCallback(MLflowLoggerCallback):
    """MLflow logger that uses log_batch for fewer DB writes per trial result.

    Ray's default MLflowLoggerCallback calls client.log_metric() once per metric,
    resulting in hundreds of individual SQLite transactions per iteration per trial.
    This subclass batches all metrics into a single client.log_batch() call.
    """

    _BATCH_SIZE = 1000  # MLflow API limit per log_batch call

    def __getstate__(self):
        # Tune writes RunConfig (including this object) into tuner.pkl before actors.
        # Never serialize mlflow_util: its clients can retain private connection URIs.
        return public_logging_settings(
            {
                "tracking_uri": self.tracking_uri,
                "registry_uri": self.registry_uri,
                "experiment_name": self.experiment_name,
                "tags": self.tags,
                "save_artifact": self.should_save_artifact,
                "log_params_on_trial_end": self.log_params_on_trial_end,
            }
        )

    def __setstate__(self, state):
        super().__init__(**state)
        self._restored_without_connection = True

    def _require_live_connection(self):
        if getattr(self, "_restored_without_connection", False):
            raise RuntimeError(
                "MLflow callback restoration is unsupported; construct a fresh segment "
                "callback with private connection settings"
            )

    def setup(self, *args, **kwargs):
        self._require_live_connection()
        return super().setup(*args, **kwargs)

    def log_trial_start(self, trial):
        self._require_live_connection()
        return super().log_trial_start(trial)

    def log_trial_result(self, iteration: int, trial: Trial, result: dict):
        self._require_live_connection()
        step = result.get("lifetime_env_steps")
        if type(step) is not int or step < 0:
            raise ValueError("MLflow requires actual lifetime_env_steps, not segment iteration")
        run_id = self._trial_runs[trial]
        timestamp_ms = int(time.time() * 1000)

        flat = flatten_dict(result)
        metrics = []
        for key, value in flat.items():
            try:
                numeric = float(value)
                if math.isfinite(numeric):
                    metrics.append(Metric(key, numeric, timestamp_ms, step))
            except (ValueError, TypeError):
                continue

        if not metrics:
            return

        client = MlflowClient(tracking_uri=self.tracking_uri)
        for i in range(0, len(metrics), self._BATCH_SIZE):
            client.log_batch(run_id, metrics=metrics[i : i + self._BATCH_SIZE])

    def log_trial_end(self, trial, failed=False):
        self._require_live_connection()
        # Tune can fail actor setup before on_trial_start creates an MLflow run.
        if trial in getattr(self, "_trial_runs", {}):
            super().log_trial_end(trial, failed=failed)

    def mark_active_trial_runs_killed(self) -> None:
        """Terminate this callback's still-running trial runs as KILLED.

        Used by the training driver on manual interruption so MLflow does not
        retain stale RUNNING rows for this process's current trials.
        """

        self._require_live_connection()
        trial_runs = getattr(self, "_trial_runs", {})
        if not trial_runs:
            return

        client = MlflowClient(tracking_uri=self.tracking_uri)
        for run_id in trial_runs.values():
            try:
                run = client.get_run(run_id)
                if run.info.status == RunStatus.to_string(RunStatus.RUNNING):
                    client.set_terminated(
                        run_id=run_id, status=RunStatus.to_string(RunStatus.KILLED)
                    )
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to mark MLflow run %s as KILLED", run_id, exc_info=True
                )


def snapshot_logging_settings() -> dict:
    """Read static operational defaults once in the driver; pass the snapshot explicitly."""
    return {
        "backend": get_logging_backend(),
        "tracking_uri": Config.MLFLOW_TRACKING_URI,
        "experiment_name": Config.MLFLOW_EXPERIMENT_NAME or Config.EXPERIMENT_NAME,
        "tracking_token": Config.MLFLOW_TRACKING_TOKEN,
        "save_artifact": Config.MLFLOW_SAVE_ARTIFACT,
        "log_params_on_trial_end": Config.MLFLOW_LOG_PARAMS_ON_TRIAL_END,
    }


def build_tune_mlflow_callbacks(settings: dict | None = None, *, segment=None):
    settings = dict(settings if settings is not None else snapshot_logging_settings())
    if settings.pop("backend") != "mlflow":
        return []
    if segment is not None:
        settings["tags"] = {
            key: str(segment[key]) if segment[key] is not None else "none"
            for key in (
                "run_id",
                "phase_id",
                "segment_id",
                "parent_run_id",
                "parent_phase_id",
                "parent_segment_id",
            )
        }
    return [BatchMLflowLoggerCallback(**settings)]


def mark_running_mlflow_runs_killed(callbacks: Iterable[Callback]) -> None:
    """Mark active runs owned by MLflow Tune callbacks as KILLED."""
    for callback in callbacks:
        if isinstance(callback, BatchMLflowLoggerCallback):
            callback.mark_active_trial_runs_killed()
