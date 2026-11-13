"""Helpers for keeping local inference script output concise."""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from collections.abc import Iterator


def configure_default_quiet_env() -> None:
    """Apply quiet defaults for local Ray/RLlib script runs."""
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
    warnings.filterwarnings("ignore", category=FutureWarning, module="ray")
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


def quiet_logs_enabled() -> bool:
    """Whether noisy setup logs should be suppressed."""
    return os.environ.get("INFERENCE_QUIET_LOGS", "1") == "1"


def set_ray_logger_levels() -> None:
    """Reduce Ray/RLlib logger verbosity."""
    for logger_name in ("ray", "ray.rllib", "ray.tune"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_stdio(enabled: bool) -> Iterator[None]:
    """Optionally suppress stdout/stderr for noisy setup blocks."""
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
