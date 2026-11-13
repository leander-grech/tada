"""CLI wrapper for launching the local MLflow UI."""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Sequence


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for `mlflow-ui` script."""
    parser = argparse.ArgumentParser(description="Start MLflow UI with project defaults.")
    parser.add_argument(
        "--backend-store-uri",
        default=os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db"),
        help="MLflow backend store URI (default: sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MLFLOW_UI_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_int_env("MLFLOW_UI_PORT") or 6006,
        help="Bind port (default: 6006).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_int_env("MLFLOW_UI_WORKERS"),
        help="Optional number of MLflow UI workers.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run `mlflow ui` using defaults with optional CLI/env overrides."""
    parser = build_parser()
    args, passthrough = parser.parse_known_args(list(argv) if argv is not None else None)

    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        args.backend_store_uri,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    cmd.extend(passthrough)
    return subprocess.call(cmd)
