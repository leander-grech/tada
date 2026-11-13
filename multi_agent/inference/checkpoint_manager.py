"""Manages discovery and lazy loading of model checkpoints."""

import logging
import re
import zipfile
from pathlib import Path
from typing import BinaryIO

from inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_NAME = "latest"

RLLIB_MARKER = "manifest.json"

MAX_CHECKPOINT_UNPACKED_SIZE = 2 * 1024**3  # 2 GiB

MAX_CHECKPOINT_MEMBERS = 4096

_CHECKPOINT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def checkpoint_name_from_upload(filename: str) -> str:
    """Return the checkpoint directory name for an uploaded ``<name>.zip``."""
    name = filename[: -len(".zip")] if filename.lower().endswith(".zip") else filename
    if not _CHECKPOINT_NAME.fullmatch(name) or ".." in name:
        raise ValueError(f"Invalid checkpoint name {name!r}")
    return name


def extract_checkpoint_zip(fileobj: BinaryIO, destination: Path) -> None:
    """Extract an uploaded archive after bounding its size and member paths."""
    with zipfile.ZipFile(fileobj) as archive:
        members = archive.infolist()
        if len(members) > MAX_CHECKPOINT_MEMBERS:
            raise ValueError(
                f"Archive has {len(members)} members, more than {MAX_CHECKPOINT_MEMBERS}"
            )
        unpacked = sum(member.file_size for member in members)
        if unpacked > MAX_CHECKPOINT_UNPACKED_SIZE:
            raise ValueError(
                f"Archive unpacked size {unpacked} exceeds {MAX_CHECKPOINT_UNPACKED_SIZE} bytes"
            )
        root = destination.resolve()
        for member in members:
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"Archive member {member.filename!r} escapes the checkpoint")
        archive.extractall(destination)


def validate_checkpoint(checkpoint_path: Path) -> None:
    """Authenticate complete saved semantics and mandatory model state."""
    from evaluation.checkpoint_adapter import inspect_inference_checkpoint

    inspect_inference_checkpoint(checkpoint_path)


class CheckpointManager:
    """Discovers available checkpoints and loads one engine at a time.

    Scans a directory for subdirectories containing ``rllib_checkpoint.json``.
    Only one InferenceEngine is kept in memory; switching shuts down the
    previous engine before loading the new one.
    """

    def __init__(
        self,
        checkpoints_dir: str | Path,
        max_checkpoints: int,
        airport: str | None = None,
    ) -> None:
        self._base_dir = Path(checkpoints_dir).absolute()
        self._available: list[str] = []
        self._active_name: str | None = None
        self._engine: InferenceEngine | None = None
        self._max_checkpoints = max_checkpoints
        self._airport = airport

    def discover(self) -> list[str]:
        """Scan base directory for valid checkpoint subdirectories."""
        if not self._base_dir.is_dir():
            self._available = []
            return self._available

        self._available = sorted(
            entry.name
            for entry in self._base_dir.iterdir()
            if entry.is_dir() and (entry / RLLIB_MARKER).is_file()
        )
        return list(self._available)

    @property
    def base_dir(self) -> Path:
        """Return the base directory being scanned for checkpoints."""
        return self._base_dir

    @property
    def available(self) -> list[str]:
        """Return the last discovered list of checkpoint names."""
        return list(self._available)

    @property
    def active(self) -> str | None:
        """Return the name of the currently loaded checkpoint, or None."""
        return self._active_name

    @property
    def engine(self) -> InferenceEngine:
        """Return the active InferenceEngine. Raises if none is loaded."""
        if self._engine is None:
            raise RuntimeError("No checkpoint is currently loaded")
        return self._engine

    def is_full(self) -> bool:
        """Check if the checkpoint directory has reached maximum capacity."""
        return len(self._available) >= self._max_checkpoints

    def select_default(self) -> str | None:
        """Pick the default checkpoint name from the available list.

        Prefers a checkpoint named ``DEFAULT_CHECKPOINT_NAME`` ("latest").
        Falls back to the last name alphabetically. Returns None if empty.
        """
        if not self._available:
            return None
        if DEFAULT_CHECKPOINT_NAME in self._available:
            return DEFAULT_CHECKPOINT_NAME
        return self._available[-1]

    def load(self, name: str) -> None:
        """Validate, shut down any active engine, and load the named checkpoint.

        Validation runs *before* shutting down the previous engine so that
        an incompatible checkpoint does not leave the server without a model.
        """
        if name not in self._available:
            raise ValueError(f"Unknown checkpoint: {name!r}. Available: {self._available}")

        checkpoint_path = self._base_dir / name
        validate_checkpoint(checkpoint_path)

        self.shutdown()

        self._engine = InferenceEngine(
            checkpoint_path,
            deterministic=True,
            deterministic_seed=0,
            airport=self._airport,
        )
        self._active_name = name

    def shutdown(self) -> None:
        """Shut down the active engine, if any."""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
            self._active_name = None


class MultiCheckpointManager:
    """Wraps per-airport `CheckpointManager` instances.

    Creates ``<base_dir>/<airport>/`` subdirectories and maintains one
    `CheckpointManager` per airport.
    """

    def __init__(
        self,
        base_dir: str | Path,
        airports: list[str],
        max_checkpoints: int,
    ) -> None:
        self._base_dir = Path(base_dir).absolute()
        self._managers: dict[str, CheckpointManager] = {}

        for airport in airports:
            key = airport.upper()
            airport_dir = self._base_dir / key.lower()
            airport_dir.mkdir(parents=True, exist_ok=True)
            self._managers[key] = CheckpointManager(airport_dir, max_checkpoints, airport=key)

    def get(self, airport: str) -> CheckpointManager:
        """Return the `CheckpointManager` for *airport* (case-insensitive)."""
        key = airport.upper()
        if key not in self._managers:
            raise ValueError(f"Unknown airport: {airport!r}. Known: {list(self._managers)}")
        return self._managers[key]

    @property
    def airports(self) -> list[str]:
        """Return the list of airport codes (upper-case)."""
        return list(self._managers)

    def discover_all(self) -> dict[str, list[str]]:
        """Run `CheckpointManager.discover` for every airport."""
        return {airport: mgr.discover() for airport, mgr in self._managers.items()}

    def shutdown_all(self) -> None:
        """Shut down engines for all airports."""
        for mgr in self._managers.values():
            mgr.shutdown()
