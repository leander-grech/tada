"""Durable immutable directory publication, independent of learner serialization."""

import os
import shutil
import tempfile
from pathlib import Path

from common.atomic_io import atomic_json, atomic_write_text, fsync_path
from common.hashing import canonical_json, file_sha256
from training.checkpoint_boundary import CheckpointBoundaryError
from training.checkpoint_manifest import MANIFEST


def _files(root):
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CheckpointBoundaryError("checkpoint symlinks are unsupported")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
        elif not path.is_dir():
            raise CheckpointBoundaryError("unsupported checkpoint file")
    return result


def publish_bundle(destination: Path, writer, *, advertise_latest=True) -> Path:
    """Publish only complete siblings; never remove an older selected bundle.

    ``writer(staging)`` writes the payload files and returns the manifest body;
    the manifest, with every file digest, is added last.
    """
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise CheckpointBoundaryError("checkpoint destination must be new or empty")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        body = writer(staging)
        files = _files(staging)
        if MANIFEST in files or not isinstance(body, dict) or "files" in body:
            raise CheckpointBoundaryError("invalid checkpoint writer payload")
        for path in files.values():
            fsync_path(path)
        digests = {name: file_sha256(path) for name, path in sorted(files.items())}
        # Written last: a manifest names only files that are already durable.
        atomic_write_text(staging / MANIFEST, canonical_json({**body, "files": digests}) + "\n")
        for directory in sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True):
            fsync_path(directory)
        fsync_path(staging)
        os.replace(staging, destination)
        fsync_path(destination.parent)
        if advertise_latest:
            atomic_json(
                destination.parent / "latest.json",
                {"version": 1, "checkpoint": destination.name},
            )
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
