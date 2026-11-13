"""Incomplete and corrupt publications never become selectable checkpoints."""

import pytest

from common.jsonio import read_json
from training import checkpoint_bundle as bundle
from training.checkpoint_manifest import MANIFEST, verify_files


def writer(staging):
    (staging / "state.pkl").write_bytes(b"binary state")
    return {"format": 1}


def verify(root):
    verify_files(root, read_json(root / MANIFEST)["files"])


def test_manifest_is_written_last_with_every_file_digest(tmp_path):
    destination = bundle.publish_bundle(tmp_path / "one", writer)
    assert {p.name for p in destination.iterdir()} == {MANIFEST, "state.pkl"}
    assert set(read_json(destination / MANIFEST)["files"]) == {"state.pkl"}
    verify(destination)
    (destination / "state.pkl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch: state.pkl"):
        verify(destination)


@pytest.mark.parametrize("payload", ["manifest", "no-body"])
def test_writer_cannot_supply_the_manifest(tmp_path, payload):
    def bad(staging):
        if payload == "manifest":
            (staging / MANIFEST).write_text("{}")
            return {"format": 1}
        (staging / "state.pkl").write_bytes(b"x")

    with pytest.raises(ValueError, match="writer payload"):
        bundle.publish_bundle(tmp_path / "one", bad)
    assert not (tmp_path / "one").exists()


def test_failed_publication_keeps_completed_latest(tmp_path):
    first = bundle.publish_bundle(tmp_path / "first", writer)
    latest = (tmp_path / "latest.json").read_bytes()

    def failed(staging):
        writer(staging)
        raise OSError("injected write failure")

    with pytest.raises(OSError, match="injected"):
        bundle.publish_bundle(tmp_path / "second", failed)
    assert (tmp_path / "latest.json").read_bytes() == latest
    verify(first)
    assert not (tmp_path / "second").exists()


@pytest.mark.parametrize("damage", ["unknown", "missing"])
def test_reader_rejects_unknown_and_missing_files(tmp_path, damage):
    destination = bundle.publish_bundle(tmp_path / "one", writer)
    if damage == "unknown":
        (destination / "unknown").write_text("extra")
    else:
        (destination / "state.pkl").unlink()
    with pytest.raises(ValueError, match="missing or unknown"):
        verify(destination)


def test_failed_atomic_rename_preserves_prior_latest(tmp_path, monkeypatch):
    bundle.publish_bundle(tmp_path / "first", writer)
    before = (tmp_path / "latest.json").read_bytes()
    original = bundle.os.replace

    def fail(source, destination):
        if destination == tmp_path / "second":
            raise OSError("injected publication failure")
        return original(source, destination)

    monkeypatch.setattr(bundle.os, "replace", fail)
    with pytest.raises(OSError, match="publication"):
        bundle.publish_bundle(tmp_path / "second", writer)
    assert (tmp_path / "latest.json").read_bytes() == before
    verify(tmp_path / "first")
    assert not (tmp_path / "second").exists()


def test_abrupt_writer_exit_never_publishes_staging(tmp_path):
    import subprocess
    import sys

    bundle.publish_bundle(tmp_path / "first", writer)
    before = (tmp_path / "latest.json").read_bytes()
    script = r"""
import os
import sys
from pathlib import Path
from training.checkpoint_bundle import publish_bundle

def writer(path):
    (path / "state.pkl").write_bytes(b"partial")
    os._exit(23)

publish_bundle(Path(sys.argv[1]), writer)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "interrupted")], check=False, timeout=30
    )
    assert result.returncode == 23
    assert (tmp_path / "latest.json").read_bytes() == before
    assert not (tmp_path / "interrupted").exists()
    staging = list(tmp_path.glob(".interrupted.staging-*"))
    assert len(staging) == 1
    assert not (staging[0] / MANIFEST).exists()
