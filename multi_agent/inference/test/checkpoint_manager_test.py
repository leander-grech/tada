"""Unit tests for inference.checkpoint_manager.MultiCheckpointManager."""

import io
import zipfile
from pathlib import Path

import pytest

import inference.checkpoint_manager as checkpoint_manager_module
from inference.checkpoint_manager import (
    RLLIB_MARKER,
    CheckpointManager,
    MultiCheckpointManager,
    checkpoint_name_from_upload,
    extract_checkpoint_zip,
)


def _zip_bytes(members: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("filename", ["../evil.zip", "a/b.zip", ".hidden.zip", ".zip", "a..b.zip"])
def test_upload_name_rejects_unsafe_filenames(filename):
    with pytest.raises(ValueError, match="Invalid checkpoint name"):
        checkpoint_name_from_upload(filename)


@pytest.mark.parametrize("filename", ["run-7_v2.zip", "run-7_v2.ZIP", "run-7_v2.Zip"])
def test_upload_name_strips_zip_suffix_in_any_case(filename):
    assert checkpoint_name_from_upload(filename) == "run-7_v2"


def test_extract_rejects_member_escaping_destination(tmp_path):
    archive = _zip_bytes({"../outside.txt": b"x"})
    with pytest.raises(ValueError, match="member"):
        extract_checkpoint_zip(archive, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()


def test_extract_rejects_oversize_unpacked_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_manager_module, "MAX_CHECKPOINT_UNPACKED_SIZE", 4)
    with pytest.raises(ValueError, match="unpacked"):
        extract_checkpoint_zip(_zip_bytes({"big.bin": b"12345"}), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_extract_rejects_archive_with_too_many_members(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_manager_module, "MAX_CHECKPOINT_MEMBERS", 1)
    with pytest.raises(ValueError, match="members"):
        extract_checkpoint_zip(_zip_bytes({"a": b"", "b": b""}), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_extract_writes_bounded_archive(tmp_path):
    extract_checkpoint_zip(_zip_bytes({f"ckpt/{RLLIB_MARKER}": b"{}"}), tmp_path)
    assert (tmp_path / "ckpt" / RLLIB_MARKER).read_bytes() == b"{}"


def _create_fake_checkpoint(base_dir: Path, name: str) -> Path:
    """Create a minimal fake checkpoint directory with the rllib marker file."""
    ckpt_dir = base_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / RLLIB_MARKER).write_text("{}")
    return ckpt_dir


class TestInit:
    def test_creates_subdirectories_for_each_airport(self, tmp_path):
        MultiCheckpointManager(tmp_path, airports=["MXP", "BGY"], max_checkpoints=10)
        assert (tmp_path / "mxp").is_dir()
        assert (tmp_path / "bgy").is_dir()


class TestGet:
    def test_returns_checkpoint_manager_for_valid_airport(self, tmp_path):
        mgr = MultiCheckpointManager(tmp_path, airports=["MXP"], max_checkpoints=10)
        result = mgr.get("MXP")
        assert isinstance(result, CheckpointManager)

    def test_case_insensitive_lookup(self, tmp_path):
        mgr = MultiCheckpointManager(tmp_path, airports=["MXP"], max_checkpoints=10)
        assert mgr.get("mxp") is mgr.get("MXP")

    def test_raises_for_unknown_airport(self, tmp_path):
        mgr = MultiCheckpointManager(tmp_path, airports=["MXP"], max_checkpoints=10)
        with pytest.raises(ValueError, match="Unknown airport"):
            mgr.get("JFK")


class TestDiscoverAll:
    def test_discovers_checkpoints_for_all_airports(self, tmp_path):
        mgr = MultiCheckpointManager(tmp_path, airports=["MXP", "BGY"], max_checkpoints=10)
        _create_fake_checkpoint(tmp_path / "mxp", "ckpt-a")
        _create_fake_checkpoint(tmp_path / "bgy", "ckpt-b")

        result = mgr.discover_all()
        assert result == {"MXP": ["ckpt-a"], "BGY": ["ckpt-b"]}

    def test_returns_empty_lists_when_no_checkpoints(self, tmp_path):
        mgr = MultiCheckpointManager(tmp_path, airports=["MXP", "BGY"], max_checkpoints=10)
        result = mgr.discover_all()
        assert result == {"MXP": [], "BGY": []}


class TestShutdownAll:
    def test_calls_shutdown_on_each_manager(self, tmp_path, monkeypatch):
        mgr = MultiCheckpointManager(tmp_path, airports=["MXP", "BGY"], max_checkpoints=10)

        shutdown_calls = []
        for airport in mgr.airports:
            inner = mgr.get(airport)

            def _tracked_shutdown(ap=airport):
                shutdown_calls.append(ap)

            monkeypatch.setattr(inner, "shutdown", _tracked_shutdown)

        mgr.shutdown_all()
        assert set(shutdown_calls) == {"MXP", "BGY"}
