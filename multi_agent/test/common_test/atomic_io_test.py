import json
import os

import pytest

from common import atomic_io
from common.atomic_io import atomic_json, atomic_write_bytes, atomic_write_text, fsync_path
from common.hashing import canonical_json


def test_atomic_write_text_creates_parents_and_replaces(tmp_path):
    path = tmp_path / "a" / "b.txt"
    atomic_write_text(path, "one")
    atomic_write_text(path, "two")
    assert path.read_text() == "two"
    assert [p.name for p in path.parent.iterdir()] == ["b.txt"]


def test_atomic_json_is_canonical_with_newline(tmp_path):
    path = tmp_path / "v.json"
    atomic_json(path, {"b": 1, "a": [2]})
    assert path.read_text() == canonical_json({"b": 1, "a": [2]}) + "\n"
    assert json.loads(path.read_text()) == {"a": [2], "b": 1}


def test_exclusive_publish_never_overwrites_and_leaves_no_temp(tmp_path):
    path = tmp_path / "once.json"
    atomic_write_bytes(path, b"first", exclusive=True)
    with pytest.raises(FileExistsError):
        atomic_write_bytes(path, b"second", exclusive=True)
    assert path.read_bytes() == b"first"
    assert [p.name for p in tmp_path.iterdir()] == ["once.json"]


def test_failed_write_keeps_previous_content_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "keep.txt"
    atomic_write_text(path, "good")

    def boom(_descriptor):
        raise OSError("disk")

    monkeypatch.setattr(atomic_io.os, "fsync", boom)
    with pytest.raises(OSError, match="disk"):
        atomic_write_text(path, "bad")
    assert path.read_text() == "good"
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_content_is_fsynced_before_publication(tmp_path, monkeypatch):
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(atomic_io.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd)))
    monkeypatch.setattr(
        atomic_io.os, "replace", lambda a, b: (order.append("replace"), real_replace(a, b))
    )
    atomic_write_text(tmp_path / "x", "v")
    assert order[:2] == ["fsync", "replace"] and order.count("fsync") == 2


def test_fsync_path_accepts_files_and_directories(tmp_path):
    (tmp_path / "f").write_text("x")
    fsync_path(tmp_path / "f")
    fsync_path(tmp_path)
