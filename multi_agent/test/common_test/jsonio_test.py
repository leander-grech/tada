import json

import pytest

from common.jsonio import read_json


def test_reads_json_document(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"k": [1]}), encoding="utf-8")
    assert read_json(path) == {"k": [1]}


def test_missing_file_raises_without_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")


def test_missing_file_returns_explicit_default_including_none(tmp_path):
    assert read_json(tmp_path / "missing.json", None) is None
    assert read_json(tmp_path / "missing.json", {}) == {}


def test_existing_file_ignores_default(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("[1]")
    assert read_json(path, None) == [1]


def test_malformed_json_still_raises_with_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        read_json(path, None)
