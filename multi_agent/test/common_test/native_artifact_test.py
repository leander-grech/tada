"""Retained native identity must fail closed on artifact or installed-byte changes."""

import json
import sys
import types
import zipfile

import pytest

from common import native_runtime
from common.hashing import file_sha256


def _native_fixture(tmp_path, monkeypatch):
    environment = tmp_path / "environment"
    environment.mkdir()
    extension = environment / "flight_simulator.so"
    extension.write_bytes(b"native-extension-fixture")
    wheel = tmp_path / "simulator.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(extension.name, extension.read_bytes())
        archive.writestr(
            "flight_simulator-1.dist-info/METADATA",
            "Name: flight-simulator\nVersion: 1\nRequires-Python: >=3.12\n",
        )
        archive.writestr("flight_simulator-1.dist-info/WHEEL", "Generator: fixture\n")
    manifest = {
        "mode": "external",
        "supplied_artifact": str(wheel),
        "supplied_sha256": file_sha256(wheel),
        "wheel": str(wheel),
        "wheel_sha256": file_sha256(wheel),
    }
    (tmp_path / "native.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(native_runtime, "STATE", tmp_path)
    monkeypatch.setattr(sys, "prefix", str(environment))
    monkeypatch.setattr(native_runtime.importlib.metadata, "version", lambda name: "1")
    monkeypatch.setitem(
        sys.modules, "flight_simulator", types.SimpleNamespace(__file__=str(extension))
    )
    return wheel, extension


def test_installed_native_bytes_must_match_retained_wheel(tmp_path, monkeypatch):
    _, extension = _native_fixture(tmp_path, monkeypatch)
    native_runtime.verify_native()
    extension.write_bytes(b"substituted-extension")
    with pytest.raises(RuntimeError, match="extension bytes do not match"):
        native_runtime.verify_native()


def test_retained_wheel_changes_are_rejected_before_import(tmp_path, monkeypatch):
    wheel, _ = _native_fixture(tmp_path, monkeypatch)
    wheel.write_bytes(b"substituted-wheel")
    with pytest.raises(RuntimeError, match="artifact checksum changed"):
        native_runtime.verify_native()


def test_extension_outside_environment_is_rejected(tmp_path, monkeypatch):
    _native_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "another-environment"))
    with pytest.raises(RuntimeError, match="outside selected environment"):
        native_runtime.verify_native()
