"""Regression checks for platform-compatible simulator wheel selection."""
import runpy
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_setup(monkeypatch, machine):
    import platform

    captured = {}
    setuptools = ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    version = namedtuple("Version", "major minor micro releaselevel serial")
    monkeypatch.setattr(sys, "version_info", version(3, 12, 0, "final", 0))
    runpy.run_path(str(ROOT / "setup.py"))
    return captured


def test_setup_selects_compatible_wheel(monkeypatch):
    setup_args = _run_setup(monkeypatch, "x86_64")
    assert any("flight_simulator-0.2.80-cp312-cp312-manylinux_2_34_x86_64.whl" in requirement
               for requirement in setup_args["install_requires"])


def test_setup_rejects_incompatible_linux_architecture(monkeypatch):
    with pytest.raises(RuntimeError, match="No compatible wheel"):
        _run_setup(monkeypatch, "aarch64")
