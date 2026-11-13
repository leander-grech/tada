import fnmatch
import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_every_console_script_target_imports():
    for name, target in PYPROJECT["project"]["scripts"].items():
        module, _, attr = target.partition(":")
        assert hasattr(importlib.import_module(module), attr), name


def test_every_source_package_is_discovered():
    find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
    sources = {p.name for p in ROOT.iterdir() if (p / "__init__.py").exists()} - {"test"}
    excluded = {d for d in sources if any(fnmatch.fnmatch(d, x) for x in find["exclude"])}
    assert find["where"] == ["."]
    assert not excluded
