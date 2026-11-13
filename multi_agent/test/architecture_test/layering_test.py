"""Enforce the layered import rule over every import, including function-level (lazy) ones."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CORE = {"common", "config", "geometry", "simulator", "outcomes", "actions", "models"}
ALLOWED = {
    "common": set(),
    "config": {"common"},
    "geometry": {"common"},
    "simulator": {"common", "config", "geometry"},
    "outcomes": {"common", "config", "geometry", "simulator"},
    "actions": {"common", "config", "geometry", "simulator", "outcomes"},
    "models": {"common", "config", "geometry", "simulator", "outcomes", "actions"},
    "atc_env": _CORE - {"atc_env"},
    "network": _CORE,
    "callbacks": _CORE | {"atc_env"},
    "training": _CORE | {"atc_env", "network", "callbacks"},
    "evaluation": _CORE | {"atc_env", "network", "training", "callbacks"},
    "inference": _CORE | {"atc_env", "network", "training", "evaluation"},
    "scripts": _CORE | {"atc_env", "network", "training", "evaluation", "callbacks", "inference"},
}
# Never ranked: main.py (composition root), test/, runs/. utils/ must not exist.
PACKAGES = set(ALLOWED)
CORE_WITHOUT_CONFIG = {
    "actions",
    "models",
    "outcomes",
    "simulator",
    "atc_env",
    "network",
    "geometry",
}
CONFIG_RESOLUTION_POINT = "atc_env/env_config.py"


def _is_test(rel: Path) -> bool:
    return "test" in rel.parts or rel.name.endswith("_test.py") or rel.name.startswith("test_")


def edges():
    """Yield (src_pkg, dst_pkg, relpath, lineno, module) for every project import, lazy included."""
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        pkg = rel.parts[0]
        if len(rel.parts) < 2 or pkg not in PACKAGES or _is_test(rel):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                dst = name.split(".")[0]
                if dst in PACKAGES and dst != pkg:
                    yield pkg, dst, rel.as_posix(), node.lineno, name


def violations():
    found = {}
    for src, dst, rel, lineno, name in edges():
        if dst not in ALLOWED.get(src, set()):
            found.setdefault((src, dst), []).append(f"{rel}:{lineno} imports {name}")
    return found


def config_reads():
    return {
        rel
        for src, _dst, rel, _lineno, name in edges()
        if src in CORE_WITHOUT_CONFIG and name == "config.config" and rel != CONFIG_RESOLUTION_POINT
    }


def test_edges_follow_layer_table():
    found = violations()
    assert not found, f"layering violations: {found}"


def test_core_packages_do_not_read_global_config():
    found = config_reads()
    assert not found, f"global Config read outside {CONFIG_RESOLUTION_POINT}: {found}"
