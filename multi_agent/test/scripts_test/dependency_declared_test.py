import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIPPED_DIRS = {"test", "runs", "ray_results", ".venv", "docs", "node_modules", "TOOLING"}
IMPORT_TO_DIST = {
    "dotenv": "python-dotenv",
    "multipart": "python-multipart",
    "torch_geometric": "torch-geometric",
    "tree": "dm-tree",
    "yaml": "pyyaml",
}


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _declared() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    specs = project["dependencies"] + sum(project["optional-dependencies"].values(), [])
    return {_normalize(re.split(r"[\[=~<>;! ]", spec, maxsplit=1)[0]) for spec in specs}


def _imported_top_levels(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return {name.split(".")[0] for name in names}


def test_third_party_imports_in_runtime_code_are_declared():
    local = {p.name for p in ROOT.iterdir()} | {p.stem for p in ROOT.glob("*.py")}
    imported = set()
    for path in ROOT.rglob("*.py"):
        if SKIPPED_DIRS.isdisjoint(path.relative_to(ROOT).parts):
            imported |= _imported_top_levels(path)
    third_party = imported - set(sys.stdlib_module_names) - local
    missing = {_normalize(IMPORT_TO_DIST.get(n, n)) for n in third_party} - _declared()
    assert not missing, sorted(missing)
