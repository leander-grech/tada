"""One explicit base plus a partial patch, then the ordinary strict run loader."""

from pathlib import Path

from common.jsonio import read_json


def _paths(data, origin):
    for key in ("reserved_manifest", "reserved_conditions"):
        if key in data and isinstance(data[key], str) and data[key]:
            data[key] = str((origin / data[key]).resolve())
    for section, names in (
        ("scenario", ("config_root",)),
        ("execution", ("output_root", "development_manifest")),
    ):
        if isinstance(data.get(section), dict):
            for name in names:
                if (
                    name in data[section]
                    and isinstance(data[section][name], str)
                    and data[section][name]
                ):
                    data[section][name] = str((origin / data[section][name]).resolve())


def _merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def read_run_input(path: Path) -> tuple[dict, Path]:
    """Preserve each explicit path's origin; prohibit chains and ambiguous envelopes."""
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("run input must be an object")
    if not {"base", "patch"} & data.keys():
        return data, path
    if (
        set(data) != {"base", "patch"}
        or not isinstance(data["base"], str)
        or not data["base"]
        or not isinstance(data["patch"], dict)
    ):
        raise ValueError("overlay requires only a base path and object patch")
    base_path = (path.parent / data["base"]).resolve()
    if base_path == path.resolve():
        raise ValueError("overlay cycle")
    base = read_json(base_path)
    if not isinstance(base, dict) or {"base", "patch"} & base.keys():
        raise ValueError("overlay base must be a complete run, not another overlay")
    if {"base", "patch"} & data["patch"].keys():
        raise ValueError("overlay patch cannot introduce inheritance")
    _paths(base, base_path.parent)
    _paths(data["patch"], path.parent)
    _merge(base, data["patch"])
    return base, base_path
