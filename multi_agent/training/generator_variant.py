"""Allowlisted scenario-generator variants, derived from frozen MXP input bytes.

A variant keeps the frozen reservation stratum (``scenario.config_sha256`` stays the parent hash)
and changes exactly one generator setting in a private snapshot of the parent configuration.
"""

import tempfile
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from common.hashing import content_digest

NO_RANDOM_EXTRA_TTL = "mxp-no-random-extra-ttl"
GENERATOR_VARIANTS = ("", NO_RANDOM_EXTRA_TTL)
_CONFIG = "scenario_generator/config.toml"
_BEFORE = b"aman_use_random_extra_ttl = true"
_AFTER = b"aman_use_random_extra_ttl = false"


def parent_config_sha256() -> str:
    from training.use_case import use_case

    return use_case("mxp").scenario_fields()["config_sha256"]


def variant_files() -> dict[str, bytes]:
    from training.run_config import scenario_config_digest
    from training.use_case import use_case

    root = use_case("mxp").config_root
    if scenario_config_digest(root) != parent_config_sha256():
        raise ValueError("generator variant parent differs from frozen MXP configuration")
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    if files[_CONFIG].count(_BEFORE) != 1:
        raise ValueError("generator variant requires exactly one original random TTL setting")
    files[_CONFIG] = files[_CONFIG].replace(_BEFORE, _AFTER)
    return files


def variant_config_sha256() -> str:
    """The variant's scenario_config_digest, computed from the derived bytes."""
    return content_digest(
        {name: sha256(data).hexdigest() for name, data in variant_files().items()}
    )


@lru_cache(maxsize=1)
def _snapshot():
    # Keep the directory owner alive until interpreter exit; each Ray process owns its copy.
    owner = tempfile.TemporaryDirectory(prefix="tada-mxp-no-random-extra-ttl-")
    root = Path(owner.name)
    for relative, data in variant_files().items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return owner, root


def snapshot_root(name: str) -> Path:
    """The authenticated variant snapshot; rejects unknown names and changed snapshot bytes."""
    from training.run_config import scenario_config_digest

    if name != NO_RANDOM_EXTRA_TTL:
        raise ValueError(f"unsupported scenario generator variant {name!r}")
    _, root = _snapshot()
    if scenario_config_digest(root) != variant_config_sha256():
        raise ValueError("scenario generator variant content mismatch")
    return root


def scenario_config_root(scenario) -> Path:
    """Authenticate a resolved scenario's configuration and return the root the simulator loads."""
    from training.run_config import scenario_config_digest

    root = Path(scenario["config_root"])
    if scenario_config_digest(root) != scenario["config_sha256"]:
        raise ValueError("scenario configuration content digest mismatch")
    name = scenario.get("generator_variant")
    if not name:
        return root
    if scenario["config_sha256"] != parent_config_sha256():
        raise ValueError("scenario generator variant requires the frozen MXP configuration")
    return snapshot_root(name)
