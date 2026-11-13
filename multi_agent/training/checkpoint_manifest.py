"""Checkpoint manifest: the one document that makes a bundle self-describing and verifiable."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from common.hashing import canonical_json, file_sha256
from training import versions
from training.checkpoint_boundary import CheckpointBoundaryError

FORMAT = 1
MANIFEST = "manifest.json"
EXACT = ("runtime", "native", "catalog")
_NATIVE = (
    "mode",
    "version",
    "source_identity",
    "source_provenance",
    "wheel_sha256",
    "extension_sha256",
)


def checkpoint_digest(path: Path) -> str:
    """Identity of a bundle: the manifest embeds every file digest."""
    return file_sha256(Path(path) / MANIFEST)


def current_identity(run) -> dict[str, Any]:
    """Runtime, native artifact and catalog identity a resume must reproduce exactly."""
    from atc_env.sector_catalog import capture_sector_catalog_identity
    from training.provenance import _runtime_identity
    from training.resource_profile import resolve_resource_profile, runtime_inventory
    from training.runtime_gate import verify_hash_policy, verify_native_conditions

    verify_native_conditions(run)
    hash_policy = verify_hash_policy()
    runtime = _runtime_identity({})
    return {
        "runtime": {
            **{key: runtime[key] for key in ("dependencies", "backend", "cuda_build", "python")},
            "python_hash_policy": hash_policy,
            "resource_profile": asdict(resolve_resource_profile(run, runtime_inventory())),
            "rollout_backend": run.resources.get("rollout_backend", "scalar"),
        },
        "native": {key: runtime["native"][key] for key in _NATIVE},
        "catalog": portable_catalog(capture_sector_catalog_identity()),
    }


def portable_catalog(identity) -> dict[str, Any]:
    return {key: identity[key] for key in ("sectors_sha256", "source_csv_sha256")}


def build_manifest(
    run, *, provenance, lineage, execution_segment, module_layout, identity
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "versions": versions.current(),
        "run_config": run.to_dict(),
        "provenance": provenance,
        **identity,
        "execution_segment": execution_segment,
        "module_layout": module_layout,
        "lineage": lineage,
    }


def require_tokens(manifest: dict[str, Any]) -> None:
    """Weights-only readers need the same semantics, not the same runtime."""
    if manifest.get("format") != FORMAT:
        raise CheckpointBoundaryError(
            f"unsupported checkpoint manifest format {manifest.get('format')!r}"
        )
    versions.require_same(manifest["versions"], versions.current())


def _fixed(section: str, value: Any, movable: tuple[str, ...]) -> Any:
    if section != "runtime" or not movable:
        return value
    profile = {k: v for k, v in value["resource_profile"].items() if k not in movable}
    return {**value, "resource_profile": profile}


def require_compatible(
    manifest: dict[str, Any], identity: dict[str, Any], *, movable: tuple[str, ...] = ()
) -> None:
    """Exact identity, except the named resource-profile keys a phase may move."""
    require_tokens(manifest)
    for key in EXACT:
        if canonical_json(_fixed(key, manifest[key], movable)) != canonical_json(
            _fixed(key, identity[key], movable)
        ):
            raise CheckpointBoundaryError(f"checkpoint {key} identity mismatch")


def verify_files(root: Path, files: dict[str, str]) -> None:
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {MANIFEST}
    if present != set(files):
        raise CheckpointBoundaryError("missing or unknown checkpoint files")
    for name, digest in files.items():
        if file_sha256(root / name) != digest:
            raise CheckpointBoundaryError(f"checkpoint file checksum mismatch: {name}")
