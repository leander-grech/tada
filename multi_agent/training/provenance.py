"""Content provenance and append-only, per-stream scenario consumption journals."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path

from common.hashing import bytes_sha256, canonical_json, file_sha256
from common.jsonio import read_json
from training.generator_variant import snapshot_root
from training.run_config import ResolvedRunConfig, scenario_config_digest


def _runtime_identity(runtime: dict) -> dict:
    """Read actual artifacts; never infer installed native version from uv.lock."""
    native_path = Path(
        runtime.get(
            "native_manifest_path", Path(__file__).resolve().parents[1] / "runs/runtime/native.json"
        )
    )
    native_document = read_json(native_path)
    if "native_manifest_path" in runtime:
        # Explicit imported evidence is not a
        # verification of the currently imported native extension.
        native = {
            key: native_document.get(key)
            for key in (
                "mode",
                "version",
                "source_identity",
                "source_provenance",
                "wheel_sha256",
                "extension_sha256",
            )
        }
        if native.get("version") is None:
            native["version"] = native_document.get("package_metadata", {}).get("version")
        native_verification = "provided-evidence-not-runtime-verified"
    else:
        from common.native_runtime import verify_native

        native = verify_native()
        native_verification = "installed-extension-matches-retained-wheel"
    packages = {}
    for name in ("flight-simulator", "torch", "ray", "numpy", "torch-geometric", "gymnasium"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch
    import torch.version

    gpus = []
    for index in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": device.name,
                "memory_bytes": device.total_memory,
                "capability": [device.major, device.minor],
            }
        )
    return {
        "native": native,
        "native_manifest_sha256": file_sha256(native_path),
        "native_verification": native_verification,
        "dependencies": packages,
        "backend": "cuda" if torch.version.cuda is not None else "cpu",
        "cuda_build": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        # Available source identity is not proof that a caller selected these
        # classes. Trainer/restore must separately verify actual configuration.
        "owned_learning": {"selection_verified": False},
        "hardware": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "visible_gpus": gpus,
        },
        "python": platform.python_version(),
    }


def write_run_manifest(
    path: Path, config: ResolvedRunConfig, runtime: dict, *, algorithm=None, execution_segment=None
) -> None:
    """Write canonical resolved config plus provenance, native, version and seed evidence.

    An actual Algorithm is inspected when provided; user-supplied proof flags
    are ignored. Provided native evidence cannot attest a live Algorithm.
    """
    from atc_env.sector_catalog import capture_sector_catalog_identity

    root = Path(__file__).resolve().parents[1]
    scenario_root = Path(config.scenario["config_root"])
    identity = _runtime_identity(runtime)
    if algorithm is not None:
        from training.runtime_gate import verify_algorithm_selection, verify_native_conditions

        if "native_manifest_path" in runtime:
            raise ValueError("provided native evidence cannot attest a live algorithm")
        verify_native_conditions(config)
        identity["owned_learning"] = verify_algorithm_selection(algorithm, config)
    manifest = {
        "version": 1,
        "execution_segment": execution_segment,
        "effective_inputs": {"sector_catalog": capture_sector_catalog_identity()},
        "config": config.to_dict(),
        "provenance": {"wheel_sha256": identity["native"]["wheel_sha256"]},
        "runtime": identity,
        "preprocessing": {"features": config.features, "versions": config.versions},
        "reservation_sha256": config.reserved_manifest["manifest_sha256"],
        "reservation_conditions_sha256": config.reserved_conditions["conditions_sha256"],
        "dependency_lock_sha256": file_sha256(root / "uv.lock")
        if (root / "uv.lock").is_file()
        else None,
        "scenario_config": {
            "declared_sha256": config.scenario["config_sha256"],
            "observed_sha256": scenario_config_digest(scenario_root),
        },
    }
    if variant := config.scenario.get("generator_variant"):
        manifest["scenario_config"]["generator_variant"] = {
            "name": variant,
            "loaded_sha256": scenario_config_digest(snapshot_root(variant)),
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest) + "\n")


def append_scenario_event(path: Path, event: dict) -> None:
    """Append a complete record; callers use one artifact path per stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(event) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("incomplete scenario journal record")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checkpoint_method_provenance(checkpoint):
    """Authenticate the model bundle and record the actual reader runtime."""
    from training.checkpoint_manifest import checkpoint_digest
    from training.checkpoints import inspect_checkpoint

    checkpoint = Path(checkpoint).resolve()
    inspected = inspect_checkpoint(checkpoint, scope="weights")
    run = inspected.run
    payload = (
        "weights.npz"
        if inspected.manifest.get("kind") == "inference_weights_v1"
        else "algorithm.pkl"
    )
    runtime = _runtime_identity({})
    return {
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "model_sha256": file_sha256(checkpoint / payload),
        "training_config": run.to_dict(),
        "model_family": "multi-agent-shared-recurrent-PPO",
        "reader_provenance": {"wheel_sha256": runtime["native"]["wheel_sha256"]},
        "reader_runtime": runtime,
        "adapter": "authenticated_weights_only_v1",
    }


def audit_consumed_training_seeds(paths, reservation):
    """Read actual consumption receipts; configured streams alone are not readiness."""
    from training.reservations import effective_reserved_seeds

    reserved = effective_reserved_seeds(reservation)
    consumed = []
    identities = {}
    for value in paths:
        path = Path(value)
        data = path.read_bytes()
        identities[str(path)] = bytes_sha256(data)
        for line in data.decode().splitlines():
            row = json.loads(line)
            if row["status"] == "consumed":
                if type(row["seed"]) is not int or row["seed"] in reserved:
                    raise ValueError("consumed training seed intersects benchmark reservations")
                consumed.append(row["seed"])
    return {
        "status": "observed" if consumed else "unobserved",
        "consumed_count": len(consumed),
        "distinct_seeds": sorted(set(consumed)),
        "journal_sha256": identities,
        "reserved_count": len(reserved),
    }
