"""Portable numeric model publication, explicitly unsupported for training resume.

The exported contract describes inference semantics and compatibility, never the
training host. Complete private checkpoint bundles retain their original format.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from training.checkpoint_boundary import CheckpointBoundaryError
from training.checkpoint_bundle import publish_bundle
from training.checkpoint_manifest import MANIFEST, require_tokens, verify_files
from training.checkpoint_preflight import module_payload_layout, validate_current_model_payload
from training.phase import saved_run
from training.run_config import (
    reseal,
    scenario_config_digest,
    validate_reservation_manifest,
    validate_reserved_conditions,
)
from training.use_case import USE_CASES

KIND = "inference_weights_v1"
_FIELDS = {
    "format", "kind", "training_resume", "versions", "run_config", "native",
    "catalog", "module_layout", "files",
}


def resolve_scenario(run):
    """Bind authenticated scenario bytes to an included frozen snapshot on this host."""
    for case in USE_CASES.values():
        if scenario_config_digest(case.config_root) == run.scenario["config_sha256"]:
            return replace(run, scenario={**run.scenario, "config_root": str(case.config_root)})
    raise CheckpointBoundaryError("no included scenario snapshot matches checkpoint hash")


def _state(modules):
    return {"learner_group": {"learner": {"rl_module": modules}}}


def _arrays(state):
    """Encode only numeric leaves; descriptors preserve Python scalars versus arrays."""
    layout = module_payload_layout(state)
    arrays = {}
    for module, values in state["learner_group"]["learner"]["rl_module"].items():
        for name, value in values.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            arrays[f"{module}/{name}"] = np.asarray(value)
    return layout, arrays


def _read_weights(checkpoint, manifest):
    modules = {}
    try:
        with np.load(Path(checkpoint) / "weights.npz", allow_pickle=False) as archive:
            expected = {
                f"{module}/{name}"
                for module, values in manifest["module_layout"].items()
                for name in values
            }
            if set(archive.files) != expected or len(archive.files) != len(expected):
                raise CheckpointBoundaryError("missing or unknown numeric model components")
            for module, values in manifest["module_layout"].items():
                modules[module] = {}
                for name, descriptor in values.items():
                    array = archive[f"{module}/{name}"]
                    if array.dtype.kind not in "bifu" or not np.isfinite(array).all():
                        raise CheckpointBoundaryError("invalid numeric model component")
                    if "scalar" in descriptor:
                        kind = descriptor["scalar"]
                        if descriptor != {"scalar": kind} or kind not in {"int", "float"}:
                            raise CheckpointBoundaryError("invalid scalar descriptor")
                        expected_dtype = np.asarray(0 if kind == "int" else 0.0).dtype
                        if array.shape != () or array.dtype != expected_dtype:
                            raise CheckpointBoundaryError("invalid numeric scalar representation")
                        value = int(array) if kind == "int" else float(array)
                    else:
                        if descriptor != {"shape": list(array.shape), "dtype": str(array.dtype)}:
                            raise CheckpointBoundaryError("incompatible numeric tensor shape/dtype")
                        value = array.copy()
                    modules[module][name] = value
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CheckpointBoundaryError("unreadable numeric inference weights") from error
    state = _state(modules)
    if module_payload_layout(state) != manifest["module_layout"]:
        raise CheckpointBoundaryError("inference model layout mismatch")
    return state


def inspect_inference_bundle(checkpoint, manifest):
    """Validate integrity, semantic sealing, reservations and current architecture."""
    checkpoint = Path(checkpoint)
    if set(manifest) != _FIELDS or manifest["training_resume"] is not False:
        raise CheckpointBoundaryError("invalid inference-only manifest")
    if {p.name for p in checkpoint.iterdir()} != {MANIFEST, "weights.npz"} or any(
        p.is_symlink() for p in checkpoint.iterdir()
    ):
        raise CheckpointBoundaryError("unknown or missing inference bundle files")
    require_tokens(manifest)
    verify_files(checkpoint, manifest["files"])
    run = saved_run(manifest["run_config"])
    validate_reservation_manifest(run.reserved_manifest)
    validate_reserved_conditions(run.reserved_conditions, run.reserved_manifest)
    if manifest["native"] != run.reserved_conditions["native_artifact"]:
        raise CheckpointBoundaryError("inference native compatibility mismatch")
    state = _read_weights(checkpoint, manifest)
    validate_current_model_payload(state, resolve_scenario(run))
    return state


def read_inference_payload(checkpoint, manifest):
    return inspect_inference_bundle(checkpoint, manifest), None


def export_inference_bundle(source, destination):
    """Export a trusted complete checkpoint into a fresh numeric inference bundle.

    Only this explicit export creates a new semantic seal. No original checkpoint
    manifest, resume state, reservation or model value is rewritten.
    """
    from training.checkpoints import inspect_checkpoint, read_checkpoint_payload

    loaded = inspect_checkpoint(source, scope="weights")
    state, _ = read_checkpoint_payload(source)
    run = loaded.run
    resources = {k: v for k, v in run.resources.items() if not k.endswith("gpu_uuid")}
    run = reseal(run, resources=resources)
    run = replace(
        run, run_id="published-model", phase_id="inference",
        execution={"output_root": "inference-results", "stop_iterations": None},
        scenario={**run.scenario, "config_root": "included-snapshot"},
    )
    layout, arrays = _arrays(state)

    def write(staging):
        np.savez(staging / "weights.npz", **arrays)
        return {
            "format": 1,
            "kind": KIND,
            "training_resume": False,
            "versions": loaded.manifest["versions"],
            "run_config": run.to_dict(),
            "native": loaded.manifest["native"],
            "catalog": loaded.manifest["catalog"],
            "module_layout": layout,
        }

    return publish_bundle(Path(destination), write, advertise_latest=False)
