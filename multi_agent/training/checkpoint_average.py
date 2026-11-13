"""Offline weights-only bundles: uniform averages or fine-tuned module tensors.

A published bundle keeps everything but the module's floating-point tensors
(value-normalizer statistics, optimizer slots, boundary, counters, manifest
semantics) from its source and is republished through ``publish_bundle``,
so every reader applies its usual integrity and payload checks. The optimizer
moments then describe the source, not the new weights: use the result for
weights-only inference, not as a training resume parent.
"""

import pickle
import shutil
from pathlib import Path

import numpy as np

from common.jsonio import read_json
from training.checkpoint_boundary import CheckpointBoundaryError
from training.checkpoint_bundle import publish_bundle
from training.checkpoint_manifest import MANIFEST
from training.checkpoints import inspect_checkpoint, read_checkpoint_payload


def _modules(state):
    return state["learner_group"]["learner"]["rl_module"]


def publish_weights(source, destination: Path, tensor) -> Path:
    """Publish ``destination``: ``source`` with each float module tensor replaced by
    ``tensor(module_id, name)``, which must keep its shape."""
    source = Path(source)
    state = read_checkpoint_payload(source)[0]
    for module_id, values in _modules(state).items():
        for name, value in values.items():
            if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                replacement = np.asarray(tensor(module_id, name), dtype=value.dtype)
                if replacement.shape != value.shape:
                    raise CheckpointBoundaryError(f"replacement changes the shape of {name}")
                values[name] = replacement

    def write(staging):
        (staging / "algorithm.pkl").write_bytes(pickle.dumps(state, protocol=5))
        for file in source.iterdir():
            if file.name not in {MANIFEST, "algorithm.pkl"}:
                shutil.copyfile(file, staging / file.name)
        return {k: v for k, v in read_json(source / MANIFEST).items() if k != "files"}

    published = publish_bundle(Path(destination), write, advertise_latest=False)
    inspect_checkpoint(published, scope="weights")
    return published


def average_checkpoints(sources, destination: Path) -> Path:
    """Publish ``destination`` whose float module tensors are the uniform mean of ``sources``."""
    sources = [Path(source) for source in sources]
    if len(sources) < 2:
        raise CheckpointBoundaryError("averaging requires at least two checkpoints")
    loaded = [inspect_checkpoint(source, scope="weights") for source in sources]
    if any(
        item.manifest["module_layout"] != loaded[-1].manifest["module_layout"] for item in loaded
    ):
        raise CheckpointBoundaryError("averaged checkpoints must share one module layout")
    states = [_modules(read_checkpoint_payload(source)[0]) for source in sources]
    return publish_weights(
        sources[-1],
        destination,
        lambda module_id, name: np.stack([item[module_id][name] for item in states]).mean(
            axis=0, dtype=np.float64
        ),
    )
