"""Pinned owned Tune hook: advertise only atomically published local bundles."""

import inspect
import shutil
from pathlib import Path

import ray
from pyarrow.fs import LocalFileSystem
from ray.train._internal.session import _TrainingResult
from ray.tune import Checkpoint, Trainable

from common.hashing import bytes_sha256
from common.jsonio import read_json
from training.checkpoint_boundary import CheckpointBoundaryError
from training.checkpoint_bundle import publish_bundle
from training.checkpoint_manifest import MANIFEST, verify_files

_REPORT_SOURCE_SHA256 = "1b0f68c2b4b76b22a14c2412b9907c23f462c8c1fca331614d72a69aac1c15e6"


def report_completed_checkpoint(trainable, checkpoint_dir, checkpoint_dict_or_path):
    from training.runtime_gate import verify_owned_class
    from training.trainable import SingleRunTrainable

    verify_owned_class(type(trainable), SingleRunTrainable, "trainable")
    if (
        ray.__version__ != "2.53.0"
        or bytes_sha256(inspect.getsource(Trainable._report_class_trainable_checkpoint).encode())
        != _REPORT_SOURCE_SHA256
    ):
        raise CheckpointBoundaryError("unsupported owned Tune checkpoint publication adapter")
    if checkpoint_dict_or_path != checkpoint_dir:
        raise CheckpointBoundaryError("Tune requires the completed supplied checkpoint directory")
    source = Path(checkpoint_dir)
    body = read_json(source / MANIFEST)
    verify_files(source, body.pop("files"))
    metrics = trainable._last_result.copy() if trainable._last_result else {}
    storage = trainable._storage
    if storage is None:
        return _TrainingResult(checkpoint=Checkpoint.from_directory(str(source)), metrics=metrics)
    if type(storage.storage_filesystem) is not LocalFileSystem:
        raise CheckpointBoundaryError(
            "complete checkpoints require supported local filesystem storage"
        )
    storage._check_validation_file()
    storage._update_checkpoint_index(metrics)
    destination = Path(storage.checkpoint_fs_path)

    def copy_payload(staging):
        for path in source.iterdir():
            if path.name == MANIFEST:
                continue
            if path.is_dir():
                shutil.copytree(path, staging / path.name)
            else:
                shutil.copyfile(path, staging / path.name)
        return body

    published = publish_bundle(destination, copy_payload)
    storage.persist_artifacts(force=storage.sync_config.sync_artifacts_on_checkpoint)
    return _TrainingResult(checkpoint=Checkpoint.from_directory(str(published)), metrics=metrics)
