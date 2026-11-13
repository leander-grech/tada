"""The manifest alone makes a bundle verifiable: file digests, version tokens, identity."""

import pytest

from common.atomic_io import atomic_json
from common.jsonio import read_json
from training import checkpoints, versions
from training.checkpoint_boundary import CheckpointBoundaryError
from training.checkpoint_bundle import publish_bundle
from training.checkpoint_manifest import MANIFEST, build_manifest
from training.checkpoints import inspect_checkpoint
from training.run_config import reseal

IDENTITY = {"runtime": {"backend": "cpu"}, "native": {"version": "n"}, "catalog": {"c": 1}}
PAYLOAD = ("algorithm.pkl", "boundary.pkl", "counters.json")


COUNTERS = {
    "version": 1,
    "lifetime_iterations": 0,
    "lifetime_env_steps": 0,
    "lifetime_updates": 0,
    "optimizer_work": {},
    "phase_start_iterations": 0,
    "curriculum": {"version": 1, "stage": None},
}


def write_payload(staging):
    # Placeholder Algorithm state: these tests stop at manifest gates before unpickling.
    for name in ("algorithm.pkl", "boundary.pkl"):
        (staging / name).write_bytes(b"not a pickle")
    atomic_json(staging / "counters.json", COUNTERS)


@pytest.fixture
def run(native_run):
    return native_run


@pytest.fixture
def body(run):
    return build_manifest(
        run,
        provenance={"commit": "c" * 40},
        lineage=[],
        execution_segment=None,
        module_layout={},
        identity=IDENTITY,
    )


@pytest.fixture
def published(tmp_path, body, monkeypatch):
    monkeypatch.setattr(checkpoints, "current_identity", lambda run: IDENTITY)
    return publish_bundle(tmp_path / "ck", lambda staging: write_payload(staging) or body)


def test_publish_writes_manifest_with_file_digests_and_no_marker_files(published):
    names = {p.name for p in published.iterdir()}
    assert names == {MANIFEST, *PAYLOAD}
    assert set(read_json(published / MANIFEST)["files"]) == names - {MANIFEST}


def test_inspect_rejects_tampered_file(published, run):
    (published / "algorithm.pkl").write_bytes(b"tampered")
    with pytest.raises(CheckpointBoundaryError, match="algorithm.pkl"):
        inspect_checkpoint(published, run)


def test_inspect_reports_token_diff(published, run, monkeypatch):
    monkeypatch.setattr("actions.rewards.REWARD_VERSION", "ma-reward-next")
    with pytest.raises(versions.TokenMismatch, match="reward: .* -> ma-reward-next"):
        inspect_checkpoint(published, run)


def test_weights_scope_needs_only_tokens_and_integrity(published, run):
    other = reseal(run, optimizer={**run.optimizer, "learning_rate": 0.123})
    with pytest.raises(ValueError, match="resume contract mismatch"):
        inspect_checkpoint(published, other)
    # Weights readers skip runtime identity and config transition and reach the payload.
    with pytest.raises(CheckpointBoundaryError, match="unreadable checkpoint state payload"):
        inspect_checkpoint(published, scope="weights")


def test_unlisted_file_is_rejected(published, run):
    (published / "extra.bin").write_bytes(b"x")
    with pytest.raises(CheckpointBoundaryError, match="unknown"):
        inspect_checkpoint(published, run)


@pytest.mark.parametrize("scope", ["resume", "weights"])
def test_both_scopes_reject_a_tampered_saved_config(published, run, scope):
    manifest = read_json(published / MANIFEST)
    optimizer = manifest["run_config"]["optimizer"]
    optimizer["learning_rate"] = optimizer["learning_rate"] * 2
    atomic_json(published / MANIFEST, manifest)
    with pytest.raises(CheckpointBoundaryError, match="invalid checkpoint semantic contract"):
        inspect_checkpoint(published, run if scope == "resume" else None, scope=scope)
