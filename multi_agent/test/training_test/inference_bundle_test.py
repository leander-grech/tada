"""Portable models retain exact weights and reject resume and damaged contracts."""

import json
from pathlib import Path

import pytest

from training.checkpoints import inspect_checkpoint, read_checkpoint_payload
from training.inference_bundle import export_inference_bundle


@pytest.fixture
def exported(tmp_path):
    source = next((Path(__file__).resolve().parents[3] / "artifacts/multi_agent").rglob("manifest.json")).parent
    destination = export_inference_bundle(source, tmp_path / "public")
    return source, destination


def test_public_export_preserves_every_weight_and_normalizer(exported):
    source, destination = exported
    original, _ = read_checkpoint_payload(source)
    loaded = inspect_checkpoint(destination, scope="weights")
    public, boundary = read_checkpoint_payload(destination)
    assert boundary is None
    left = original["learner_group"]["learner"]["rl_module"]
    right = public["learner_group"]["learner"]["rl_module"]
    assert left.keys() == right.keys()
    for module in left:
        assert left[module].keys() == right[module].keys()
        for name, value in left[module].items():
            actual = right[module][name]
            if type(value) in (int, float):
                assert type(actual) is type(value)
                assert actual == value
            else:
                expected = value.detach().cpu().numpy() if hasattr(value, "detach") else value
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
                assert actual.tobytes() == expected.tobytes()
    assert loaded.run.scenario["config_root"]
    assert {p.name for p in destination.iterdir()} == {"manifest.json", "weights.npz"}
    with pytest.raises(ValueError, match="inference-only"):
        inspect_checkpoint(destination, loaded.run)


@pytest.mark.parametrize("damage", ["checksum", "shape", "tokens", "reservation", "scenario"])
def test_public_reader_rejects_damaged_bundle(exported, damage):
    _, destination = exported
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "checksum":
        with (destination / "weights.npz").open("ab") as stream:
            stream.write(b"damage")
    elif damage == "shape":
        module = next(iter(manifest["module_layout"].values()))
        descriptor = next(value for value in module.values() if "shape" in value)
        descriptor["shape"] = [999999]
    elif damage == "tokens":
        manifest["versions"][next(iter(manifest["versions"]))] = "unsupported"
    elif damage == "reservation":
        manifest["run_config"]["reserved_manifest"]["heldout"][0]["seed"] += 1
    else:
        manifest["run_config"]["scenario"]["config_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_checkpoint(destination, scope="weights")


def test_numeric_reader_refuses_pickle_payload_even_with_updated_checksum(exported):
    import numpy as np

    from common.hashing import file_sha256

    _, destination = exported
    payload = destination / "weights.npz"
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays[next(iter(arrays))] = np.array([{"unsafe": "object"}], dtype=object)
    np.savez(payload, **arrays)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["weights.npz"] = file_sha256(payload)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="numeric inference weights"):
        read_checkpoint_payload(destination)


def test_public_method_provenance_authenticates_numeric_payload(exported, monkeypatch):
    from common.hashing import file_sha256
    from training import provenance

    _, destination = exported
    monkeypatch.setattr(
        provenance, "_runtime_identity", lambda _: {"native": {"wheel_sha256": "fixture"}}
    )
    method = provenance.checkpoint_method_provenance(destination)
    assert method["model_sha256"] == file_sha256(destination / "weights.npz")
    assert method["checkpoint_sha256"] == file_sha256(destination / "manifest.json")
    assert method["training_config"]["semantic_digest"] == inspect_checkpoint(
        destination, scope="weights"
    ).run.semantic_digest
