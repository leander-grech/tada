"""Run manifests record actual native artifact identity without leaking secrets."""

import hashlib
import json

import pytest

from training.provenance import write_run_manifest
from training.run_config import load_run_config


@pytest.mark.parametrize("lock_bytes", [None, b"resolved-dependency-lock"])
def test_manifest_records_native_artifact_provenance_and_omits_secrets(
    config_path, tmp_path, monkeypatch, lock_bytes
) -> None:
    from training import provenance

    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setattr(provenance, "__file__", str(source_root / "training/provenance.py"))
    if lock_bytes is not None:
        (source_root / "uv.lock").write_bytes(lock_bytes)
    monkeypatch.setenv("PRIVATE_API_KEY", "do-not-publish-this-value")
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            {
                "provenance": "external/unverified",
                "version": "0.2.80",
                "source_identity": None,
                "wheel_sha256": "b" * 64,
            }
        )
    )
    path = tmp_path / "manifest.json"
    config = load_run_config(config_path)
    write_run_manifest(
        path,
        config,
        {
            "native_manifest_path": str(native),
            "environment": dict(PRIVATE_API_KEY="do-not-publish-this-value"),
        },
    )
    text = path.read_text()
    manifest = json.loads(text)
    assert "do-not-publish-this-value" not in text
    assert "PRIVATE_API_KEY" not in text
    assert manifest["runtime"]["native"]["source_identity"] is None
    assert manifest["runtime"]["native"]["version"] == "0.2.80"
    assert manifest["config"]["semantic_digest"] == config.semantic_digest
    assert (
        manifest["reservation_conditions_sha256"] == config.reserved_conditions["conditions_sha256"]
    )
    assert manifest["config"]["reserved_conditions"]["purpose"] == "fixture"
    assert manifest["provenance"]["wheel_sha256"] == "b" * 64
    assert manifest["dependency_lock_sha256"] == (
        hashlib.sha256(lock_bytes).hexdigest() if lock_bytes is not None else None
    )


def test_manifest_cannot_accept_a_boolean_as_actor_selection_proof(native_run, tmp_path):
    path = tmp_path / "manifest.json"
    write_run_manifest(path, native_run, {"selection_verified": True})
    assert json.loads(path.read_text())["runtime"]["owned_learning"]["selection_verified"] is False
