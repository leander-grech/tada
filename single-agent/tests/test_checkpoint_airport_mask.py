"""The selected checkpoint's mask metadata remains compatible with policy loading."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from actions import action_set


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "single_agent"


def test_selected_tmb_metadata_accepts_procedure_mask_and_preserves_clearances(monkeypatch):
    checkpoint = ARTIFACTS / "mxp" / "final_model.zip"
    monkeypatch.delenv("TADA_ACTION_SET_OVERRIDE", raising=False)
    monkeypatch.setattr(action_set, "AIRPORT_MASK", "procedure")
    monkeypatch.setattr(action_set, "ACTION_SET_VERSION", "v2")

    assert action_set.run_airport_mask(checkpoint) == "procedure"
    action_set.check_run_action_set(checkpoint)
    assert action_set.AIRPORT_MASKS["procedure"]["v2"] == [
        1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]


def test_selected_pms_metadata_accepts_unmasked_policy(monkeypatch):
    checkpoint = ARTIFACTS / "pms" / "final_model.zip"
    monkeypatch.delenv("TADA_ACTION_SET_OVERRIDE", raising=False)
    monkeypatch.setattr(action_set, "AIRPORT_MASK", None)
    monkeypatch.setattr(action_set, "ACTION_SET_VERSION", "v2pms")

    assert action_set.run_airport_mask(checkpoint) is None
    action_set.check_run_action_set(checkpoint)


def test_mask_mismatch_still_blocks_checkpoint_loading(monkeypatch, tmp_path):
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"action_set": "v2", "airport_mask": "procedure"})
    )
    monkeypatch.delenv("TADA_ACTION_SET_OVERRIDE", raising=False)
    monkeypatch.setattr(action_set, "AIRPORT_MASK", None)
    monkeypatch.setattr(action_set, "ACTION_SET_VERSION", "v2")

    with pytest.raises(SystemExit, match="Airport-mask mismatch"):
        action_set.check_run_action_set(tmp_path / "model.zip")
