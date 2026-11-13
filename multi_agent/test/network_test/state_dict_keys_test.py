"""Registered parameter names are checkpoint identity: module moves must not rename them."""

import json
from pathlib import Path

from support.model import build_module

GOLDEN = Path(__file__).parent / "rlm_state_dict_keys.json"


def test_rlm_state_dict_keys_are_stable():
    assert sorted(build_module().state_dict()) == json.loads(GOLDEN.read_text())
