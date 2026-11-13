"""Search selections enter the evaluation forms as the attempt of their search's policy column."""

import json

import pytest

from evaluation.forms.rows import policy_index
from scripts.export_evaluation_forms import search_attempts

RESULTS = {"method": {"checkpoint_sha256": "c"}}


def _search(tmp_path, attempt, checkpoint="c"):
    path = tmp_path / f"attempt-{attempt}.json"
    selected = {"completed_result": {"attempt_id": f"search-{attempt}"}}
    method = {"checkpoint_sha256": checkpoint, "search_attempt": attempt}
    path.write_text(json.dumps({"method": method, "worlds": {"w": {"selected": selected}}}))
    return path


def test_each_search_attempt_fills_its_own_policy_column(tmp_path):
    attempts = search_attempts(RESULTS, [_search(tmp_path, 0), _search(tmp_path, 3)])
    assert [(policy_index(a), a["completed"]["attempt_id"]) for a in attempts] == [
        (0, "search-0"),
        (3, "search-3"),
    ]


def test_raw_attempts_fill_greedy_then_stochastic_columns():
    assert policy_index({"mode": "deterministic", "ordinal": 0}) == 0
    assert policy_index({"mode": "stochastic", "ordinal": 0}) == 1


def test_a_search_of_another_checkpoint_is_refused(tmp_path):
    with pytest.raises(ValueError, match="another checkpoint"):
        search_attempts(RESULTS, [_search(tmp_path, 0, checkpoint="other")])
