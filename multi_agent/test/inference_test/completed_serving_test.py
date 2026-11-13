"""Candidate eligibility precedes timing and reward."""

import pytest

from inference.service import NoValidSolutionError, select_best_candidate


def candidate(index, reason, eligible=True, timing=None, reward=0):
    return dict(
        run_index=index,
        end_reason=reason,
        eligible=eligible,
        timing_upper_bound_s=timing,
        total_deviation=None,
        actions=[],
        cumulative_reward=reward,
        max_infringement=0,
        command_cost=0,
    )


def test_safe_success_precedes_high_reward_delayed_and_unsafe_low_deviation():
    good = candidate(1, "success", timing=60, reward=-100)
    assert (
        select_best_candidate(
            [candidate(0, "success", False, 0), good, candidate(2, "delayed", True, 1, 1000)],
            "test",
        )
        == good
    )


def test_unverified_legacy_and_incomplete_cannot_be_selected():
    legacy = candidate(0, "success")
    del legacy["eligible"]
    with pytest.raises(NoValidSolutionError):
        select_best_candidate([legacy, candidate(1, "success", False)], "test")


def test_candidate_exception_is_retained_as_incomplete(monkeypatch):
    from types import SimpleNamespace
    from typing import Any, cast

    from inference import service

    monkeypatch.setattr(service, "World", SimpleNamespace(from_json=lambda value: object()))

    def fail(*args, **kwargs):
        raise ValueError("unsupported owned world")

    monkeypatch.setattr(service, "run_single_candidate", fail)
    with pytest.raises(NoValidSolutionError) as raised:
        service.run_inference_candidate(cast(Any, object()), "{}", "owned", run_count_override=1)
    candidates = raised.value.candidates
    assert len(candidates) == 1
    assert not candidates[0]["complete"] and not candidates[0]["eligible"]
    assert "unsupported owned world" in candidates[0]["diagnostics"]["error"]


def test_diagnostic_main_selects_same_canonical_candidate_as_service(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from inference.test import inference_smoke as diagnostic

    results = [
        candidate(0, "success", False, 0),
        candidate(1, "success", True, 20),
        candidate(2, "success", True, 10),
    ]
    results[2]["command_cost"] = 3
    expected = select_best_candidate(results, "service")["run_index"]
    inputs = tmp_path / "world.json"
    inputs.write_text("{}")
    monkeypatch.setattr(
        diagnostic,
        "_parse_args",
        lambda: SimpleNamespace(
            runs=3, world_json=str(inputs), checkpoint="fixture", explore=False, print_actions=0
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "World",
        SimpleNamespace(from_json=lambda _: SimpleNamespace(aircraft=[], age_in_seconds=0)),
    )
    monkeypatch.setattr(diagnostic, "configure_default_quiet_env", lambda: None)
    monkeypatch.setattr(diagnostic.ray, "init", lambda **_: None)
    monkeypatch.setattr(diagnostic.ray, "shutdown", lambda: None)
    pending = iter([{k: v for k, v in row.items() if k != "run_index"} for row in results])
    monkeypatch.setattr(
        diagnostic,
        "InferenceEngine",
        lambda *_, **__: SimpleNamespace(
            run_episode_with_metrics=lambda *_, **__: next(pending), shutdown=lambda: None
        ),
    )
    tables = []
    logs = []
    monkeypatch.setattr(
        diagnostic, "_print_table", lambda headers, rows, **_: tables.append((headers, rows))
    )
    monkeypatch.setattr(
        diagnostic, "runtime_print", lambda *args, **kwargs: logs.append((args, kwargs))
    )
    diagnostic.main()
    selection = next(rows for headers, rows in tables if "Score" in headers)
    assert [i for i, row in enumerate(selection) if "best" in row[-1]] == [expected]
    assert selection[0][-1] == "(filtered)"
    assert "None" in selection[2]
    scoring = next(values for args, values in logs if args[1] == "scoring")
    assert "weights" not in scoring
