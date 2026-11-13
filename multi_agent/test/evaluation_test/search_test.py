"""Receding-horizon search: incumbent kept, prefixes forced, stop when solved."""

from types import SimpleNamespace

import pytest

from evaluation.search import report
from evaluation.search.receding import search_worlds
from evaluation.search.runner import ForcedPrefix
from outcomes import vocabulary


def candidate(end_reason, steps, errors=(0, 0), commands=5):
    return {
        "end_reason": end_reason,
        "eligible": end_reason in vocabulary.SELECTABLE_END_REASONS,
        "max_infringement": float(end_reason == vocabulary.INFRINGEMENT),
        "grading_threshold_s": 60,
        "timing": {
            "arrivals": {f"A{i}": {"kind": "exact", "error_s": e} for i, e in enumerate(errors)}
        },
        "timing_upper_bound_s": sum(abs(e) for e in errors),
        "command_cost": commands,
        "completed_result": {
            "actions": [
                {"call_sign": cs, "action": a, "decision_time_s": t}
                for t, step in steps.items()
                for cs, a in step.items()
            ]
        },
    }


GREEDY = {10.0: {"A": 3}, 20.0: {"B": 0}, 30.0: {"A": 1}}


def test_search_keeps_incumbent_until_a_sample_solves_and_stops_there():
    calls = []

    def simulate(requests):
        calls.append(requests)
        results = []
        for r in requests:
            prefix = dict(r["prefix"])
            if r["until_s"] == 20.0 and r["policy_seed"] == 1:
                results.append(candidate(vocabulary.SUCCESS, {**prefix, 20.0: {"B": 2}}))
            else:
                results.append(candidate(vocabulary.INFRINGEMENT, {**prefix, r["until_s"]: {}}))
        return results

    start = candidate(vocabulary.DELAYED, GREEDY, errors=(0, 90))
    states = search_worlds(
        {"w": start},
        simulate,
        candidates=4,
        every=1,
        policy_seed=lambda world, plan, index: index,
    )
    state = states["w"]
    assert state["incumbent"]["end_reason"] == vocabulary.SUCCESS
    assert [len(r) for r in calls] == [3, 3]
    assert [r["until_s"] for r in calls[1]] == [20.0] * 3
    assert all(dict(r["prefix"]) == {10.0: {"A": 3}} for r in calls[1])
    assert state["simulated_episodes"] == 7
    assert state["prefix_mismatches"] == 0
    assert state["improvements"] == [{"plan": 1, "until_s": 20.0, "explore": True}]


def test_patience_ends_a_world_after_planning_points_that_keep_the_incumbent():
    steps = {float(t): {"A": 1} for t in range(10, 90, 10)}
    rounds = []

    def simulate(requests):
        rounds.append(requests[0]["until_s"])
        return [
            candidate(
                vocabulary.DELAYED,
                {**dict(r["prefix"]), **{t: s for t, s in steps.items() if t >= r["until_s"]}},
                errors=(0, 80 if r["until_s"] == 20.0 and r["policy_seed"] == 0 else 100),
            )
            for r in requests
        ]

    start = candidate(vocabulary.DELAYED, steps, errors=(0, 90))
    states = search_worlds(
        {"w": start}, simulate, candidates=2, every=1, policy_seed=lambda *a: a[2], patience=2
    )
    assert rounds == [10.0, 20.0, 30.0, 40.0]
    assert states["w"]["improvements"] == [{"plan": 1, "until_s": 20.0, "explore": True}]
    unlimited = search_worlds(
        {"w": start}, simulate, candidates=2, every=1, policy_seed=lambda *a: a[2]
    )
    assert unlimited["w"]["plans"] == len(steps)


def test_plans_branch_at_the_first_decision_and_every_n_decisions_after_it():
    steps = {float(t): {"A": 1} for t in range(10, 90, 10)}
    rounds = []

    def simulate(requests):
        rounds.append(requests[0]["until_s"])
        return [candidate(vocabulary.DELAYED, steps, errors=(0, 100)) for _ in requests]

    start = candidate(vocabulary.DELAYED, steps, errors=(0, 90))
    state = search_worlds(
        {"w": start}, simulate, candidates=2, every=3, policy_seed=lambda *a: a[2]
    )["w"]
    assert rounds == [10.0, 40.0, 70.0]
    # Replayed decisions before each planning point (0 + 3 + 6) out of all simulated ones.
    assert (state["prefix_decisions"], state["simulated_decisions"]) == (9, 4 * len(steps))


def test_greedy_continuation_is_simulated_once_the_incumbent_came_from_a_sample():
    requests_seen = []

    def simulate(requests):
        requests_seen.append([r["explore"] for r in requests])
        return [
            candidate(
                vocabulary.DELAYED,
                {**dict(r["prefix"]), r["until_s"]: {"A": 4}, 40.0: {"A": 1}},
                errors=(0, 80 if r["policy_seed"] == 0 else 100),
            )
            for r in requests
        ]

    start = candidate(vocabulary.DELAYED, GREEDY, errors=(0, 90))
    search_worlds({"w": start}, simulate, candidates=3, every=1, policy_seed=lambda *a: a[2])
    assert requests_seen[0] == [True, True]
    assert requests_seen[1] == [False, True]


def test_a_sampled_start_gets_a_greedy_continuation_at_its_first_planning_point():
    requests_seen = []

    def simulate(requests):
        requests_seen.append([r["explore"] for r in requests])
        return [
            candidate(vocabulary.DELAYED, {**dict(r["prefix"]), r["until_s"]: {}}) for r in requests
        ]

    start = candidate(vocabulary.DELAYED, GREEDY, errors=(0, 90))
    search_worlds(
        {"w": start},
        simulate,
        candidates=3,
        every=1,
        policy_seed=lambda *a: a[2],
        greedy_start=False,
    )
    assert requests_seen[0] == [False, True]


class _Runner(ForcedPrefix):
    def __init__(self, env):
        self.env = SimpleNamespace(envs=[SimpleNamespace(unwrapped=env)])


def test_prefix_is_forced_only_before_the_planning_point_of_the_requested_attempt():
    env = SimpleNamespace(attempt_id="a", simulator=SimpleNamespace(world=None))
    runner = _Runner(env)
    runner.forced_attempt_id, runner.forced_until_s = "a", 20.0
    runner.forced_steps = {10.0: {"A": 3}}
    policy = {"A": 9, "B": 9}

    def executed_at(seconds):
        env.simulator.world = SimpleNamespace(age_in_seconds=seconds)
        return runner._executed(0, policy)

    assert executed_at(10.0) == {"A": 3, "B": 0}
    assert executed_at(15.0) == {"A": 0, "B": 0}
    assert executed_at(20.0) == policy
    env.attempt_id = "overrun"
    assert executed_at(10.0) == policy


def _searched(seed, end_reason, episodes, errors):
    selected = {
        **candidate(end_reason, {}, errors=errors),
        "attempt_id": f"{seed}-{end_reason}",
        "mode": "search",
        "ordinal": 0,
        "action_counts": {"learned_command_count": 5, "preprocessing_command_count": 0},
    }
    selected["timing"]["target_count"] = len(errors)
    return {
        "seed": seed,
        "simulated_episodes": episodes,
        "selected": selected,
        "start_reproduced": True,
        "prefix_mismatches": 0,
        "failed_candidates": [],
        "replay_checks": {
            "replay_problem": None,
            "identical_outcome": True,
            "identical_ledger": True,
            "identical_actions": True,
            "inexact_advances": 0,
            "timing_event_mismatches": 0,
        },
    }


def _search_output(attempt, row):
    method = dict.fromkeys(report.SHARED_METHOD_FIELDS, "same")
    return {"method": {**method, "search_attempt": attempt}, "worlds": {"w": row}}


def test_best_of_searches_keeps_the_solved_search_and_counts_all_their_compute():
    searches = [
        _search_output(0, _searched(7, vocabulary.DELAYED, 9, (0, 90))),
        _search_output(1, _searched(7, vocabulary.DELAYED, 30, (0, 70))),
        _search_output(2, _searched(7, vocabulary.SUCCESS, 4, (0, 50))),
    ]
    procedures = report.compose(searches)["procedures"]
    assert procedures["search"]["summary"]["solved"] == 0
    best = procedures["best-of-2-searches"]
    assert best["summary"]["solved"] == 1
    assert best["worlds"]["w"]["simulated_episodes"] == 34
    assert best["summary"]["problems"] == 0
    with pytest.raises(ValueError, match="distinct"):
        report.compose([searches[0], searches[1], searches[1]])
