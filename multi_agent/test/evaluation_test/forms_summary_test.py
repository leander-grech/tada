"""Results table: greedy and pooled stochastic columns, per-column success range, clearances."""

import csv

from actions.action_table import Action
from evaluation.forms.rows import EPISODE_HEADER, episode_row
from evaluation.forms.summary import table

EXTEND, SLOW = int(Action.LENGTHEN_TROMBONE), int(Action.SLOW_DOWN_SMALL)
TRACE = {
    "warning_pairs": [],
    "targets": [{"callsign": "A", "eta_s": 100.0}, {"callsign": "B", "eta_s": 200.0}],
    "initial_landing_s": {"A": 90.0, "B": 230.0},
    "final_landing_s": {"A": 101.0, "B": 199.0},
    "policy_actions": {"A:0": 4, f"A:{EXTEND}": 1, "B:0": 3, f"B:{SLOW}": 2},
    "auto_solve_actions": {f"A:{EXTEND}": 1},
}


def _trace(seed, mode, ordinal, kind, separation_events=0):
    return {
        **TRACE,
        "seed": seed,
        "mode": mode,
        "ordinal": ordinal,
        "outcome": {"kind": kind},
        "separation_events": separation_events,
    }


def test_greedy_and_pooled_stochastic_rows_count_outcomes_and_per_column_successes(tmp_path):
    traces = [
        _trace(7, "deterministic", 0, "success"),
        _trace(8, "deterministic", 0, "safe_delayed"),
        _trace(7, "stochastic", 0, "success"),
        _trace(8, "stochastic", 0, "success"),
        _trace(7, "stochastic", 1, "infringement", separation_events=1),
        _trace(8, "stochastic", 1, "safe_delayed"),
    ]
    path = tmp_path / "per-episode.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(EPISODE_HEADER)
        writer.writerows(episode_row(t, "mxp") for t in traces)
    greedy, stochastic = table({"policy": path})
    assert greedy["policy_type"] == "Greedy (0)"
    assert (greedy["episodes"], greedy["success_rate"]) == (2, 0.5)
    assert greedy["delayed_count"] == 1
    assert (stochastic["policies"], stochastic["episodes"]) == (2, 4)
    assert stochastic["success_count"] == 2 and stochastic["Loss of separation_count"] == 1
    # Column 1 solved both worlds and column 2 none; together they solved both.
    assert (stochastic["success_per_policy_min"], stochastic["success_per_policy_max"]) == (0, 2)
    assert stochastic["worlds_solved_by_any_policy"] == 2
    # Two extensions (one auto-solve) and two slow-downs in every episode of the trace.
    assert stochastic["mean_clearances"] == 4
