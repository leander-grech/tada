"""Search selection: stop-when-solved compute and the safety-first ranking."""

from outcomes import vocabulary
from outcomes.selection import search_select


def candidate(index, end_reason, errors, *, commands=10, infringement=False):
    return {
        "run_index": index,
        "end_reason": end_reason,
        "eligible": end_reason in vocabulary.SELECTABLE_END_REASONS,
        "max_infringement": float(infringement),
        "grading_threshold_s": 60,
        "timing": {
            "arrivals": {f"A{i}": {"kind": "exact", "error_s": e} for i, e in enumerate(errors)}
        },
        "timing_upper_bound_s": sum(abs(e) for e in errors),
        "command_cost": commands,
    }


def test_fallback_stops_at_the_first_solved_attempt():
    candidates = [
        candidate(0, vocabulary.INFRINGEMENT, [0, 0], infringement=True),
        candidate(1, vocabulary.DELAYED, [10, 90]),
        candidate(2, vocabulary.SUCCESS, [50, 50], commands=30),
        candidate(3, vocabulary.SUCCESS, [0, 0], commands=1),
    ]
    selected, simulated = search_select(candidates, stop_when_solved=True)
    assert (selected["run_index"], simulated) == (2, 3)
    selected, simulated = search_select(candidates, stop_when_solved=False)
    assert (selected["run_index"], simulated) == (3, 4)


def test_unsolved_ranking_puts_safety_then_on_time_flights_before_deviation():
    candidates = [
        candidate(0, vocabulary.INFRINGEMENT, [0, 0, 0], infringement=True),
        candidate(1, vocabulary.DELAYED, [70, 70, 70]),
        candidate(2, vocabulary.DELAYED, [0, 0, 200]),
    ]
    selected, simulated = search_select(candidates, stop_when_solved=True)
    assert (selected["run_index"], simulated) == (2, 3)


def test_fewer_commands_break_equal_timing_ties():
    candidates = [
        candidate(0, vocabulary.DELAYED, [0, 100], commands=40),
        candidate(1, vocabulary.DELAYED, [0, 100], commands=12),
    ]
    assert search_select(candidates, stop_when_solved=True)[0]["run_index"] == 1
