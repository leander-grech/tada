"""Canonical selection shared by serving and current offline evidence consumers."""

from collections.abc import Mapping

from outcomes import vocabulary


def is_selectable(candidate: Mapping) -> bool:
    return (
        bool(candidate.get("eligible", False))
        and candidate.get("end_reason") in vocabulary.SELECTABLE_END_REASONS
    )


def candidate_score(candidate: Mapping) -> tuple:
    """Eligibility, success, conservative timing, all commands, stable attempt order."""
    timing = candidate.get("timing_upper_bound_s")
    return (
        not is_selectable(candidate),
        candidate["end_reason"] != vocabulary.SUCCESS,
        timing is None,
        timing if timing is not None else float("inf"),
        candidate.get("command_cost", len(candidate["actions"])),
        candidate["run_index"],
    )


def best_candidate(candidates):
    return min((c for c in candidates if is_selectable(c)), key=candidate_score, default=None)


def is_solved(candidate: Mapping) -> bool:
    return is_selectable(candidate) and candidate["end_reason"] == vocabulary.SUCCESS


def on_time_count(candidate: Mapping) -> int:
    """Targets whose whole timing interval lies within the grading threshold."""
    threshold = candidate["grading_threshold_s"]
    return sum(
        abs(row["error_s"]) <= threshold
        if row["kind"] == "exact"
        else row["kind"] == "interval"
        and max(abs(row["lower_error_s"]), abs(row["upper_error_s"])) <= threshold
        for row in candidate["timing"]["arrivals"].values()
    )


def search_score(candidate: Mapping) -> tuple:
    """Solved, no infringement, on-time flights, conservative deviation, commands, order."""
    timing = candidate.get("timing_upper_bound_s")
    return (
        not is_solved(candidate),
        bool(candidate["max_infringement"]) or candidate["end_reason"] == vocabulary.INFRINGEMENT,
        -on_time_count(candidate),
        timing if timing is not None else float("inf"),
        candidate["command_cost"],
        candidate["run_index"],
    )


def search_select(candidates, *, stop_when_solved: bool):
    """Best candidate in simulation order and the number of candidates simulated."""
    simulated = []
    for candidate in candidates:
        simulated.append(candidate)
        if stop_when_solved and is_solved(candidate):
            break
    return min(simulated, key=search_score, default=None), len(simulated)
