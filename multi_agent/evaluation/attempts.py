"""Attempt records: per-attempt and per-seed rows and their CSV round trips."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

from outcomes import vocabulary

SOLVED_END_REASON = vocabulary.SUCCESS


DELAYED_END_REASON = vocabulary.DELAYED


INFRINGEMENT_END_REASON = vocabulary.INFRINGEMENT


@dataclasses.dataclass(frozen=True)
class AttemptRecord:
    """One episode of one scenario, at a specific attempt index."""

    seed: int
    attempt: int
    solved: bool
    end_reason: str | None
    max_dev: float | None
    """Worst single-aircraft absolute time deviation, seconds."""
    init_dev_per_ac: float | None
    """Mean absolute deviation per aircraft in the initial scenario, seconds."""
    final_dev_per_ac: float | None
    """Reported mean; exact-known only with final_timing_* metadata, otherwise unverified."""
    infr_severity: float
    """Forecast severity diagnostic; never realized safety evidence."""
    ep_return: float
    steps: int
    nz_actions: int
    nz_ex_trb: int
    spd: int
    aircraft: int
    final_timing_exact_count: int | None = None
    final_timing_target_count: int | None = None
    final_timing_interval_count: int | None = None
    final_timing_unknown_count: int | None = None
    final_timing_source: str | None = None
    infringement_source: str = "legacy_unspecified"
    attempt_id: str | None = None
    completed_result_json: str | None = None


@dataclasses.dataclass(frozen=True)
class SeedOutcome:
    """Aggregate of every attempt spent on one scenario seed."""

    seed: int
    solved: bool
    attempts_used: int
    first_success_attempt: int | None
    censored: bool
    """True when the attempt cap was reached without solving."""
    best_max_dev: float | None
    best_end_reason: str | None
    deterministic_solved: bool | None
    deterministic_end_reason: str | None
    deterministic_max_dev: float | None


def canonical_attempt(attempt: AttemptRecord):
    """Read attempt evidence; rows lacking canonical evidence remain diagnostic only."""
    if attempt.completed_result_json is None:
        if attempt.attempt_id is not None:
            raise ValueError("current attempt is missing canonical evidence")
        return attempt, None
    from atc_env.completed_scenario import CompletedScenarioResult

    result = CompletedScenarioResult.from_dict(json.loads(attempt.completed_result_json))
    canonical = completed_attempt(result)
    if result.attempt_id != attempt.attempt_id or canonical.seed != attempt.seed:
        raise ValueError("canonical attempt identity does not match CSV identity")
    return dataclasses.replace(canonical, attempt=attempt.attempt), {
        **result.consumer_fields(),
        "run_index": attempt.attempt,
    }


def decode_attempts(attempts):
    decoded = [canonical_attempt(a) for a in attempts]
    identities = [a.attempt_id for a, candidate in decoded if candidate is not None]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate canonical attempt identity")
    return decoded


def _write_csv_rows(
    path: Path,
    rows: list[Any],
    *,
    row_type: type[Any],
    sort_key: Callable[[Any], Any],
) -> None:
    ordered_rows = sorted(rows, key=sort_key)
    fields = [f.name for f in dataclasses.fields(row_type)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(dataclasses.asdict(row))


def write_attempts_raw(path: Path, attempts: list[AttemptRecord]) -> None:
    """Write one row per attempt, ordered by seed then attempt index."""
    _write_csv_rows(
        path,
        attempts,
        row_type=AttemptRecord,
        sort_key=lambda a: (a.seed, a.attempt),
    )


def read_attempts_raw(path: Path) -> list[AttemptRecord]:
    """Read back a file written by :func:`write_attempts_raw`.

    Stored runs are the only record of a benchmark: the episodes are gone, so
    every later comparison has to start from these rows.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            AttemptRecord(
                seed=int(row["seed"]),
                attempt=int(row["attempt"]),
                solved=row["solved"] == "True",
                end_reason=row["end_reason"] or None,
                max_dev=float(row["max_dev"]) if row["max_dev"] else None,
                init_dev_per_ac=float(row["init_dev_per_ac"]) if row["init_dev_per_ac"] else None,
                final_dev_per_ac=float(row["final_dev_per_ac"])
                if row["final_dev_per_ac"]
                else None,
                infr_severity=float(row["infr_severity"]),
                ep_return=float(row["ep_return"]),
                steps=int(row["steps"]),
                nz_actions=int(row["nz_actions"]),
                nz_ex_trb=int(row["nz_ex_trb"]),
                spd=int(row["spd"]),
                aircraft=int(row["aircraft"]),
                **{
                    field: int(row[field]) if row.get(field) else None
                    for field in (
                        "final_timing_exact_count",
                        "final_timing_target_count",
                        "final_timing_interval_count",
                        "final_timing_unknown_count",
                    )
                },
                final_timing_source=row.get("final_timing_source") or None,
                infringement_source=row.get("infringement_source") or "legacy_unspecified",
                attempt_id=row.get("attempt_id") or None,
                completed_result_json=row.get("completed_result_json") or None,
            )
            for row in csv.DictReader(handle)
        ]


def write_attempts_per_seed(path: Path, outcomes: list[SeedOutcome]) -> None:
    """Write one row per scenario seed."""
    _write_csv_rows(
        path,
        outcomes,
        row_type=SeedOutcome,
        sort_key=lambda o: o.seed,
    )


def read_attempts_per_seed(path: Path) -> list[SeedOutcome]:
    """Read back a file written by :func:`write_attempts_per_seed`.

    The three ``deterministic_*`` columns are optional: a run without a greedy
    pass leaves them empty, and a trimmed archive may drop them entirely.  Only
    the greedy baselines are lost in that case, never the per-seed outcomes.
    """

    def optional_bool(value: str) -> bool | None:
        return None if not value else value == "True"

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            SeedOutcome(
                seed=int(row["seed"]),
                solved=row["solved"] == "True",
                attempts_used=int(row["attempts_used"]),
                first_success_attempt=(
                    None if not row["first_success_attempt"] else int(row["first_success_attempt"])
                ),
                censored=row["censored"] == "True",
                best_max_dev=float(row["best_max_dev"]) if row["best_max_dev"] else None,
                best_end_reason=row["best_end_reason"] or None,
                deterministic_solved=optional_bool(row.get("deterministic_solved", "")),
                deterministic_end_reason=row.get("deterministic_end_reason") or None,
                deterministic_max_dev=(
                    None
                    if not row.get("deterministic_max_dev")
                    else float(row["deterministic_max_dev"])
                ),
            )
            for row in csv.DictReader(handle)
        ]


def completed_attempt(result):
    """Compatibility row with canonical JSON evidence for offline round trips."""
    import json

    from actions.actions import count_nonzero_actions

    fields = result.consumer_fields()
    timing = fields["timing"]
    count = timing["target_count"]
    exact = timing["exact_count"] == count and count > 0
    initial = result.diagnostics.get("raw_initial_deviation")
    nz, ex, speed = count_nonzero_actions(result.actions)
    return AttemptRecord(
        seed=int(result.diagnostics["seed"]) if result.diagnostics.get("seed") is not None else -1,
        attempt=0,
        solved=fields["eligible"] and fields["end_reason"] == vocabulary.SUCCESS,
        end_reason=fields["end_reason"],
        max_dev=max(abs(v["error_s"]) for v in timing["arrivals"].values()) if exact else None,
        init_dev_per_ac=initial / count if initial is not None and count else None,
        final_dev_per_ac=timing["exact_mean_absolute_error_s"],
        infr_severity=0.0,
        ep_return=result.cumulative_reward,
        steps=result.diagnostics.get("steps", 0),
        nz_actions=nz,
        nz_ex_trb=ex,
        spd=speed,
        aircraft=count,
        final_timing_exact_count=timing["exact_count"],
        final_timing_target_count=count,
        final_timing_interval_count=timing["interval_count"],
        final_timing_unknown_count=timing["unknown_count"],
        final_timing_source=timing["source"],
        infringement_source="forecast_unavailable",
        attempt_id=result.attempt_id,
        completed_result_json=json.dumps(result.to_dict(), allow_nan=False),
    )


def completed_summary_row(result):
    """Shared evaluation and detailed-report terminal columns."""
    fields = result.consumer_fields()
    attempt = completed_attempt(result)
    return {
        **fields,
        "seed": attempt.seed,
        "outcome": fields["end_reason"],
        "init_dev": attempt.init_dev_per_ac,
        "final_dev": attempt.final_dev_per_ac,
        "final_dev_exact_known_mean_s": attempt.final_dev_per_ac,
        "final_timing": fields["timing"],
        "forecast_infringement_severity": None,
        "steps": attempt.steps,
        "nz_actions": attempt.nz_actions,
        "nz_ex_trb": attempt.nz_ex_trb,
        "spd": attempt.spd,
        "aircraft": attempt.aircraft,
        "aircraft_count": attempt.aircraft,
        "cum_reward": attempt.ep_return,
        "nonzero_actions": attempt.nz_actions,
        "completed_result": result.to_dict(),
    }
