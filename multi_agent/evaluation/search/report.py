"""Search procedures from rollout_search outputs: search attempt 0 and the best of the N
reseeded attempts 1..N (paper style: all N run, compute summed). Each world row also counts
the validity problems of its sources.

Example:
  .venv/bin/python -m evaluation.search.report \
      --searches <dir>/search/attempt-*.json --output <dir>/search/report.json
"""

from pathlib import Path

from common.atomic_io import atomic_json
from common.hashing import content_digest
from common.jsonio import read_json
from evaluation.benchmark.procedures import summarize_procedure, world_row
from outcomes.selection import search_select

# Method fields that must agree for searches to be composed into one procedure.
SHARED_METHOD_FIELDS = (
    "checkpoint_sha256",
    "policy_time_limit_steps",
    "candidates",
    "every_decision_times",
    "patience",
)


def problems(row):
    """Validity problems of one searched world (none for a reportable search)."""
    checks = row["replay_checks"]
    return (
        (not row["start_reproduced"])
        + bool(row["prefix_mismatches"])
        + bool(row["failed_candidates"])
        + (checks["replay_problem"] is not None)
        + sum(not checks[k] for k in ("identical_outcome", "identical_ledger", "identical_actions"))
        + bool(checks["inexact_advances"])
        + bool(checks["timing_event_mismatches"])
    )


def compose(searches):
    by_attempt = {s["method"]["search_attempt"]: s for s in searches}
    if len(by_attempt) != len(searches) or 0 not in by_attempt or len(searches) < 2:
        raise ValueError("searches need attempt 0 and distinct reseeded attempts")
    deterministic = by_attempt.pop(0)
    stochastic = list(by_attempt.values())
    method = {key: deterministic["method"][key] for key in SHARED_METHOD_FIELDS}
    if any({key: s["method"][key] for key in SHARED_METHOD_FIELDS} != method for s in stochastic):
        raise ValueError("searches use different checkpoints, limits or settings")
    worlds = deterministic["worlds"]
    if any(s["worlds"].keys() != worlds.keys() for s in stochastic):
        raise ValueError("searches cover different worlds")
    best_of = f"best-of-{len(stochastic)}-searches"
    procedures = {"search": {}, best_of: {}}
    for world, row in worlds.items():
        procedures["search"][world] = {
            **world_row(row["selected"], row["simulated_episodes"], row["seed"]),
            "problems": problems(row),
        }
        rows = [s["worlds"][world] for s in stochastic]
        best, _ = search_select(
            [{**r["selected"], "run_index": i} for i, r in enumerate(rows)],
            stop_when_solved=False,
        )
        procedures[best_of][world] = {
            **world_row(best, sum(r["simulated_episodes"] for r in rows), row["seed"]),
            "problems": sum(problems(r) for r in rows),
        }
    result = {
        "version": 1,
        "method": {
            "name": "receding-horizon-search",
            **method,
            "reseeded_attempts": sorted(by_attempt),
            "sources": {
                "deterministic": content_digest(deterministic),
                "stochastic": [content_digest(s) for s in stochastic],
            },
        },
        "procedures": {
            name: {
                "summary": {
                    **summarize_procedure(list(rows.values())),
                    "problems": sum(r["problems"] for r in rows.values()),
                },
                "worlds": rows,
            }
            for name, rows in procedures.items()
        },
    }
    result["result_sha256"] = content_digest(result)
    return result


def main():
    import argparse

    from common.runtime_log import runtime_print

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--searches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compose([read_json(p) for p in args.searches])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result, exclusive=True)
    for name, procedure in result["procedures"].items():
        runtime_print("Search", name.upper(), **procedure["summary"])


if __name__ == "__main__":
    main()
