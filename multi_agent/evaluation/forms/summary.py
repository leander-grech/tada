"""Results table of filled evaluation forms: per variant, procedure and policy type, the
outcome counts and rates, timing and clearance burden of the per-episode CSV rows.

Greedy is policy column 0; the stochastic columns 1-20 are pooled, with the per-column
success range and the worlds any column solved beside the pooled rates. Safety, timing
and clearances are separate columns, never combined into one score.

Example:
  .venv/bin/python -m evaluation.forms.summary --output <dir>/summary.csv \
      --forms "policy=<dir>/raw/per-episode.csv" "policy + search=<dir>/search/per-episode.csv"
"""

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.forms.rows import EPISODE_HEADER, LEAD, PMS_ACTIONS, TMB_ACTIONS

END_REASONS = ("success", "delayed", "timeout", "airspace_exit", "Loss of separation")
POLICY_TYPES = {"Greedy (0)": range(0, 1), "Stochastic (1-20)": range(1, 21)}


def episodes(path):
    """The per-episode rows of one forms CSV as dicts."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.reader(handle))[len(EPISODE_HEADER) :]
    blocks = {"TMB": len(LEAD), "PMS": len(LEAD) + len(TMB_ACTIONS)}
    for row in rows:
        start = blocks[row[1]]
        actions = [int(c) for c in row[start : start + len(PMS_ACTIONS)]]
        yield {
            "procedure": row[1],
            "seed": int(row[2]),
            "policy": int(row[3]),
            "end_reason": row[4],
            "initial_s": float(row[5]),
            "final_s": float(row[6]),
            "aman_sequence": row[7] == "Yes",
            "infringement": row[8] == "Yes",
            # Every non-"do nothing" clearance (the first column of each block).
            "clearances": sum(actions[1:]),
        }


def summarize(variant, procedure, policy_type, rows):
    count = len(rows)
    reasons = Counter(r["end_reason"] for r in rows)
    per_policy = Counter(r["policy"] for r in rows if r["end_reason"] == "success")
    policies = sorted({r["policy"] for r in rows})
    solved = [per_policy[p] for p in policies]
    return {
        "variant": variant,
        "procedure": procedure,
        "policy_type": policy_type,
        "policies": len(policies),
        "episodes": count,
        **{f"{reason}_count": reasons[reason] for reason in END_REASONS},
        **{f"{reason}_rate": round(reasons[reason] / count, 4) for reason in END_REASONS},
        "aman_sequence_met_rate": round(sum(r["aman_sequence"] for r in rows) / count, 4),
        "infringement_rate": round(sum(r["infringement"] for r in rows) / count, 4),
        "mean_initial_abs_deviation_s": round(statistics.fmean(r["initial_s"] for r in rows), 2),
        "mean_final_abs_deviation_s": round(statistics.fmean(r["final_s"] for r in rows), 2),
        "mean_clearances": round(statistics.fmean(r["clearances"] for r in rows), 2),
        "success_per_policy_min": min(solved),
        "success_per_policy_median": statistics.median(solved),
        "success_per_policy_max": max(solved),
        "worlds_solved_by_any_policy": len(
            {r["seed"] for r in rows if r["end_reason"] == "success"}
        ),
    }


def table(forms):
    """Summary rows for ``forms`` = {variant: per-episode CSV path}."""
    rows = []
    for variant, path in forms.items():
        grouped = defaultdict(list)
        for row in episodes(path):
            policy_type = next(n for n, cols in POLICY_TYPES.items() if row["policy"] in cols)
            grouped[row["procedure"], policy_type].append(row)
        rows += [
            summarize(variant, procedure, policy_type, grouped[procedure, policy_type])
            for procedure in sorted({p for p, _ in grouped}, reverse=True)
            for policy_type in POLICY_TYPES
            if grouped[procedure, policy_type]
        ]
    return rows


def main():
    import argparse

    from common.runtime_log import runtime_print

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forms", nargs="+", required=True, help="VARIANT=per-episode.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = table(dict(item.split("=", 1) for item in args.forms))
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        runtime_print("Summary", "ROW", **row)


if __name__ == "__main__":
    main()
