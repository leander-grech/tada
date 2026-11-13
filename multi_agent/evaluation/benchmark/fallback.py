"""Fallback best-of-N: run greedy, then sample attempts until one is solved or N are spent.

Every attempt is a deterministic function of its canonical identity (benchmark, world,
mode, ordinal, checkpoint), so the method is composed exactly from a
deterministic and a stochastic results.json of one checkpoint: the attempts
it consumes are the ones an online run would simulate. The output is a separate method
record with its own compute count, never a single-attempt or pass@k result.

Example:
  .venv/bin/python -m evaluation.benchmark.fallback \
      --deterministic runs/supplied-seeds/forms-v3/mxp-rollout4/greedy-20/results.json \
      --stochastic runs/supplied-seeds/forms-v3/mxp-rollout4/stoch-20/results.json \
      --output runs/supplied-seeds/fallback/mxp-rollout4.json
"""

from pathlib import Path

from common.atomic_io import atomic_json
from common.hashing import content_digest
from common.jsonio import read_json
from evaluation.benchmark.evidence import AttemptEvidence
from evaluation.benchmark.procedures import summarize_procedure, world_row
from outcomes.selection import search_select

METHOD = "fallback-best-of-n"
# Method fields that must agree for attempts to be composed into one procedure.
SHARED_METHOD_FIELDS = ("checkpoint_sha256", "policy_time_limit_steps")


def load_results(path):
    data = read_json(path)
    body = {key: value for key, value in data.items() if key != "result_sha256"}
    if data.get("result_sha256") != content_digest(body):
        raise ValueError(f"results digest mismatch: {path}")
    return data


def candidates_by_world(data):
    attempts = [AttemptEvidence.from_dict(row) for row in data["attempts"]]
    worlds = {w["world_id"]: [] for w in data["manifest"]["worlds"]}
    for attempt in sorted(attempts, key=lambda a: a.ordinal):
        worlds[attempt.world_id].append(
            {
                **attempt.completed.consumer_fields(),
                "mode": attempt.mode,
                "ordinal": attempt.ordinal,
            }
        )
    return worlds


def compose(deterministic, stochastic, *, max_samples):
    """Greedy, best of the samples, and the fallback, each with its own compute."""
    if deterministic["protocol"]["mode"] != "deterministic":
        raise ValueError("fallback needs a deterministic primary")
    if stochastic["protocol"]["mode"] != "stochastic" or stochastic["protocol"]["k"] < max_samples:
        raise ValueError("fallback needs at least max_samples stochastic attempts")
    if deterministic["manifest"]["manifest_sha256"] != stochastic["manifest"]["manifest_sha256"]:
        raise ValueError("fallback sources use different manifests")
    method = {key: deterministic["method"].get(key) for key in SHARED_METHOD_FIELDS}
    if method != {key: stochastic["method"].get(key) for key in SHARED_METHOD_FIELDS}:
        raise ValueError("fallback sources use different checkpoints or time limits")
    seeds = {w["world_id"]: w["seed"] for w in deterministic["manifest"]["worlds"]}
    greedy = candidates_by_world(deterministic)
    samples = {w: c[:max_samples] for w, c in candidates_by_world(stochastic).items()}
    procedures = {"greedy": {}, f"best-of-{max_samples}": {}, METHOD: {}}
    for world, seed in seeds.items():
        ordered = [{**c, "run_index": i} for i, c in enumerate(greedy[world] + samples[world])]
        selections = {
            "greedy": search_select(ordered[:1], stop_when_solved=False),
            f"best-of-{max_samples}": search_select(ordered[1:], stop_when_solved=False),
            METHOD: search_select(ordered, stop_when_solved=True),
        }
        for name, (candidate, simulated) in selections.items():
            procedures[name][world] = world_row(candidate, simulated, seed)
    result = {
        "version": 1,
        "method": {
            "name": METHOD,
            "max_samples": max_samples,
            **method,
            "sources": {
                "deterministic": deterministic["result_sha256"],
                "stochastic": stochastic["result_sha256"],
            },
        },
        "manifest_sha256": deterministic["manifest"]["manifest_sha256"],
        "procedures": {
            name: {"summary": summarize_procedure(list(rows.values())), "worlds": rows}
            for name, rows in procedures.items()
        },
    }
    result["result_sha256"] = content_digest(result)
    return result


def main():
    import argparse

    from common.runtime_log import runtime_print

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--stochastic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()
    result = compose(
        load_results(args.deterministic),
        load_results(args.stochastic),
        max_samples=args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result, exclusive=True)
    for name, procedure in result["procedures"].items():
        runtime_print("Fallback", name.upper(), **procedure["summary"])


if __name__ == "__main__":
    main()
