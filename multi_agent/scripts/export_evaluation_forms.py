"""Fill the partner's per-aircraft and per-episode evaluation forms from benchmark results.

Two steps:

  replay  re-runs every recorded attempt of one results.json open-loop on parallel env
          runners and writes per-attempt traces (exact landing seconds, action counts,
          safety flags) plus their validity checks;
  forms   turns trace files into the two CSVs in the template layout.

With ``--search`` the replayed attempts are the selected attempts of rollout_search
outputs (one per world and search; search attempt a fills policy a) instead of the
results' own attempts; ``--results`` (the greedy results the searches started from)
then gives only the manifest and method they must share.

Example:
  .venv/bin/python -m scripts.export_evaluation_forms replay --procedure mxp \
      --checkpoint <ckpt> --results <dir>/greedy-1/results.json \
      --out runs/forms/mxp-greedy.json
  .venv/bin/python -m scripts.export_evaluation_forms forms \
      --traces runs/forms/{mxp,pms}-{greedy,stoch}.json --out-dir runs/forms
"""

import argparse
import csv
from collections import Counter
from functools import partial
from pathlib import Path

from common.atomic_io import atomic_json
from common.jsonio import read_json

SEEDS_CSV = Path(__file__).resolve().parents[1] / "config/contracts/supplied-evaluation-seeds.csv"


def search_attempts(data, paths):
    """Selected attempts of rollout_search outputs bound to ``data``'s checkpoint."""
    attempts = []
    for path in paths:
        search = read_json(path)
        method = search["method"]
        if method["checkpoint_sha256"] != data["method"]["checkpoint_sha256"]:
            raise ValueError(f"{path} searched another checkpoint")
        attempts += [
            {
                "world_id": world,
                "mode": "search",
                "ordinal": method["search_attempt"],
                "completed": row["selected"]["completed_result"],
            }
            for world, row in search["worlds"].items()
        ]
    return attempts


def replay(procedure, checkpoint, results, out, runners, search=()):
    import ray

    from common.runtime_log import runtime_print
    from evaluation.checkpoint_adapter import build_inference_algorithm
    from evaluation.forms.replay import replay_shard, split_inputs

    data = read_json(results)
    if search:
        data = {**data, "attempts": search_attempts(data, search)}
    manifest = data["manifest"]
    extended = {k: data["method"][k] for k in ("policy_time_limit_steps",) if k in data["method"]}
    inputs = out.with_suffix(".inputs")
    split_inputs(data, inputs)
    del data
    ray.init(num_cpus=runners, include_dashboard=False)
    algorithm = None
    try:
        algorithm = build_inference_algorithm(
            checkpoint,
            eval_env_config={
                "evaluation_seeds": [w["seed"] for w in manifest["worlds"]],
                "evaluation_seed_shards": runners,
                **extended,
            },
            callbacks=[],
            timestep_budget=None,
            explore=False,
            num_env_runners=runners,
            benchmark_manifest=manifest,
        )
        traces = [
            trace
            for shard in algorithm.eval_env_runner_group.foreach_env_runner(
                partial(replay_shard, inputs=str(inputs)), local_env_runner=False
            )
            for trace in shard
        ]
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
    summary = summarize(traces)
    atomic_json(
        out,
        {
            "procedure": procedure,
            "checkpoint": str(checkpoint),
            "results": str(results),
            "search": [str(p) for p in search],
            "summary": summary,
            "traces": traces,
        },
    )
    runtime_print("Forms", "REPLAYED", out=str(out), **summary)


def summarize(traces):
    checks = [t["checks"] for t in traces]
    return {
        "attempts": len(traces),
        "replay_problems": sum(c["replay_problem"] is not None for c in checks),
        "identical_outcome": sum(c["identical_outcome"] for c in checks),
        "identical_ledger": sum(c["identical_ledger"] for c in checks),
        "identical_actions": sum(c["identical_actions"] for c in checks),
        "live_advances": sum(c["live_advances"] for c in checks),
        "inexact_advances": sum(c["inexact_advances"] for c in checks),
        "timing_event_mismatches": sum(len(c["timing_event_mismatches"]) for c in checks),
        "outcomes": dict(Counter(t["outcome"]["kind"] for t in traces)),
    }


def forms(trace_paths, out_dir):
    from common.runtime_log import runtime_print
    from evaluation.forms.rows import (
        AIRCRAFT_HEADER,
        EPISODE_HEADER,
        aircraft_rows,
        episode_row,
        policy_index,
    )

    with SEEDS_CSV.open() as handle:
        seed_order = {int(row["seed"]): i for i, row in enumerate(csv.DictReader(handle))}
    rows = []
    for path in trace_paths:
        data = read_json(path)
        rows += [(data["procedure"], trace) for trace in data["traces"]]
    rows.sort(key=lambda r: (r[0], seed_order[r[1]["seed"]], policy_index(r[1])))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per-aircraft.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(AIRCRAFT_HEADER)
        for procedure, trace in rows:
            writer.writerows(aircraft_rows(trace, procedure))
    with (out_dir / "per-episode.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(EPISODE_HEADER)
        writer.writerows(episode_row(trace, procedure) for procedure, trace in rows)
    runtime_print("Forms", "WRITTEN", episodes=len(rows), out_dir=str(out_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("replay")
    run.add_argument("--procedure", choices=("mxp", "pms"), required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--runners", type=int, default=25)
    run.add_argument("--search", type=Path, nargs="+", default=())
    fill = commands.add_parser("forms")
    fill.add_argument("--traces", type=Path, nargs="+", required=True)
    fill.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        replay(args.procedure, args.checkpoint, args.results, args.out, args.runners, args.search)
    else:
        forms(args.traces, args.out_dir)


if __name__ == "__main__":
    main()
