"""Receding-horizon rollout search on chosen benchmark worlds (evaluation.search.receding).

Single-agent style attempts: every search starts from the greedy attempt of a deterministic
results.json of the same checkpoint (manifest and policy time limit bound to it) on that
attempt's world RNG, and must reproduce it exactly. Attempt 0 seeds the sampled candidates
as ``search-policy:<plan>:<index>`` under ordinal 0; attempt ``a`` reseeds only those
samples under ordinal ``a`` (the forms' policy column ``a``). Every selected plan is
forms-replayed with its validity checks. Without ``--seeds`` every world is searched (solved
starts cost one episode, simulated once for all attempts).

Several attempts are searched in one session sharing the runners, one output each
(``--out`` with an ``{attempt}`` field); every candidate keeps its own seed, so a
search's result does not depend on the attempts beside it.

Example:
  .venv/bin/python -m scripts.rollout_search --checkpoint <ckpt> \
      --results <dir>/greedy-1/results.json --every 10 --attempts $(seq 0 20) \
      --out <dir>/search/attempt-{attempt}.json
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

from common.atomic_io import atomic_json
from common.hashing import canonical_json, content_digest
from common.jsonio import read_json


def main():
    import ray

    from atc_env.completed_scenario import CompletedScenarioResult
    from common.runtime_log import runtime_print
    from evaluation.benchmark.contract import attempt_rng_id, attempt_rng_ids
    from evaluation.benchmark.procedures import summarize_procedure, world_row
    from evaluation.checkpoint_adapter import build_inference_algorithm
    from evaluation.search.receding import search_worlds
    from evaluation.search.runner import SearchEnvRunner, simulate_shard, trace_shard

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--every", type=int, default=1, help="replan every N decision times")
    parser.add_argument(
        "--patience", type=int, help="stop a world after N planning points without a change"
    )
    parser.add_argument("--attempts", type=int, nargs="+", default=[0])
    parser.add_argument("--runners", type=int, default=24)
    args = parser.parse_args()
    if len(args.attempts) > 1 and "{attempt}" not in str(args.out):
        raise ValueError("--out needs an {attempt} field to search several attempts")

    data = read_json(args.results)
    if data["protocol"]["mode"] != "deterministic":
        raise ValueError("searches start from the greedy attempts of deterministic results")
    manifest = data["manifest"]
    benchmark_id = manifest["compatibility_sha256"]
    by_seed = {w["seed"]: w for w in manifest["worlds"]}
    worlds = {by_seed[s]["world_id"]: by_seed[s] for s in args.seeds or by_seed}
    # A search is keyed by (attempt, world id); all fly the greedy attempt's world RNG.
    keys = [(a, w) for a in args.attempts for w in worlds]
    world_of = {k: worlds[k[1]] for k in keys}
    rng = {w: attempt_rng_ids(benchmark_id, w, 0) for w in worlds}
    recorded = {
        row["world_id"]: row["completed"]["actions"]
        for row in data["attempts"]
        if row["world_id"] in worlds
    }
    if recorded.keys() != worlds.keys():
        raise ValueError("results lack the greedy attempt of a searched world")

    def policy_seed(key, plan, index):
        return attempt_rng_id(benchmark_id, key[1], key[0], f"search-policy:{plan}:{index}")

    ray.init(num_cpus=args.runners, include_dashboard=False)
    algorithm = None
    started = time.monotonic()
    rounds = 0
    failures = {k: [] for k in keys}
    try:
        algorithm = build_inference_algorithm(
            args.checkpoint,
            eval_env_config={
                "evaluation_seeds": [w["seed"] for w in manifest["worlds"]],
                "evaluation_seed_shards": args.runners,
                "policy_time_limit_steps": data["method"]["policy_time_limit_steps"],
            },
            callbacks=[],
            timestep_budget=None,
            explore=False,
            num_env_runners=args.runners,
            benchmark_manifest=manifest,
            env_runner_cls=SearchEnvRunner,
        )
        group = algorithm.eval_env_runner_group

        def simulate(requests):
            nonlocal rounds
            rounds += 1
            results = dict(
                pair
                for shard in group.foreach_env_runner(
                    partial(
                        simulate_shard,
                        requests=requests,
                        worlds=world_of,
                        frames={k: rng[k[1]]["frame"] for k in keys},
                        runners=args.runners,
                    ),
                    local_env_runner=False,
                )
                for pair in shard
            )
            candidates = [
                {
                    **CompletedScenarioResult.from_dict(results[i]).consumer_fields(),
                    "mode": "search",
                    "ordinal": request["world"][0],
                }
                for i, request in enumerate(requests)
            ]
            for request, candidate in zip(requests, candidates):
                error = candidate["completed_result"]["diagnostics"]["sampling"]["tail_error"]
                if not candidate["complete"] or error:
                    failures[request["world"]].append(
                        {
                            "until_s": request["until_s"],
                            "explore": request["explore"],
                            "policy_seed": request["policy_seed"],
                            "complete": candidate["complete"],
                            "error": error,
                        }
                    )
            runtime_print("Search", "ROUND", round=rounds, simulated=len(requests))
            return candidates

        greedy = dict(
            zip(
                worlds,
                simulate(
                    [
                        {
                            "world": (args.attempts[0], w),
                            "until_s": None,
                            "prefix": [],
                            "policy_seed": rng[w]["policy"],
                            "explore": False,
                        }
                        for w in worlds
                    ]
                ),
            )
        )
        start = {k: greedy[k[1]] for k in keys}
        reproduced = {
            w: content_digest(c["completed_result"]["actions"]) == content_digest(recorded[w])
            for w, c in greedy.items()
        }
        states = search_worlds(
            start,
            simulate,
            candidates=args.candidates,
            every=args.every,
            policy_seed=policy_seed,
            patience=args.patience,
        )
        rows = [
            {
                "world_id": k[1],
                "mode": "search",
                "ordinal": k[0],
                # The persisted (JSON) form, as forms replay compares recorded attempts.
                "completed": json.loads(canonical_json(state["incumbent"]["completed_result"])),
            }
            for k, state in states.items()
        ]
        traces = dict(
            pair
            for shard in group.foreach_env_runner(
                partial(
                    trace_shard,
                    rows=rows,
                    worlds=worlds,
                    benchmark_id=benchmark_id,
                    runners=args.runners,
                ),
                local_env_runner=False,
            )
            for pair in shard
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    wall_seconds = time.monotonic() - started
    traced = {k: traces[i] for i, k in enumerate(states)}
    for attempt in args.attempts:
        per_world = {}
        for k, state in states.items():
            if k[0] != attempt:
                continue
            trace = traced[k]
            per_world[k[1]] = {
                **world_row(state["incumbent"], state["simulated_episodes"], world_of[k]["seed"]),
                "start": world_row(start[k], 1, world_of[k]["seed"]),
                "start_reproduced": reproduced[k[1]],
                "plans": state["plans"],
                "improvements": state["improvements"],
                "prefix_mismatches": state["prefix_mismatches"],
                "prefix_decisions": state["prefix_decisions"],
                "simulated_decisions": state["simulated_decisions"],
                "failed_candidates": failures[k],
                "replay_checks": trace["checks"],
                "final_landing_s": trace["final_landing_s"],
                "targets": trace["targets"],
                "selected": state["incumbent"],
            }
        summary = summarize_procedure(list(per_world.values()))
        out = Path(str(args.out).format(attempt=attempt))
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            out,
            {
                "method": {
                    "name": "receding-horizon-search",
                    "candidates": args.candidates,
                    "every_decision_times": args.every,
                    "patience": args.patience,
                    "search_attempt": attempt,
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": data["method"]["checkpoint_sha256"],
                    "policy_time_limit_steps": data["method"]["policy_time_limit_steps"],
                    "results_sha256": data["result_sha256"],
                },
                # Session totals, shared by every attempt searched in it.
                "wall_seconds": wall_seconds,
                "rounds": rounds,
                "session_attempts": args.attempts,
                "summary": summary,
                "worlds": per_world,
            },
        )
        runtime_print("Search", "DONE", attempt=attempt, **summary)
        for row in per_world.values():
            runtime_print(
                "Search",
                "WORLD",
                attempt=attempt,
                seed=row["seed"],
                end_reason=row["end_reason"],
                on_time=row["on_time_count"],
                deviation=row["timing_upper_bound_s"],
                commands=row["command_cost"],
                episodes=row["simulated_episodes"],
                mismatches=row["prefix_mismatches"],
                failed=len(row["failed_candidates"]),
                replay_problem=row["replay_checks"]["replay_problem"],
                identical_outcome=row["replay_checks"]["identical_outcome"],
                identical_actions=row["replay_checks"]["identical_actions"],
            )


if __name__ == "__main__":
    main()
