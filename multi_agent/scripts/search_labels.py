"""Search labels for distillation: receding-horizon search (evaluation.search.receding) on fresh
operational worlds, then each safe final plan replayed once with its observations recorded.

Worlds come from a reservation-guarded ``operational_evaluation`` seed stream that also
excludes the supplied cohort, so no benchmark or reserved world is ever searched or labelled.
Every search starts greedy, or with ``--start sampled`` from one sampled attempt per world, so
the labels cover the states the stochastic policy visits. A recorded replay that does not reproduce its plan's actions
exactly is dropped and counted.

Example:
  .venv/bin/python -m scripts.search_labels --checkpoint <ckpt> --worlds 2000 \
      --patience 3 --out runs/<dir>/labels
"""

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

from common.atomic_io import atomic_json
from common.hashing import content_digest


def _rng(*identity):
    return int(content_digest(["search-labels", *identity])[:8], 16)


def main():
    import ray

    from atc_env.completed_scenario import CompletedScenarioResult
    from common.runtime_log import runtime_print
    from evaluation.benchmark.procedures import summarize_procedure, world_row
    from evaluation.benchmark.supplied import supplied_seeds
    from evaluation.checkpoint_adapter import build_inference_algorithm, inspect_inference_checkpoint
    from evaluation.search.labels import LabelEnvRunner, fresh_shard
    from evaluation.search.receding import policy_steps, search_worlds
    from training.checkpoint_manifest import checkpoint_digest
    from training.reservations import effective_reserved_seeds
    from training.seed_stream import SEED_MAX, ScenarioSeedStream

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--stream-root", type=int, default=20261104)
    parser.add_argument("--start", choices=["greedy", "sampled"], default="greedy")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--policy-time-limit-scale", type=float, default=1.5)
    parser.add_argument("--runners", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run = inspect_inference_checkpoint(args.checkpoint)
    reserved = effective_reserved_seeds(run.env_payload()["reserved_manifest"])
    # Supplied seeds above the native range can never be drawn.
    supplied = frozenset(seed for seed in supplied_seeds() if seed <= SEED_MAX)
    stream = ScenarioSeedStream(
        args.stream_root, (0, 0, 0), reserved | supplied, domain="operational_evaluation"
    )
    seeds = [stream.next_seed() for _ in range(args.worlds)]
    frames = {seed: _rng("frame", seed) for seed in seeds}
    steps = round(args.policy_time_limit_scale * run.scenario["episode_steps"])
    sampled = args.start == "sampled"
    # The finished search is kept so a failed replay can be rerun without searching again.
    search_path = args.out / "search.pkl"
    identity = content_digest(
        [
            checkpoint_digest(args.checkpoint),
            seeds,
            args.start,
            args.candidates,
            args.patience,
            steps,
        ]
    )
    reused = search_path.exists()

    ray.init(num_cpus=args.runners, include_dashboard=False)
    algorithm = None
    started = time.monotonic()
    try:
        algorithm = build_inference_algorithm(
            args.checkpoint,
            eval_env_config={
                "evaluation_seeds": seeds,
                "evaluation_seed_shards": args.runners,
                "policy_time_limit_steps": steps,
            },
            callbacks=[],
            timestep_budget=None,
            explore=False,
            num_env_runners=args.runners,
            env_runner_cls=LabelEnvRunner,
        )
        group = algorithm.eval_env_runner_group

        def simulate(requests, record=False):
            rows = {
                index: (completed, recorded)
                for shard in group.foreach_env_runner(
                    partial(
                        fresh_shard,
                        requests=requests,
                        frames=frames,
                        runners=args.runners,
                        record=record,
                    ),
                    local_env_runner=False,
                )
                for index, completed, recorded in shard
            }
            runtime_print("SearchLabels", "ROUND", simulated=len(requests))
            candidates = [
                {
                    **CompletedScenarioResult.from_dict(rows[i][0]).consumer_fields(),
                    "mode": "search",
                    "ordinal": 0,
                }
                for i in range(len(requests))
            ]
            return (candidates, [rows[i][1] for i in range(len(requests))]) if record else candidates

        if reused:
            with search_path.open("rb") as stream_in:
                saved = pickle.load(stream_in)
            if saved["identity"] != identity:
                raise ValueError(f"{search_path} was searched with different inputs")
            start, states = saved["start"], saved["states"]
        else:
            start = dict(
                zip(
                    seeds,
                    simulate(
                        [
                            {
                                "world": s,
                                "until_s": None,
                                "prefix": [],
                                "policy_seed": _rng("start", s) if sampled else 0,
                                "explore": sampled,
                            }
                            for s in seeds
                        ]
                    ),
                )
            )
            states = search_worlds(
                start,
                simulate,
                candidates=args.candidates,
                every=1,
                policy_seed=lambda world, plan, index: _rng("policy", world, plan, index),
                greedy_start=not sampled,
                patience=args.patience,
            )
            args.out.mkdir(parents=True, exist_ok=True)
            with search_path.open("xb") as stream_out:
                pickle.dump(
                    {"identity": identity, "start": start, "states": states}, stream_out, protocol=5
                )
        safe = [s for s, state in states.items() if not state["incumbent"]["max_infringement"]]
        replays, recorded = simulate(
            [
                {
                    "world": s,
                    "until_s": float("inf"),
                    "prefix": sorted(
                        (t, step) for t, step in policy_steps(states[s]["incumbent"]).items() if step
                    ),
                    "policy_seed": 0,
                    "explore": False,
                }
                for s in safe
            ],
            record=True,
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()

    labels, dropped = {}, []
    for seed, replay, steps_recorded in zip(safe, replays, recorded):
        incumbent = states[seed]["incumbent"]["completed_result"]["actions"]
        if content_digest(replay["completed_result"]["actions"]) != content_digest(incumbent):
            dropped.append(seed)
            continue
        labels[seed] = steps_recorded
    worlds = {
        seed: {
            "start": world_row(start[seed], 1, seed),
            "final": world_row(state["incumbent"], state["simulated_episodes"], seed),
            "improvements": state["improvements"],
            "prefix_mismatches": state["prefix_mismatches"],
            "labelled": seed in labels,
        }
        for seed, state in states.items()
    }
    with (args.out / "labels.pkl").open("xb") as stream_out:
        pickle.dump(labels, stream_out, protocol=5)
    summary = {
        "start": summarize_procedure([w["start"] for w in worlds.values()]),
        "final": summarize_procedure([w["final"] for w in worlds.values()]),
        "labelled_worlds": len(labels),
        "dropped_replays": dropped,
        "prefix_mismatches": sum(w["prefix_mismatches"] for w in worlds.values()),
        "improved_worlds": sum(bool(w["improvements"]) for w in worlds.values()),
    }
    atomic_json(
        args.out / "summary.json",
        {
            "method": {
                "name": "receding-horizon-search-labels",
                "checkpoint": str(args.checkpoint),
                "start": args.start,
                "candidates": args.candidates,
                "patience": args.patience,
                "policy_time_limit_steps": steps,
                "stream": stream.get_state(),
                "seeds_sha256": content_digest(seeds),
                "search_reused": reused,
            },
            "wall_seconds": time.monotonic() - started,
            "summary": summary,
            "worlds": {str(seed): row for seed, row in worlds.items()},
        },
    )
    runtime_print("SearchLabels", "DONE", **summary)


if __name__ == "__main__":
    main()
