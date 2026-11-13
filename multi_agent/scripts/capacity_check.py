"""Procedure capacity: can each flight's own clearances reach its target at all?

Greedy single-flight capacity analysis over the supplied benchmark worlds.
Every world starts at its policy start (after auto-solve) and runs to the policy time
limit. The do-nothing episode gives each flight's deviation d0 (landing − target,
negative = early). Every flight outside ±60 s is then flown ALONE, the others doing
nothing, with a greedy policy that picks the first legal clearance of the observation
mask (our mask rules) at every 45 s decision until it lands. Here the flight
decides at every step (no agent selection) and separation never ends the
episode, so the env's step is replaced by its interpret/simulate parts:

  delay:   LENGTHEN_TROMBONE, SLOW_DOWN_LARGE > MEDIUM > SMALL, then a turn-and-return
           when the previous decision was not a turn;
  advance: VECTOR_TO_ILS, SHORTEN_TROMBONE, SPEED_UP_MEDIUM > SMALL, then SKIP_4..1.

A flight is capacity-feasible when its greedy landing lies within the grading
threshold or beyond the target on the other side; a world is feasible when every
flight is. Separation is ignored, so feasibility is necessary, not sufficient.

Example:
  .venv/bin/python -m scripts.capacity_check --checkpoint <checkpoint_dir> \
      --manifest <manifest.json> \
      --out <output.json>
"""

import argparse
from functools import partial
from pathlib import Path
from uuid import uuid4

from actions.action_table import Action
from common.atomic_io import atomic_json
from common.jsonio import read_json

THRESHOLD_S = 60
TURNS = (Action.TURN_LEFT_AND_RETURN_TO_ROUTE, Action.TURN_RIGHT_AND_RETURN_TO_ROUTE)
DELAY = (
    Action.LENGTHEN_TROMBONE,
    Action.SLOW_DOWN_LARGE,
    Action.SLOW_DOWN_MEDIUM,
    Action.SLOW_DOWN_SMALL,
    *TURNS,
)
ADVANCE = (
    Action.VECTOR_TO_ILS,
    Action.SHORTEN_TROMBONE,
    Action.SPEED_UP_MEDIUM,
    Action.SPEED_UP_SMALL,
    Action.SKIP_4_WAYPOINTS_NOT_NEXT,
    Action.SKIP_3_WAYPOINTS_NOT_NEXT,
    Action.SKIP_2_WAYPOINTS_NOT_NEXT,
    Action.SKIP_1_WAYPOINT,
)


def greedy(callsign, preferences):
    """Joint-action chooser: first legal preference for one flight, never two turns in a row."""
    previous = Action.DO_NOTHING

    def choose(obs):
        nonlocal previous
        if callsign not in obs:
            return {}
        mask = obs[callsign]["action_mask"]
        previous = next(
            (a for a in preferences if mask[a] and not (a in TURNS and previous in TURNS)),
            Action.DO_NOTHING,
        )
        return {callsign: int(previous)}

    return choose


def decision_step(env, callsign, choose):
    """One 45 s decision of ``callsign`` (None: nobody) without agent selection or episode
    ends (separation is ignored)."""
    env._conflict_override_cache = None
    env._step_deviation_proxies = None
    env._step_selection_infr_map = None
    env._step_prev_world_all_infr = None
    env._agents_allowed_to_act = set() if callsign is None else {callsign}
    actions = {} if callsign is None else choose(env._build_observations([callsign]))
    interpreted, *_ = env._interpret_actions(actions)
    env._simulate_and_reward(interpreted)
    env._check_pending_returns()


def fly(env, benchmark_id, world, callsign=None, preferences=()):
    """One episode from the policy start, ``callsign`` flown greedily; landing second of every flight."""
    from evaluation.benchmark.contract import attempt_rng_ids
    from evaluation.forms.landing import step_landings
    from evaluation.forms.replay import LiveLandingTrace
    from evaluation.sampling import CompletedScenarioCollector

    env.completed_collector = CompletedScenarioCollector()
    env._next_attempt_id, env._next_attempt_seed = f"capacity-{uuid4().hex}", world["seed"]
    env.benchmark_request = {"rng": attempt_rng_ids(benchmark_id, world["world_id"], 0)}
    env.reset()
    choose = greedy(callsign, preferences)
    with LiveLandingTrace(env.simulator) as trace:
        for _ in range(int(env.policy_time_limit_steps)):
            aircraft = {ac.call_sign: ac for ac in env.simulator.world.aircraft}
            if all(ac.mission.completed for ac in aircraft.values()):
                break
            flying = callsign is not None and not aircraft[callsign].mission.completed
            decision_step(env, callsign if flying else None, choose)
    _, after = step_landings(env.simulator.world, int(env.noop_verification_max_seconds))
    return {**after, **trace.landed}


def world_capacity(env, benchmark_id, world):
    eta = {t["callsign"]: t["eta_s"] for t in world["targets"]}
    nothing = fly(env, benchmark_id, world)
    flights = {}
    for callsign, target in eta.items():
        d0 = nothing[callsign] - target if callsign in nothing else None
        row = {"d0_s": d0, "greedy_s": None}
        if d0 is not None and abs(d0) > THRESHOLD_S:
            preferences = ADVANCE if d0 > 0 else DELAY
            landed = fly(env, benchmark_id, world, callsign, preferences)
            if callsign in landed:
                row["greedy_s"] = landed[callsign] - target
        row["feasible"] = d0 is not None and (
            abs(d0) <= THRESHOLD_S
            or (
                row["greedy_s"] is not None
                and (row["greedy_s"] <= THRESHOLD_S if d0 > 0 else row["greedy_s"] >= -THRESHOLD_S)
            )
        )
        flights[callsign] = row
    return {
        "seed": world["seed"],
        "world_id": world["world_id"],
        "feasible": all(r["feasible"] for r in flights.values()),
        "flights": flights,
    }


def capacity_shard(runner, *, manifest_path):
    from common.env_utils import get_base_env

    env = get_base_env(runner.env)
    manifest = read_json(manifest_path)
    worlds = {w["seed"]: w for w in manifest["worlds"]}
    return [
        world_capacity(env, manifest["compatibility_sha256"], worlds[seed])
        for seed in env._eval_seeds
    ]


def main():
    import ray

    from common.runtime_log import runtime_print
    from evaluation.checkpoint_adapter import (
        build_inference_algorithm,
        inspect_inference_checkpoint,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runners", type=int, default=24)
    parser.add_argument("--policy-time-limit-scale", type=float, default=1.5)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    saved = inspect_inference_checkpoint(args.checkpoint).scenario["episode_steps"]
    steps = round(args.policy_time_limit_scale * saved)
    ray.init(num_cpus=args.runners, include_dashboard=False)
    algorithm = None
    try:
        algorithm = build_inference_algorithm(
            args.checkpoint,
            eval_env_config={
                "evaluation_seeds": [w["seed"] for w in manifest["worlds"]],
                "evaluation_seed_shards": args.runners,
                "policy_time_limit_steps": steps,
            },
            callbacks=[],
            timestep_budget=None,
            explore=False,
            num_env_runners=args.runners,
            benchmark_manifest=manifest,
        )
        worlds = [
            row
            for shard in algorithm.eval_env_runner_group.foreach_env_runner(
                partial(capacity_shard, manifest_path=str(args.manifest)), local_env_runner=False
            )
            for row in shard
        ]
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
    worlds.sort(key=lambda w: w["seed"])
    atomic_json(
        args.out,
        {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "policy_time_limit_steps": steps,
            "feasible_worlds": sum(w["feasible"] for w in worlds),
            "worlds": worlds,
        },
    )
    runtime_print(
        "Capacity",
        "DONE",
        worlds=len(worlds),
        feasible=sum(w["feasible"] for w in worlds),
        infeasible=[w["seed"] for w in worlds if not w["feasible"]],
    )


if __name__ == "__main__":
    main()
