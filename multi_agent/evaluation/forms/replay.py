"""Open-loop replay of recorded benchmark attempts with exact landing seconds.

Recorded executed actions are fed back into a fresh
benchmark env on the same world and attempt RNG, without a policy. Every live native
advance is re-run one second at a time from the same world and commands; the result
must equal the live world, so the landing seconds observed in between belong to the
evaluated trajectory. Aircraft still airborne at the terminal step land in the
no-command continuation that the no-op verifier also simulates.

Each trace carries the checks that make it trustworthy: identical outcome, ledger and
actions versus the recorded attempt, exact decomposition of every live advance, and
agreement with every native MISS_TIMED_LANDING timestamp.
"""

import json
from collections import Counter, OrderedDict

from flight_simulator import World

from evaluation.forms.landing import (
    continue_to_landing,
    same_world,
    step_landings,
    timing_event_mismatches,
    warning_separation_pairs,
)

REPLAYED_FIELDS = ("call_sign", "action", "decision_time_s", "phase", "route_after")


class LiveLandingTrace:
    """Spies the env's live ``outcome_owner.advance`` calls for the duration of one attempt."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.landed: dict[str, float] = {}
        self.advances = 0
        self.inexact = 0

    def __enter__(self):
        owner = self.simulator.outcome_owner
        live = owner.advance

        def advance(world, steps, commands=()):
            commands = list(commands)
            result = live(world, steps, commands)
            fine, landed = step_landings(world, steps, commands)
            self.advances += 1
            self.inexact += not same_world(fine, result)
            self.landed.update(landed)
            return result

        owner.advance = advance
        return self

    def __exit__(self, *exc):
        del self.simulator.outcome_owner.advance


def _stripped(actions):
    return [{k: a.get(k) for k in REPLAYED_FIELDS} for a in actions]


def _recorded_steps(actions):
    """Policy-step actions by decision time (pre-episode auto-solve and internal returns excluded)."""
    steps = OrderedDict()
    for a in actions:
        if a.get("phase") is None and not a.get("return_wp"):
            steps.setdefault(a["decision_time_s"], {})[a["call_sign"]] = int(a["action"])
    return steps


def _world(data) -> World:
    return World.from_json(json.dumps(data))


def initial_landings(manifest_world, max_seconds):
    """Landing seconds of the pre-auto-solve world with no commands at all."""
    return continue_to_landing(_world(manifest_world["pre_auto_solve_world"]), max_seconds)


def replay_attempt(env, manifest, world, row):
    """Replay one recorded attempt; returns the live trace and the replayed completed record."""
    from evaluation.benchmark.contract import attempt_rng_ids
    from evaluation.sampling import CompletedScenarioCollector

    recorded = row["completed"]
    # Search attempts that kept their greedy start share its record; a repeated id would
    # reuse the env's previous capture, so each replay gets its own (mode, column) id.
    attempt_id = f"forms-{row['mode']}-{row['ordinal']}-{recorded['attempt_id']}"
    env.completed_collector = CompletedScenarioCollector()
    env._next_attempt_id, env._next_attempt_seed = attempt_id, world["seed"]
    # A search attempt (ordinal: its policy column) flies its greedy start's world RNG.
    ordinal = 0 if row["mode"] == "search" else row["ordinal"]
    env.benchmark_request = {
        "rng": attempt_rng_ids(manifest["compatibility_sha256"], row["world_id"], ordinal)
    }
    obs, _ = env.reset()
    pending = list(_recorded_steps(recorded["actions"]).items())
    problem = None
    with LiveLandingTrace(env.simulator) as trace:
        for _ in range(int(env.policy_time_limit_steps) + 2):
            now = float(env.simulator.world.age_in_seconds)
            actions = {}
            if pending and abs(now - pending[0][0]) < 1e-6:
                decided, actions = pending.pop(0)
                if not set(actions) <= set(obs):
                    problem = f"t={decided}: recorded {sorted(actions)} not observed {sorted(obs)}"
                    break
            elif pending and now > pending[0][0]:
                problem = f"skipped recorded step t={pending[0][0]} (now {now})"
                break
            obs, _, term, trunc, _ = env.step({a: actions.get(a, 0) for a in obs})
            if term.get("__all__") or trunc.get("__all__"):
                break
    if pending and problem is None:
        problem = f"{len(pending)} recorded steps not reached"
    replayed = env.completed_collector.take(attempt_id)
    return trace, problem, replayed, env.simulator.world


def _action_counts(actions):
    return dict(Counter(f"{a['call_sign']}:{int(a['action'])}" for a in actions))


def attempt_trace(env, manifest, world, row, initial):
    """Everything the forms need for one attempt, plus its validity checks."""
    from common.hashing import content_digest

    trace, problem, replayed, terminal = replay_attempt(env, manifest, world, row)
    recorded = row["completed"]
    cap = int(env.noop_verification_max_seconds)
    end_world, after = step_landings(terminal, cap)
    final = {**after, **trace.landed}
    replay = json.loads(json.dumps(replayed.to_dict())) if replayed is not None else None
    policy = [a for a in recorded["actions"] if a.get("phase") is None and not a.get("return_wp")]
    auto_solve = [a for a in recorded["actions"] if a.get("phase") == "pre_episode"]
    return {
        "seed": world["seed"],
        "world_id": row["world_id"],
        "mode": row["mode"],
        "ordinal": row["ordinal"],
        "outcome": recorded["outcome"],
        "separation_events": len(recorded["ledger"].get("separation_events", [])),
        "exit_events": len(recorded["ledger"].get("exit_events", [])),
        "targets": world["targets"],
        "initial_landing_s": initial,
        "final_landing_s": final,
        "warning_pairs": sorted(warning_separation_pairs(end_world, env.settings.core.safety)),
        "policy_actions": _action_counts(policy),
        "auto_solve_actions": _action_counts(auto_solve),
        "checks": {
            "replay_problem": problem,
            "identical_outcome": replay is not None
            and content_digest(replay["outcome"]) == content_digest(recorded["outcome"]),
            "identical_ledger": replay is not None
            and content_digest(replay["ledger"]) == content_digest(recorded["ledger"]),
            "identical_actions": replay is not None
            and _stripped(replay["actions"]) == _stripped(recorded["actions"]),
            "live_advances": trace.advances,
            "inexact_advances": trace.inexact,
            "timing_event_mismatches": timing_event_mismatches(end_world, final),
        },
    }


def split_inputs(results, directory):
    """One small file per world (its manifest entry and attempts), so each env runner
    loads only the worlds it owns instead of the whole results file."""
    from pathlib import Path

    from common.atomic_io import atomic_json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = results["manifest"]
    for world in manifest["worlds"]:
        attempts = [
            {k: row[k] for k in ("world_id", "mode", "ordinal")}
            | {
                "completed": {
                    k: row["completed"][k] for k in ("attempt_id", "actions", "outcome", "ledger")
                }
            }
            for row in results["attempts"]
            if row["world_id"] == world["world_id"]
        ]
        atomic_json(
            directory / f"{world['seed']}.json",
            {
                "compatibility_sha256": manifest["compatibility_sha256"],
                "world": world,
                "attempts": attempts,
            },
        )


def replay_shard(runner, *, inputs):
    """Env-runner task: trace every recorded attempt of the world seeds this runner owns."""
    from pathlib import Path

    from common.env_utils import get_base_env
    from common.jsonio import read_json

    env = get_base_env(runner.env)
    cap = int(env.noop_verification_max_seconds)
    traces = []
    for seed in env._eval_seeds:
        data = read_json(Path(inputs) / f"{seed}.json")
        manifest = {"compatibility_sha256": data["compatibility_sha256"]}
        initial = initial_landings(data["world"], cap)
        traces += [
            attempt_trace(env, manifest, data["world"], row, initial) for row in data["attempts"]
        ]
    return traces
