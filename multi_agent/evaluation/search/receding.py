"""Receding-horizon rollout search over joint policy actions.

At each planning point the search simulates candidates from the incumbent's executed
prefix to episode end and keeps the best by ``search_score``. The incumbent is always a
candidate, so a search never ends worse than its start (greedy, or a
sampled attempt). Candidates are a greedy continuation (skipped while the incumbent
already is one from an earlier point, as a greedy start is from its first) and
sampled continuations seeded by (world, plan, index). Planning points are the
incumbent's first decision time and every ``every``-th after it (the single-agent
search's M decisions per plan): between decision times no agent acts, so ``every=1`` is
the finest useful granularity. A search stops once its incumbent is solved, fewer than
``every`` decision times are left, or (with ``patience``) that many planning points in a
row kept the incumbent.

Every simulation restarts from reset and executes the prefix verbatim (replay restore),
so its prefix must reproduce the incumbent's; disagreements are counted as mismatches.
The forced (replayed) and all simulated decision times are counted, so the share a
snapshot restore would save is measured rather than assumed.
"""

from outcomes.selection import is_solved, search_score


def policy_steps(candidate):
    """Policy-step commands by decision time; no-ops dropped (forcing them is the default)."""
    steps = {}
    for a in candidate["completed_result"]["actions"]:
        if a.get("phase") is None and not a.get("return_wp"):
            step = steps.setdefault(float(a["decision_time_s"]), {})
            if int(a["action"]):
                step[a["call_sign"]] = int(a["action"])
    return steps


def _prefix(steps, until_s):
    return {t: s for t, s in steps.items() if t < until_s and s}


def search_worlds(
    starts, simulate, *, candidates, every, policy_seed, greedy_start=True, patience=None
):
    """Search every world in lockstep rounds; ``simulate`` runs one round's requests.

    ``starts`` maps world → start candidate (consumer fields with ``completed_result``),
    greedy or, with ``greedy_start=False``, sampled;
    ``simulate(requests)`` returns one candidate per request, in order;
    ``policy_seed(world, plan, index)`` seeds a continuation (index −1: greedy);
    ``patience`` (None: unlimited) planning points without a change end a world's search.
    """
    states = {
        world: {
            "incumbent": start,
            "greedy_from_s": float("-inf") if greedy_start else float("inf"),
            "time_s": float("-inf"),
            "plans": 0,
            "unchanged_plans": 0,
            "simulated_episodes": 1,
            "improvements": [],
            "prefix_mismatches": 0,
            "prefix_decisions": 0,
            "simulated_decisions": len(policy_steps(start)),
        }
        for world, start in starts.items()
    }
    while True:
        round_requests = []
        for world, state in states.items():
            if is_solved(state["incumbent"]) or state["unchanged_plans"] == patience:
                continue
            later = [t for t in sorted(policy_steps(state["incumbent"])) if t > state["time_s"]]
            # Plans branch at the first decision time and every ``every`` decisions after it.
            step = every if state["plans"] else 1
            if len(later) < step:
                continue
            until_s = later[step - 1]
            prefix = _prefix(policy_steps(state["incumbent"]), until_s)
            greedy_needed = state["greedy_from_s"] > until_s
            requests = [(-1, False)] * greedy_needed + [
                (index, True) for index in range(candidates - 1 - greedy_needed)
            ]
            state["time_s"] = until_s
            round_requests += [
                {
                    "world": world,
                    "until_s": until_s,
                    "prefix": sorted(prefix.items()),
                    "policy_seed": policy_seed(world, state["plans"], index),
                    "explore": explore,
                }
                for index, explore in requests
            ]
        if not round_requests:
            break
        results = simulate(round_requests)
        for world, state in states.items():
            pairs = [(r, c) for r, c in zip(round_requests, results) if r["world"] == world]
            if not pairs:
                continue
            until_s = pairs[0][0]["until_s"]
            expected = _prefix(policy_steps(state["incumbent"]), until_s)
            steps = [policy_steps(c) for _, c in pairs]
            state["prefix_mismatches"] += sum(_prefix(s, until_s) != expected for s in steps)
            state["prefix_decisions"] += sum(t < until_s for s in steps for t in s)
            state["simulated_decisions"] += sum(map(len, steps))
            pool = [{**state["incumbent"], "run_index": 0}] + [
                {**c, "run_index": i} for i, (_, c) in enumerate(pairs, start=1)
            ]
            best = min(pool, key=search_score)
            if best["run_index"]:
                request = pairs[best["run_index"] - 1][0]
                state["incumbent"] = best
                state["greedy_from_s"] = float("inf") if request["explore"] else until_s
                state["improvements"].append(
                    {"plan": state["plans"], "until_s": until_s, "explore": request["explore"]}
                )
            state["unchanged_plans"] = 0 if best["run_index"] else state["unchanged_plans"] + 1
            state["plans"] += 1
            state["simulated_episodes"] += len(pairs)
    return states
