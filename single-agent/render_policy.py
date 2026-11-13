"""Render / benchmark a trained policy on the eval scenarios.

Beyond rendering an MP4 this script can:
  * address scenarios by seed: --seed / --seeds pin exactly which scenario is generated
    (via env.reset(options={"scenario_seed": ...})), rather than walking the eval pool;
  * retry a scenario until it is solved (--attempts N): each attempt samples the policy
    afresh, the first solved one wins, and only that attempt is rendered;
  * export the winning action sequence per seed as JSON (--solutions-json), replayable
    exactly (see the "NOT cleaned up" caveat on write_solutions_json);
  * toggle policy stochasticity (--stochastic) vs greedy/deterministic (default);
  * run an eval-time *shield* that refuses a proposed clearance when it would reduce
    reward (--refuse-on-reward-drop) and/or introduce a critical conflict
    (--refuse-on-critical-conflict), falling back to DO_NOTHING or the next-best safe
    clearance (--refuse-fallback); the refusal is a live one-step look-ahead via
    env.peek_action (no episode state is mutated);
  * report the graduated success-tier levels (bonus_tier / success_tier / frac within
    tier thresholds) for a model.

The per-episode table and the aggregate analysis are ALWAYS printed, including after a
Ctrl-C or a crash mid-run, for whatever episodes finished.

Examples:
    # benchmark an agent across all eval seeds, no video:
    python render_policy.py --model experiments/<run>/best/best_model.zip \
        --all-eval-seeds --no-video

    # hunt for a solution on three scenarios, up to 20 stochastic tries each, and render
    # the ones that worked:
    python render_policy.py --model .../best_model.zip --seeds 41,7,123 --attempts 20

    # render with the conflict shield (next-best fallback) and stochastic sampling:
    python render_policy.py --model .../best_model.zip --out shielded.mp4 \
        --refuse-on-critical-conflict --refuse-fallback next_best --stochastic
"""

import argparse
import datetime as _dt
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering

import numpy as np
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed as sb3_set_random_seed

from actions.action_set import (
    check_checkpoint_compatible,
    sync_attention_to_checkpoint,
    sync_obs_width_to_checkpoint,
)
from actions.actions import Action
from atc_env.single_agent_env import ATCSingleAgentEnv
from atc_env.windowed_env import WindowedATCEnv
from config.config import Config
from utils.render_utils import make_interstitial_frames

DO_NOTHING = int(Action.DO_NOTHING)


# --------------------------------------------------------------------------------------
# action plumbing
# --------------------------------------------------------------------------------------

def _predict(model, env, obs, deterministic):
    """Predict a (aircraft, clearance) action for the autoregressive action."""
    action, _ = model.predict(obs, deterministic=deterministic)
    return action


def _flat_to_pair(env, action):
    """Normalise a predicted action to (a_air, a_clr) for the autoregressive action."""
    arr = np.asarray(action).ravel()
    return int(arr[0]), int(arr[1])


def _again_of(action) -> int:
    """The reselection flag of a predicted action (0 for an (aircraft, clearance) action)."""
    arr = np.asarray(action).ravel()
    return int(arr[2]) if arr.size > 2 else 0


def _pair_to_action(env, a_air, a_clr, again=0):
    """Build the env-accepted action object from an (a_air, a_clr) pair, plus the reselection
    flag when the env has one."""
    if len(getattr(env.action_space, "nvec", ())) == 3:
        return np.array([a_air, a_clr, again], dtype=np.int64)
    return np.array([a_air, a_clr], dtype=np.int64)


def _safe_peek(env, a_air, a_clr):
    """env.peek_action, returning None if the candidate can't be interpreted/simulated.

    Some clearances the action mask marks valid still raise inside interpret_action (a latent
    mask/interpreter inconsistency the trained policy simply avoids); a shield that probes
    every clearance must tolerate that rather than crash the run.
    """
    try:
        return env.peek_action(a_air, a_clr)
    except Exception:
        return None


def apply_shield(env, obs, a_air, a_clr, args):
    """Maybe refuse a proposed clearance. Returns (a_clr_final, refusal_reason|None).

    A clearance is refused when an enabled check fires:
      * reward drop: peek reward_delta < -refuse_reward_eps (issuing is worse than DO_NOTHING)
      * critical conflict: the action introduces a near-horizon conflict DO_NOTHING would not
    On refusal we either issue DO_NOTHING or search the aircraft's valid clearances for the
    best one that passes the enabled checks (next_best).
    """
    shielding = args.refuse_on_reward_drop or args.refuse_on_critical_conflict
    if not shielding or a_clr == DO_NOTHING:
        return a_clr, None

    peek = _safe_peek(env, a_air, a_clr)
    if peek is None:
        # The proposed clearance can't be interpreted in this (shield-altered) state; it would
        # also crash env.step(), so refuse it and fall back.
        reason = "interpret_error"
    else:
        drop = args.refuse_on_reward_drop and peek["reward_delta"] < -args.refuse_reward_eps
        conflict = args.refuse_on_critical_conflict and peek["creates_critical_conflict"]
        if not (drop or conflict):
            return a_clr, None  # accepted as proposed
        reason = "reward_drop" if drop else "critical_conflict"

    if args.refuse_fallback == "do_nothing":
        return DO_NOTHING, reason

    # next_best: highest-reward valid clearance that passes the enabled checks (DO_NOTHING floor).
    # peek_clearances shares one DO_NOTHING baseline across all candidates (the expensive path).
    mask = np.asarray(obs["mask_action_per_ac"][a_air], dtype=bool)
    candidates = [int(c) for c in np.flatnonzero(mask) if int(c) != DO_NOTHING]
    try:
        evals = env.peek_clearances(a_air, candidates)
    except Exception:
        evals = {}
    best_clr, best_delta = DO_NOTHING, 0.0
    for clr, pk in evals.items():
        ok = (
            (not args.refuse_on_reward_drop or pk["reward_delta"] >= -args.refuse_reward_eps)
            and (not args.refuse_on_critical_conflict or not pk["creates_critical_conflict"])
        )
        if ok and pk["reward_delta"] > best_delta:
            best_clr, best_delta = clr, pk["reward_delta"]
    return best_clr, reason


# --------------------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------------------

@dataclass
class Episode:
    """One played episode: what happened, and enough to replay it exactly."""

    scenario_seed: int
    reward: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    end_reason: Optional[str] = None
    refusals: Counter = field(default_factory=Counter)
    attempt: int = 1
    # Every step's issued (slot, clearance), including DO_NOTHING and any shield override.
    # Feeding this back through replay_episode() reproduces the episode bit for bit.
    actions: List[Tuple[int, int]] = field(default_factory=list)
    # The same steps annotated for export (callsign, position, reward).
    steps: List[Dict[str, Any]] = field(default_factory=list)
    frames: List[np.ndarray] = field(default_factory=list)

    @property
    def tier(self) -> int:
        return int(self.metrics.get("bonus_tier") or 0)

    @property
    def abs_dev(self) -> float:
        v = self.metrics.get("deviation_abs_final")
        return float(v) if v is not None else float("inf")

    @property
    def n_clearances(self) -> int:
        return sum(1 for act in self.actions if act[1] != DO_NOTHING)

    def rank_key(self):
        """Sort key for "which attempt was best".

        Windowed env: the lexicographic objective -- no loss of separation, then the sum of
        per-flight bracket values, then the FEWEST clearances. 10-aircraft env: highest tier,
        then tightest schedule, then most reward (unchanged)."""
        m = self.metrics
        if "window_bracket_score" in m:
            return (not bool(m.get("had_separation", False)),
                    float(m["window_bracket_score"]),
                    -float(m.get("window_clearances", 0.0)))
        return (self.tier, -self.abs_dev, self.reward)


def _record_step(ep: Episode, env, info: Dict[str, Any], reward: float, step: int) -> None:
    """Append one issued clearance to the episode record.

    Read back out of ``info`` rather than from what we asked for: the env reports the slot,
    callsign and clearance it actually executed, which is what a replay has to reproduce.
    """
    a_air, a_clr = int(info["aircraft_index"]), int(info["clearance"])
    callsign = info.get("callsign")
    # (slot, clearance), plus the again flag under reselection, as executed.
    ep.actions.append((a_air, a_clr, int(info["again"])) if "again" in info else (a_air, a_clr))

    pos = None
    try:
        ac = env.simulator.world.get_aircraft_by_callsign(str(callsign))
        p = ac.state.position_nm_nm_ft
        pos = (round(float(p.x), 2), round(float(p.y), 2), round(float(p.z), 0))
    except Exception:
        pass  # landed/left the world this step; the clearance itself is still the record

    ep.steps.append({
        "step": int(step),
        "t_s": float(info.get("time_s", step * Config.TIME_BETWEEN_ACTIONS)),
        "slot": a_air,
        "callsign": callsign,
        "clearance": a_clr,
        "action": Action(a_clr).name,
        "x_nm": pos[0] if pos else None,
        "y_nm": pos[1] if pos else None,
        "alt_ft": pos[2] if pos else None,
        "reward": round(float(reward), 4),
    })


def play_episode(
    model,
    env,
    args,
    scenario_seed: int,
    record_frames: bool,
    replay_actions: Optional[List[Tuple[int, int]]] = None,
    paired_attempt: Optional[int] = None,
) -> Episode:
    """Play one episode of ``scenario_seed`` and return the record.

    With ``replay_actions`` the policy is not consulted at all: the given (slot, clearance)
    pairs are issued in order. Scenario generation and the simulator are both deterministic
    given the seed, so a replay reproduces the original episode exactly -- which is what lets
    the attempt search run cheaply without frames and render only the winner.
    """
    deterministic = not args.stochastic
    if paired_attempt is not None:
        # Seeded exactly like analysis/score_windowed.py --attempts: the frame pinned by the
        # scenario seed, attempt 0 deterministic, attempt a >= 1 sampled with its own torch seed.
        # The rendered episode is then the one the analysis scored.
        import torch

        np.random.seed(int(scenario_seed) % (2**32))
        torch.manual_seed((int(scenario_seed) * 1009 + int(paired_attempt)) % (2**31))
        deterministic = int(paired_attempt) == 0
        obs, info = env.reset(seed=int(scenario_seed), options={"scenario_seed": int(scenario_seed)})
    else:
        obs, info = env.reset(seed=args.rng_seed, options={"scenario_seed": int(scenario_seed)})
    ep = Episode(scenario_seed=int(scenario_seed))
    if record_frames:
        ep.frames.append(env.render())

    done = False
    steps = 0
    # Per-scenario horizon: env truncates at env.max_steps_per_episode; cap as a safety net.
    # Reselection: up to max_picks decisions per clock step.
    max_steps = (int(env.max_steps_per_episode) + 2) * max(1, int(getattr(env, "_max_picks", 1)))
    while not done and steps < max_steps:
        if replay_actions is not None:
            if steps >= len(replay_actions):
                break
            a_air, a_clr = replay_actions[steps][:2]
            again = int(replay_actions[steps][2]) if len(replay_actions[steps]) > 2 else 0
        else:
            action = _predict(model, env, obs, deterministic)
            a_air, a_clr = _flat_to_pair(env, action)
            again = _again_of(action)
            a_clr, reason = apply_shield(env, obs, a_air, a_clr, args)
            if reason is not None:
                ep.refusals[reason] += 1
        try:
            obs, reward, terminated, truncated, info = env.step(_pair_to_action(env, a_air, a_clr, again))
        except Exception:
            # Safety net: a mask-valid clearance that interpret_action can't execute in this
            # (off-distribution) state. interpret_action runs before any world mutation, so
            # retrying with DO_NOTHING for this aircraft is safe.
            ep.refusals["interpret_error"] += 1
            obs, reward, terminated, truncated, info = env.step(
                _pair_to_action(env, a_air, DO_NOTHING)
            )
        ep.reward += float(reward)
        _record_step(ep, env, info, float(reward), steps)
        if record_frames:
            ep.frames.append(env.render())
        done = terminated or truncated
        steps += 1

    ep.metrics = info.get("episode_metrics", {}) or {}
    ep.end_reason = info.get("end_reason")
    return ep


def solve_seed_paired(model, env, args, scenario_seed: int) -> Tuple[Episode, int]:
    """Attempt 0 (deterministic) plus args.paired_attempts sampled attempts, seeded as the
    scorer does; keep the best by Episode.rank_key (for the windowed env: no loss of separation,
    then bracket score, then fewest clearances). Never renders; the winner is replayed."""
    best: Optional[Episode] = None
    for attempt in range(0, int(args.paired_attempts) + 1):
        ep = play_episode(model, env, args, scenario_seed, record_frames=False, paired_attempt=attempt)
        ep.attempt = attempt
        m = ep.metrics
        print(f"    attempt {attempt}{' (deterministic)' if attempt == 0 else ''}: end={ep.end_reason} "
              f"on_time={m.get('window_on_time_rate', float('nan')):.2f} bracket={m.get('window_bracket_score', float('nan')):.1f} "
              f"clearances={m.get('window_clearances', float('nan')):.0f}")
        if best is None or ep.rank_key() > best.rank_key():
            best = ep
    assert best is not None
    print(f"    best: attempt {best.attempt}")
    return best, int(args.paired_attempts) + 1


def solve_seed(model, env, args, scenario_seed: int, solved_tier: int) -> Tuple[Episode, int]:
    """Play ``scenario_seed`` up to ``args.attempts`` times, stopping at the first solution.

    Returns (chosen_episode, attempts_used). "Solved" is bonus_tier >= ``solved_tier``. When
    no attempt clears the bar the best one by :meth:`Episode.rank_key` is returned, so the
    render still shows this scenario's closest approach to a solution rather than an
    arbitrary try.

    The search never renders: frames are the expensive part and only the winner needs them.
    """
    best: Optional[Episode] = None
    for attempt in range(1, int(args.attempts) + 1):
        ep = play_episode(model, env, args, scenario_seed, record_frames=False)
        ep.attempt = attempt
        # --all-attempts: never stop at the first solution. Under the lexicographic objective a
        # later solved attempt with fewer clearances is strictly better.
        solved = ep.tier >= solved_tier and not getattr(args, "all_attempts", False)
        print(f"    attempt {attempt}/{args.attempts}: tier={ep.tier} "
              f"|dev|={ep.abs_dev:.0f}s end={ep.end_reason}"
              + ("  <- solved" if solved else ""))
        if best is None or ep.rank_key() > best.rank_key():
            best = ep
        if solved:
            return ep, attempt
    assert best is not None
    print(f"    no solution in {args.attempts} attempts; keeping the best "
          f"(attempt {best.attempt}, tier {best.tier})")
    return best, int(args.attempts)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

_EP_ROW_FMT = (
    "{n:>3} {seed:>10} {att:>7} {steps:>5} {reward:>9} {end:<12} {tier:>4} "
    "{solved:>6} {dev:>8} {maxdev:>8} {minsep:>7} {clr:>4}"
)
_EP_HEADER = _EP_ROW_FMT.format(
    n="#", seed="seed", att="attempt", steps="steps", reward="reward", end="end_reason",
    tier="tier", solved="solved", dev="|dev| s", maxdev="max dev", minsep="min sep", clr="clr",
)


def _episode_row(i: int, ep: Episode, attempts_used: int, solved_tier: int) -> str:
    m = ep.metrics
    maxdev = m.get("success_max_aircraft_dev")
    minsep = m.get("min_separation_nm")
    return _EP_ROW_FMT.format(
        n=i,
        seed=ep.scenario_seed,
        att=f"{ep.attempt}/{attempts_used}",
        steps=int(m.get("episode_steps") or len(ep.actions)),
        reward=f"{ep.reward:.1f}",
        end=str(ep.end_reason),
        tier=ep.tier,
        solved="yes" if ep.tier >= solved_tier else "no",
        dev="-" if ep.abs_dev == float("inf") else f"{ep.abs_dev:.0f}",
        maxdev="-" if maxdev is None else f"{float(maxdev):.0f}",
        minsep="-" if minsep is None else f"{float(minsep):.1f}",
        clr=ep.n_clearances,
    )


def _stat_line(label: str, values: List[float], fmt: str = "{:.1f}") -> str:
    """'label : mean X  median X  min X  max X' over the given values."""
    if not values:
        return f"{label:<22}: (none)"
    parts = [
        ("mean", statistics.fmean(values)),
        ("median", statistics.median(values)),
        ("min", min(values)),
        ("max", max(values)),
    ]
    return f"{label:<22}: " + "  ".join(f"{k} {fmt.format(v)}" for k, v in parts)


def print_analysis(episodes: List[Episode], attempts_used: Dict[int, int], args,
                   solved_tier: int, shielding: bool) -> None:
    """Print the per-episode table and the aggregate analysis.

    Called from a `finally`, so it must tolerate a partial run: every aggregate is computed
    over whatever episodes actually finished.
    """
    print("\n=== episodes ===")
    print(_EP_HEADER)
    print("-" * len(_EP_HEADER))
    for i, ep in enumerate(episodes, start=1):
        print(_episode_row(i, ep, attempts_used.get(ep.scenario_seed, 1), solved_tier))

    n = len(episodes)
    if n == 0:
        print("\n(no episodes completed)")
        return

    tiers = Counter(ep.tier for ep in episodes)
    solved = [ep for ep in episodes if ep.tier >= solved_tier]
    devs = [ep.abs_dev for ep in episodes if ep.abs_dev != float("inf")]
    maxdevs = [float(ep.metrics["success_max_aircraft_dev"]) for ep in episodes
               if ep.metrics.get("success_max_aircraft_dev") is not None]
    minseps = [float(ep.metrics["min_separation_nm"]) for ep in episodes
               if ep.metrics.get("min_separation_nm") is not None]
    end_reasons = Counter(str(ep.end_reason) for ep in episodes)
    clearances = [float(ep.n_clearances) for ep in episodes]

    clearance_mix: Counter = Counter()
    for ep in episodes:
        for act in ep.actions:
            clr = act[1]
            if clr != DO_NOTHING:
                clearance_mix[Action(clr).name] += 1

    total_refusals: Counter = Counter()
    for ep in episodes:
        total_refusals.update(ep.refusals)

    print("\n=== analysis ===")
    print(f"{'episodes':<22}: {n}")
    print(f"{'solved (tier >= ' + str(solved_tier) + ')':<22}: "
          f"{len(solved)} / {n} ({len(solved) / n:.0%})")
    print(f"{'bonus_tier histogram':<22}: "
          + "  ".join(f"t{t}={tiers.get(t, 0)}" for t in range(0, 6)))
    print(f"{'mean bonus_tier':<22}: {statistics.fmean([ep.tier for ep in episodes]):.2f}")
    print(_stat_line("final |deviation| (s)", devs, "{:.0f}"))
    print(_stat_line("worst aircraft dev (s)", maxdevs, "{:.0f}"))
    print(_stat_line("min separation (nm)", minseps, "{:.2f}"))
    print(_stat_line("clearances / episode", clearances, "{:.1f}"))
    print(f"{'end reasons':<22}: {dict(end_reasons)}")
    print(f"{'clearance mix':<22}: "
          + (", ".join(f"{k}={v}" for k, v in clearance_mix.most_common()) or "(none issued)"))

    if int(args.attempts) > 1:
        used = [float(attempts_used.get(ep.scenario_seed, 1)) for ep in episodes]
        print(_stat_line("attempts per seed", used, "{:.1f}"))
        won_on = Counter(ep.attempt for ep in solved)
        print(f"{'solved on attempt':<22}: "
              + (", ".join(f"#{k}x{v}" for k, v in sorted(won_on.items())) or "(none solved)"))
    if shielding:
        print(f"{'refusals':<22}: {dict(total_refusals) if total_refusals else 0}")

    unsolved = [ep.scenario_seed for ep in episodes if ep.tier < solved_tier]
    if unsolved:
        print(f"{'unsolved seeds':<22}: "
              + ", ".join(str(s) for s in unsolved[:30])
              + (f" (+{len(unsolved) - 30} more)" if len(unsolved) > 30 else ""))


# --------------------------------------------------------------------------------------
# solution export
# --------------------------------------------------------------------------------------

def _jsonable(o):
    """Fallback encoder for the numpy scalars the env's metrics dicts carry."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_solutions_json(path: str, episodes: List[Episode], attempts_used: Dict[int, int],
                         args, solved_tier: int) -> None:
    """Write the per-seed action sequences that were actually played.

    These are RAW policy output, deliberately: every step is recorded, DO_NOTHING included,
    so ``raw_actions`` replays the episode exactly (see play_episode's replay_actions). That
    is the right thing to persist, but it is NOT the right thing to hand to a controller --
    the sequences are long and full of redundancy (consecutive speed-ups and slow-downs on the
    same aircraft that net out, clearances with no effect on the outcome). Minimising a
    solution while keeping it working is a separate pass over this file; ``cleanup.applied``
    records that it has not run.
    """
    payload = {
        "schema": "tada-single-agent-solutions/1",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "policy": "stochastic" if args.stochastic else "deterministic",
        "solved_tier": int(solved_tier),
        "attempts_allowed": int(args.attempts),
        "cleanup": {
            "applied": False,
            "note": "raw played actions; consecutive speed changes are NOT accumulated and "
                    "no-op clearances are NOT filtered out",
        },
        "env": {
            "time_between_actions_s": float(Config.TIME_BETWEEN_ACTIONS),
            "difficulty": str(Config.DEFAULT_EVALUATION_DIFFICULTY),
            "n_clearances": int(Config.N_CLEARANCES),
            "n_aircraft_max": int(Config.N_AIRCRAFT_MAX),
            "clearance_names": {int(a): a.name for a in Action},
        },
        "solutions": [],
    }

    for ep in episodes:
        commands = [s for s in ep.steps if s["clearance"] != DO_NOTHING]
        per_aircraft: Dict[str, List[Dict[str, Any]]] = {}
        for cmd in commands:
            per_aircraft.setdefault(str(cmd["callsign"]), []).append(cmd)
        payload["solutions"].append({
            "scenario_seed": ep.scenario_seed,
            "solved": bool(ep.tier >= solved_tier),
            "bonus_tier": ep.tier,
            "end_reason": ep.end_reason,
            "reward": round(float(ep.reward), 4),
            "attempts_used": int(attempts_used.get(ep.scenario_seed, 1)),
            "chosen_attempt": int(ep.attempt),
            "n_steps": len(ep.actions),
            "n_clearances": ep.n_clearances,
            "metrics": ep.metrics,
            # Grouped by aircraft: the view a cleanup pass wants, since accumulating
            # consecutive speed changes is a per-aircraft operation.
            "per_aircraft": per_aircraft,
            "commands": commands,
            "raw_actions": [[int(x) for x in act] for act in ep.actions],
        })

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)
    n_solved = sum(1 for s in payload["solutions"] if s["solved"])
    print(f"wrote {path} ({len(payload['solutions'])} seeds, {n_solved} solved)")


# --------------------------------------------------------------------------------------
# output naming
# --------------------------------------------------------------------------------------

def _model_tag(model_path):
    """Short, descriptive id for the model: the run dir (atc_run_*) if the model lives under one
    (e.g. run_best_model), otherwise just the file stem."""
    parts = os.path.normpath(model_path).split(os.sep)
    stem = os.path.splitext(parts[-1])[0] if parts else "model"
    run = next((p for p in parts if p.startswith("atc_run")), None)
    return f"{run}_{stem}" if run else stem


def _auto_out_name(args, seeds, shielding, ext=".mp4"):
    """Build '<model>_<param>_<param>...<ext>' from the run config (used when --out is omitted)."""
    tokens = [_model_tag(args.model), "stoch" if args.stochastic else "det"]
    if shielding:
        if args.refuse_on_reward_drop:
            tokens.append("rdrop")
        if args.refuse_on_critical_conflict:
            tokens.append("conflict")
        tokens.append(args.refuse_fallback)
    else:
        tokens.append("noshield")
    if args.all_eval_seeds:
        tokens.append("alleval")
    elif len(seeds) == 1:
        tokens.append(f"seed{seeds[0]}")
    else:
        tokens.append(f"ep{len(seeds)}")
    if int(args.attempts) > 1:
        tokens.append(f"try{args.attempts}")
    return "_".join(tokens) + ext


def _unique_path(path):
    """Return path, or path with _0/_1/... inserted before the extension if it already exists."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 0
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


def resolve_seeds(args) -> List[int]:
    """Which scenario seeds to play, in order.

    --seeds wins, then --seed (repeated --episodes times so one scenario can be replayed),
    then --all-eval-seeds, then the first --episodes entries of the eval pool.
    """
    if args.seeds:
        return [int(s) for s in str(args.seeds).replace(",", " ").split()]
    if args.seed is not None:
        return [int(args.seed)] * max(1, int(args.episodes))
    pool = [int(s) for s in Config.SCENARIO_EVAL_SEEDS]
    if args.all_eval_seeds:
        return pool
    return pool[: max(1, int(args.episodes))]


# --------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a saved policy .zip")
    ap.add_argument("--out", default=None,
                    help="output mp4; if omitted, auto-named <model>_<params>.mp4 in the cwd "
                         "(with _0/_1/... appended on collision)")
    # Scenario selection
    ap.add_argument("--episodes", type=int, default=1,
                    help="number of eval-pool scenarios to play (in pool order)")
    ap.add_argument("--seed", type=int, default=None,
                    help="pin the SCENARIO seed: generate exactly this scenario instead of "
                         "walking the eval pool (repeated --episodes times)")
    ap.add_argument("--seeds", default=None,
                    help="comma/space separated list of scenario seeds to play")
    ap.add_argument("--all-eval-seeds", action="store_true",
                    help="run every Config.SCENARIO_EVAL_SEEDS scenario (overrides --episodes)")
    ap.add_argument("--rng-seed", type=int, default=None,
                    help="seed the sampling RNGs (policy sampling and the observation "
                         "translation matrix) to make the whole run reproducible. Independent "
                         "of the scenario seed. Attempts within a run still differ from each "
                         "other -- the RNG advances -- but the run as a whole repeats exactly")
    # Attempts / solution export
    ap.add_argument("--attempts", type=int, default=1,
                    help="stochastic attempts per scenario seed, stopping at the first solved "
                         "one; the solved (or best) attempt is the one rendered")
    ap.add_argument("--paired-attempts", type=int, default=None, metavar="K",
                    help="attempt 0 deterministic + K sampled attempts, seeded exactly like "
                         "analysis/score_windowed.py --attempts K; render the best by the objective")
    ap.add_argument("--all-attempts", action="store_true",
                    help="play every attempt and keep the best by Episode.rank_key, instead of "
                         "stopping at the first solved one")
    ap.add_argument("--solved-tier", type=int, default=5,
                    help="bonus_tier that counts as solved for --attempts (default 5 = the "
                         "full success criterion)")
    ap.add_argument("--solutions-json", default=None,
                    help="where to write the per-seed action sequences; auto-named next to "
                         "the mp4 if omitted, 'none' to skip")
    # Video
    ap.add_argument("--env", choices=["default", "windowed"], default="default",
                    help="'windowed': the 20-flight stream env (atc_env/windowed_env.py)")
    ap.add_argument("--flights", type=int, default=None,
                    help="windowed env only: flights per scenario (default: WindowedConfig's 20)")
    ap.add_argument("--use-case", type=int, choices=(1, 2), default=None,
                    help="windowed: scenario, overriding the run's: 1 = MXP trombone, 2 = BGY point merge")
    ap.add_argument("--segments", type=int, default=1,
                    help="windowed env only: stitch this many generated scenarios per stream")
    ap.add_argument("--gap", type=float, nargs=2, default=(120.0, 900.0), metavar=("MIN", "MAX"),
                    help="windowed env only: cooling gap between stitched segments, seconds")
    ap.add_argument("--approach-view", choices=["time", "3d"], default="time",
                    help="top-right panel: 'time' = altitude-vs-time side view of the predicted "
                         "traffic with infringements marked (default); '3d' = the 3D approach view")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--no-video", action="store_true", help="skip frame rendering (fast benchmark)")
    ap.add_argument("--predict-points", type=int, default=12,
                    help="samples along each aircraft's predicted trajectory drawn on the map "
                         "and the 3D view (0 disables; costs ~one extra rollout per frame)")
    ap.add_argument("--approach-zoom-nm", type=float, default=25.0,
                    help="half-width (nm) of the 3D approach view around the airport")
    # Policy stochasticity
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions from the policy instead of greedy/deterministic")
    # Eval-time clearance shield
    ap.add_argument("--refuse-on-reward-drop", action="store_true",
                    help="refuse a clearance whose one-step reward is worse than doing nothing")
    ap.add_argument("--refuse-reward-eps", type=float, default=0.0,
                    help="tolerance: refuse only if reward_delta < -eps (default 0.0)")
    ap.add_argument("--refuse-on-critical-conflict", action="store_true",
                    help="refuse a clearance that introduces a critical near-horizon conflict")
    ap.add_argument("--refuse-fallback", choices=["do_nothing", "next_best"], default="do_nothing",
                    help="what to issue when a clearance is refused")
    # Between-episode black title cards
    ap.add_argument("--interstitial-frames", type=int, default=3,
                    help="number of black title-card frames inserted before each episode "
                         "(0 disables). Shows the episode's chosen parameters in white text.")
    ap.add_argument("--success-hold-frames", type=int, default=8,
                    help="extra copies of the final frame held when an episode ends in success, "
                         "so the video pauses on the solved state (0 disables).")
    return ap


def main():
    args = build_parser().parse_args()

    # Retrying a scenario only means anything if the attempts can differ. A greedy policy
    # replays the same episode every time (the scenario is a pure function of its seed), so
    # --attempts implies sampling unless the caller pinned --rng-seed on purpose.
    if int(args.attempts) > 1 and not args.stochastic:
        args.stochastic = True
        print("[render] --attempts > 1: enabling --stochastic (a deterministic policy would "
              "replay the same episode every attempt)")

    # Two RNGs sit outside the scenario seed and both change what a run produces: numpy draws
    # the per-episode observation translation matrix (which moves a deterministic policy's
    # output), and torch draws the stochastic policy's samples. Seeding only numpy -- as this
    # script used to -- leaves a --stochastic run unreproducible.
    if args.rng_seed is not None:
        sb3_set_random_seed(int(args.rng_seed))

    seeds = resolve_seeds(args)
    solved_tier = int(args.solved_tier)
    shielding = args.refuse_on_reward_drop or args.refuse_on_critical_conflict

    check_checkpoint_compatible(args.model)
    sync_attention_to_checkpoint(args.model)
    sync_obs_width_to_checkpoint(args.model)  # sequence-feature checkpoints are 16 wide
    model = PPO.load(args.model, device="cpu")
    env_config = {
        "evaluation_mode": True,  # per-scenario horizon
        "render_predict_points": 0 if args.no_video else int(args.predict_points),
        "render_approach_zoom_nm": float(args.approach_zoom_nm),
        "render_approach_view": args.approach_view,
    }
    if args.env == "windowed":
        from analysis.lookahead import env_config_for

        env_config.update(env_config_for(args.model))  # the MDP the run trained in (e.g. release)
        if getattr(args, "use_case", None):
            env_config["use_case"] = int(args.use_case)  # e.g. zero-shot on point merge
        env_config["stitch_segments"] = int(args.segments)
        env_config["max_episode_steps_cap"] = 100_000  # every stream to its own horizon, as score_windowed
        env_config["stitch_gap_s"] = tuple(args.gap)
        if args.flights:
            env_config["scenario_aircraft_count"] = int(args.flights)
        env = WindowedATCEnv(config=env_config)
    else:
        env = ATCSingleAgentEnv(config=env_config)

    print(f"[render] model={args.model} seeds={len(seeds)} attempts={args.attempts} "
          f"stochastic={args.stochastic} shield={shielding} fallback={args.refuse_fallback}")

    # Human-readable shield description for the interstitial cards.
    if shielding:
        _checks = []
        if args.refuse_on_reward_drop:
            _checks.append("reward-drop")
        if args.refuse_on_critical_conflict:
            _checks.append("conflict")
        shield_desc = f"{'+'.join(_checks)} -> {args.refuse_fallback}"
    else:
        shield_desc = "none"

    out_path = None
    writer = None
    if not args.no_video:
        out_path = args.out if args.out else _unique_path(_auto_out_name(args, seeds, shielding))
        # Stream frames straight to the container: a full episode is hundreds of 2025x1350
        # frames (~8 MB each), so holding every episode's frames to write once at the end
        # costs gigabytes and grows with --episodes and --attempts.
        writer = imageio.get_writer(out_path, fps=args.fps)

    episodes: List[Episode] = []
    attempts_used: Dict[int, int] = {}
    n_frames = 0

    try:
        for i, scenario_seed in enumerate(seeds):
            print(f"\n[{i + 1}/{len(seeds)}] seed {scenario_seed}")

            if args.paired_attempts is not None:
                ep, used = solve_seed_paired(model, env, args, scenario_seed)
                attempts_used[scenario_seed] = used
                if not args.no_video:
                    ep = _carry_over(ep, play_episode(
                        model, env, args, scenario_seed, record_frames=True,
                        replay_actions=ep.actions, paired_attempt=ep.attempt,
                    ))
            elif int(args.attempts) > 1:
                ep, used = solve_seed(model, env, args, scenario_seed, solved_tier)
                attempts_used[scenario_seed] = used
                if not args.no_video:
                    # Replay the chosen attempt to get its frames. Deterministic given the
                    # seed and the action sequence, so this is the same episode, not a
                    # re-roll: only now do we pay for rendering.
                    ep = _carry_over(ep, play_episode(
                        model, env, args, scenario_seed,
                        record_frames=True, replay_actions=ep.actions,
                    ))
            else:
                attempts_used[scenario_seed] = 1
                ep = play_episode(model, env, args, scenario_seed,
                                  record_frames=not args.no_video)

            episodes.append(ep)
            m = ep.metrics
            print(f"  steps={m.get('episode_steps')} reward={ep.reward:.1f} "
                  f"end={ep.end_reason} bonus_tier={ep.tier} "
                  f"success={m.get('success')} abs_dev={m.get('deviation_abs_final')} "
                  f"clearances={ep.n_clearances} "
                  f"attempt={ep.attempt}/{attempts_used[scenario_seed]} "
                  f"refusals={dict(ep.refusals) if ep.refusals else 0}")

            if writer is not None and ep.frames:
                n_frames += _write_episode(writer, ep, i, len(seeds), args, shield_desc,
                                           solved_tier)
            # Frames are the memory hog; the record we keep for the analysis is small.
            ep.frames = []
    except KeyboardInterrupt:
        print("\n[render] interrupted — reporting on the episodes that finished")
    finally:
        if writer is not None:
            writer.close()
        print_analysis(episodes, attempts_used, args, solved_tier, shielding)
        if out_path and n_frames:
            print(f"\nwrote {out_path} ({n_frames} frames @ {args.fps} fps)")
        skip_json = str(args.solutions_json or "").lower() == "none"
        if episodes and not skip_json:
            json_path = args.solutions_json or _unique_path(
                _auto_out_name(args, seeds, shielding, ext="_solutions.json")
            )
            write_solutions_json(json_path, episodes, attempts_used, args, solved_tier)


def _carry_over(searched: Episode, rendered: Episode) -> Episode:
    """Keep the searched attempt's identity (which attempt number won) on its rendered twin.

    Also checks that the replay actually reproduced the attempt. It always has so far -- the
    scenario is a pure function of its seed and the simulator is deterministic -- but the
    whole search-then-replay design rests on that, and a silent divergence would mean the
    rendered video and the exported solution describe a different episode than the one the
    search scored. Cheap to check, so check it.
    """
    rendered.attempt = searched.attempt
    rendered.refusals = searched.refusals
    if (rendered.actions != searched.actions
            or abs(rendered.reward - searched.reward) > 1e-6
            or rendered.end_reason != searched.end_reason):
        print(f"  [warn] replay of seed {searched.scenario_seed} diverged from the searched "
              f"attempt (reward {searched.reward:.3f} -> {rendered.reward:.3f}, "
              f"end {searched.end_reason} -> {rendered.end_reason}); "
              f"the video and JSON describe the replay")
    return rendered


def _write_episode(writer, ep: Episode, index: int, total: int, args, shield_desc: str,
                   solved_tier: int) -> int:
    """Append one episode's title card + frames to the video. Returns frames written."""
    written = 0
    if args.interstitial_frames > 0:
        h, w = ep.frames[0].shape[:2]
        card_fields = [
            ("scenario seed", ep.scenario_seed),
            ("policy", "stochastic" if args.stochastic else "deterministic"),
            ("shield", shield_desc),
            ("interval", f"{int(Config.TIME_BETWEEN_ACTIONS)}s"),
            ("attempt", f"{ep.attempt}"),
            ("outcome", f"{ep.end_reason} (tier {ep.tier}/5)"),
        ]
        if "window_n_flights" in ep.metrics:  # windowed env: the per-flight outcome
            m = ep.metrics
            n = int(m["window_n_flights"])
            card_fields.append(("flights on time",
                                f"{round(float(m['window_on_time_rate']) * n)}/{n} "
                                f"(landed {int(m['window_n_landed'])})"))
        # Pad labels to a common width so the colons line up (robust to linters that
        # would collapse hand-inserted padding inside the f-strings).
        label_w = max(len(label) for label, _ in card_fields)
        card_lines = [f"{label:<{label_w}} : {value}" for label, value in card_fields]
        for frame in make_interstitial_frames(
            w, h, f"Episode {index + 1} / {total}", card_lines,
            n_frames=args.interstitial_frames,
        ):
            writer.append_data(frame)
            written += 1

    for frame in ep.frames:
        writer.append_data(frame)
        written += 1

    # Hold on a solved episode's final frame so it's easy to pause on.
    if args.success_hold_frames > 0 and ep.tier >= solved_tier:
        for _ in range(int(args.success_hold_frames)):
            writer.append_data(ep.frames[-1])
            written += 1
    return written


if __name__ == "__main__":
    main()
