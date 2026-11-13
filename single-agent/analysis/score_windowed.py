"""Score supplied policies in the windowed environment with explicit scenario and
observation-frame seeds. Supports per-flight records and sampled attempts. Configure
model paths, scenario inputs and runtime metadata explicitly."""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODELS = None
_ENV = None
_SHIELD = None  # argparse-like namespace for render_policy.apply_shield, or None
_VETO = None  # analysis.forecast_veto.ForecastVeto (the k-step window shield), or None
_ATTEMPTS = 0   # extra stochastic attempts per seed (attempt 0 is always the deterministic policy)
_LOOKAHEAD = []  # one analysis.lookahead.LookaheadAgent per model when --lookahead K > 1
_OBSERVER = None  # optional episode observer (analysis/evaluation_records.py): begin/before_step/after_step/end; read-only
HARD_SOLVED_DEV_S = 30.0  # the hard-solved target: every flight within +/-30 s (solved: +/-60 s)
DO_NOTHING_MODEL = "do-nothing"  # --models do-nothing: the no-clearance floor on the same streams


def _init_worker(model_paths: list[str], n_flights: int, shield: bool, segments: int = 1,
                 gap_s: tuple = (120.0, 900.0), attempts: int = 0, lookahead: int = 0,
                 use_case: int | None = None) -> None:
    global _MODELS, _ENV, _SHIELD, _ATTEMPTS, _LOOKAHEAD
    _ATTEMPTS = int(attempts)
    import torch

    torch.set_num_threads(1)  # one process per core; torch threads would oversubscribe
    from stable_baselines3 import PPO

    from actions.action_set import check_checkpoint_compatible
    from atc_env.windowed_env import WindowedATCEnv

    _MODELS = []
    for path in model_paths:
        if path == DO_NOTHING_MODEL:
            _MODELS.append(None)  # the floor: never a clearance
            continue
        check_checkpoint_compatible(path)
        _MODELS.append(PPO.load(path, device="cpu"))
    reward_mode = {}
    if int(lookahead) > 1:
        from analysis.lookahead import LookaheadAgent, reward_config_for
        from config.config import Config as _C

        _LOOKAHEAD = [LookaheadAgent(m, pth, int(lookahead), float(_C.PBRS_GAMMA))
                      for m, pth in zip(_MODELS, model_paths)]
        # The lookahead scores r + gamma V(s'), so r must be the reward the critic was trained
        # on (mode and bust penalty). Only the reward depends on this; every scored metric is
        # the same whatever the reward.
        cfgs = {tuple(sorted(reward_config_for(pth).items())) for pth in model_paths}
        if len(cfgs) > 1:
            raise SystemExit(f"--lookahead: the models were trained on different rewards {sorted(cfgs)}; "
                             "score them separately")
        reward_mode = dict(cfgs.pop())
    # The MDP each run trained in (e.g. release of locked flights changes the window): the models
    # scored together must share it, and the env is built with it.
    from analysis.lookahead import env_config_for

    envs = {tuple(sorted(env_config_for(pth).items())) for pth in model_paths}
    if len(envs) > 1:
        raise SystemExit(f"the models were trained in different envs {sorted(envs)}; score them separately")
    env_cfg = dict(envs.pop())
    if use_case is not None:
        env_cfg["use_case"] = int(use_case)  # e.g. zero-shot: an MXP-trained model on point merge
    _ENV = WindowedATCEnv({
        **env_cfg,
        **reward_mode,
        "evaluation_mode": True,
        "scenario_aircraft_count": int(n_flights),
        # Stitched streams: `segments` generated scenarios of n_flights each, joined by a
        # random cooling gap (deterministic per seed).
        "stitch_segments": int(segments),
        "stitch_gap_s": tuple(gap_s),
        # Evaluation runs every stream to its own horizon (last scheduled landing + margin); the
        # training cap (200 steps per segment) is shorter than some point-merge streams.
        "max_episode_steps_cap": 100_000,
    })
    if shield:
        from types import SimpleNamespace

        # render_policy's conflict shield: refuse a clearance that introduces a near-horizon
        # conflict DO_NOTHING would not, and issue the best clearance that passes instead.
        _SHIELD = SimpleNamespace(
            refuse_on_reward_drop=False, refuse_reward_eps=0.0,
            refuse_on_critical_conflict=True, refuse_fallback="next_best",
        )


def _horizon_capped(env) -> bool:
    """True if the episode-length cap ends the episode before the stream's own horizon (last scheduled
    landing + margin): flights still airborne then are the evaluator's doing, not the agent's."""
    from config.config import Config
    from config.windowed_config import WindowedConfig

    natural = math.ceil(env.episode_time_horizon / float(Config.TIME_BETWEEN_ACTIONS)) + int(WindowedConfig.HORIZON_STEP_MARGIN)
    return env.max_steps_per_episode < natural


def _score_seed(seed: int) -> tuple[list[dict], list[dict]]:
    import torch

    rows, flights = [], []
    for mi, model in enumerate(_MODELS):
        for attempt in range(1 + _ATTEMPTS):
            # Attempt 0 is the deterministic policy, as deployed. Attempts 1.. sample from it,
            # each with its own torch seed, on the SAME scenario and observation frame.
            deterministic = attempt == 0
            torch.manual_seed((int(seed) * 1009 + attempt) % (2**31))
            np.random.seed(int(seed) % (2**32))  # pins the frame -> paired across models/attempts
            obs, _ = _ENV.reset(seed=int(seed), options={"scenario_seed": int(seed)})
            if _OBSERVER is not None:
                _OBSERVER.begin(_ENV, obs, int(seed), attempt)
            done, steps, ret = False, 0, 0.0
            # Safety net only: the env truncates at its own horizon; zero-time picks are extra calls.
            max_steps = (int(_ENV.max_steps_per_episode) + 2) * max(1, int(getattr(_ENV, "_max_picks", 1)))
            info: dict = {}
            refusals = vetoes = 0
            while not done and steps < max_steps:
                if model is None:
                    slot = next(i for i, cs in enumerate(_ENV._prev_aircraft_order) if cs)
                    action = np.array([slot, 0, 0] if "mask_select" in obs else [slot, 0])  # DO_NOTHING
                elif deterministic and _LOOKAHEAD:
                    action = _LOOKAHEAD[mi].act(_ENV, obs)  # critic-guided one-step lookahead
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                if _SHIELD is not None:
                    from render_policy import apply_shield

                    flat = np.asarray(action).ravel()
                    a_air, a_clr = int(flat[0]), int(flat[1])
                    a_clr_final, reason = apply_shield(_ENV, obs, a_air, a_clr, _SHIELD)
                    refusals += int(reason is not None)
                    action = flat.astype(np.int64).copy()  # keeps a reselection flag, if any
                    action[1] = a_clr_final
                if _VETO is not None and model is not None:
                    action, override = _VETO(_ENV, obs, action)
                    vetoes += int(override is not None)
                if _OBSERVER is not None:
                    _OBSERVER.before_step(_ENV, obs, action)
                obs, reward, terminated, truncated, info = _ENV.step(action)
                if _OBSERVER is not None:
                    _OBSERVER.after_step(_ENV, info)
                done = terminated or truncated
                ret += float(reward)
                steps += 1
            if _OBSERVER is not None:
                _OBSERVER.end(_ENV, info)
            m = info.get("episode_metrics") or {}
            rows.append(
                {
                    "model_idx": mi,
                    "attempt": attempt,
                    "seed": int(seed),
                    "on_time": float(m.get("window_on_time_rate", 0.0)),
                    "sep": bool(m.get("had_separation", False)),
                    "all_on_time": bool(m.get("success", False)),
                    "landed": float(m.get("window_n_landed", 0.0)),
                    "n_flights": float(m.get("window_n_flights", 0.0)),
                    "mean_dev": float(m.get("window_mean_abs_landing_dev", 0.0)),
                    "max_dev": float(m.get("window_max_abs_landing_dev", 0.0)),
                    "end_reason": str(m.get("end_reason")),
                    "steps": steps,
                    "ep_return": ret,
                    "refusals": refusals,
                    "veto_overrides": vetoes,
                    "aman_order": float(m.get("window_aman_order_rate", 0.0)),
                    "swaps": float(m.get("window_sequence_swaps", 0.0)),
                    "clean_stretch": float(m.get("window_clean_stretch_rate", 0.0)),
                    "clearances": float(m.get("window_clearances", 0.0)),
                    "bracket_score": float(m.get("window_bracket_score", 0.0)),
                    "horizon_capped": _horizon_capped(_ENV),
                }
            )
            r = rows[-1]
            # Targets: solved = every flight lands within +/-60 s, no loss of separation;
            # hard-solved = the same within +/-30 s.
            r["solved"] = bool(r["all_on_time"] and not r["sep"])
            r["hard_solved"] = bool(not r["sep"] and r["n_flights"] > 0 and r["landed"] >= r["n_flights"]
                                    and r["max_dev"] <= HARD_SOLVED_DEV_S)
            if attempt:
                continue  # per-flight rows describe the deterministic policy only
            # Per flight, by queue position (AMAN order): does performance drift along the stream?
            queue = sorted(_ENV._eta, key=_ENV._eta.get)
            for pos, cs in enumerate(queue):
                dev = _ENV._landing_devs.get(cs)
                flights.append({
                    "model_idx": mi, "seed": int(seed), "queue_pos": pos, "callsign": cs,
                    "landed": dev is not None,
                    "on_time": dev is not None and abs(dev) <= 60.0,
                    "abs_dev": abs(dev) if dev is not None else "",
                })
    return rows, flights


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    from config.config import Config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="checkpoints; the FIRST is the reference")
    ap.add_argument("--seeds", type=int, default=100, help="first N of Config.SCENARIO_EVAL_SEEDS")
    ap.add_argument("--seed-file", default=None,
                    help="score these stream seeds instead (one per line, e.g. from analysis/long_stream_seeds.py); "
                         "--seeds then caps how many are used")
    ap.add_argument("--flights", type=int, default=20, help="flights per scenario (stream length)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--csv", type=str, default=None, help="per-episode CSV output")
    ap.add_argument("--flights-csv", type=str, default=None, help="per-flight CSV (queue position)")
    ap.add_argument("--segments", type=int, default=1,
                    help="stitch this many independently generated --flights scenarios per stream")
    ap.add_argument("--gap", type=float, nargs=2, default=(120.0, 900.0), metavar=("MIN", "MAX"),
                    help="cooling gap between stitched segments, seconds (uniform)")
    ap.add_argument("--attempts", type=int, default=0,
                    help="extra STOCHASTIC attempts per seed after the deterministic one; reports "
                         "pass@1/pass@K and the best attempt under the lexicographic objective")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="K > 1: the deterministic attempt uses critic-guided one-step lookahead over "
                         "the policy's K most likely actions (analysis/lookahead.py)")
    ap.add_argument("--shield", action="store_true",
                    help="apply render_policy's conflict shield (refuse + next_best) to every model")
    ap.add_argument("--use-case", type=int, choices=(1, 2), default=None,
                    help="scenario, overriding the run's: 1 = MXP trombone, 2 = BGY point merge (zero-shot tests)")
    args = ap.parse_args()

    for p in args.models:
        if p != DO_NOTHING_MODEL and not Path(p).exists():
            sys.exit(f"missing checkpoint: {p}")
    # All models in one scoring run share the env, so they must share an observation width.
    from actions.action_set import sync_obs_width_to_checkpoint

    widths = set()
    for p in args.models:
        if p == DO_NOTHING_MODEL:
            continue
        sync_obs_width_to_checkpoint(p)
        widths.add(Config.AIRCRAFT_SCALAR_LEN)
    if len(widths) > 1:
        sys.exit(f"models have different observation widths {sorted(widths)}; score them separately")
    if args.seed_file:
        seeds = [int(x) for x in Path(args.seed_file).read_text().split() if x.strip()][: args.seeds]
    else:
        seeds = [int(s) for s in Config.SCENARIO_EVAL_SEEDS][: args.seeds]
    stream = (f"{args.segments}x{args.flights}-flight stitched (gap {args.gap[0]:.0f}-{args.gap[1]:.0f} s)"
              if args.segments > 1 else f"{args.flights}-flight")
    print(f"Scoring {len(args.models)} model(s) on {len(seeds)} fixed seeds, {stream} "
          f"streams, deterministic, frame pinned per seed, {args.workers} workers"
          f"{', conflict shield ON' if args.shield else ''}"
          f"{', scenario: use case ' + str(args.use_case) if args.use_case else ''}.\n")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.models, args.flights, args.shield, args.segments, tuple(args.gap),
                        args.attempts, args.lookahead, args.use_case)) as pool:
        per_seed = pool.map(_score_seed, seeds, chunksize=1)
    all_rows = [r for seed_rows, _ in per_seed for r in seed_rows]
    rows = [r for r in all_rows if r["attempt"] == 0]  # the deterministic policy
    n_capped = len({r["seed"] for r in all_rows if r["horizon_capped"]})
    if n_capped:
        print(f"WARNING: on {n_capped} seeds the episode cap ends the stream before its last scheduled landing; "
              "flights still airborne there were cut off by the evaluation, not lost by the agent\n")
    flight_rows = [f for _, seed_flights in per_seed for f in seed_flights]

    by_model = {i: sorted((r for r in rows if r["model_idx"] == i), key=lambda r: r["seed"])
                for i in range(len(args.models))}
    ref = by_model[0]
    for i, path in enumerate(args.models):
        rs = by_model[i]
        n = len(rs)
        on = np.array([r["on_time"] for r in rs])
        k_sep = sum(r["sep"] for r in rs)
        k_all = sum(r["all_on_time"] for r in rs)
        print(f"=== [{i}] {path}")
        print(f"  on_time       {on.mean():.3f}  (SE {on.std(ddof=1) / math.sqrt(n):.3f})")
        lo, hi = wilson(k_sep, n)
        print(f"  separation    {k_sep / n:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
        lo, hi = wilson(k_all, n)
        print(f"  all on time   {k_all / n:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
        k_solved, k_hard = sum(r["solved"] for r in rs), sum(r["hard_solved"] for r in rs)
        print(f"  solved        {k_solved} / {n} (+/-60 s, no LoS)   hard-solved {k_hard} / {n} (+/-{HARD_SOLVED_DEV_S:.0f} s, no LoS)")
        if args.shield:
            print(f"  shield        {np.mean([r['refusals'] for r in rs]):.1f} refusals / episode")
        # On-time by queue position, in five bands: flat = the agent is stable along the stream.
        fr = [f for f in flight_rows if f["model_idx"] == i]
        n_pos = max(f["queue_pos"] for f in fr) + 1
        edges = np.linspace(0, n_pos, 6).round().astype(int)
        bands = []
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            sel = [f["on_time"] for f in fr if lo_ <= f["queue_pos"] < hi_]
            bands.append(f"{lo_ + 1}-{hi_}: {np.mean(sel):.2f}")
        print(f"  on time by queue position   " + "   ".join(bands))
        print(f"  sequence      landed in AMAN position {np.mean([r['aman_order'] for r in rs]):.3f},"
              f" swaps {np.mean([r['swaps'] for r in rs]):.2f} / stream,"
              f" clean 10-flight stretches {np.mean([r['clean_stretch'] for r in rs]):.3f}")
        print(f"  clearances    {np.mean([r['clearances'] for r in rs]):.1f} / stream"
              f" ({np.mean([r['clearances'] / max(1, r['steps']) for r in rs]):.0%} of steps)")
        print(f"  landed        {np.mean([r['landed'] for r in rs]):.1f} / {np.mean([r['n_flights'] for r in rs]):.0f}"
              f"   landing |dev| mean {np.mean([r['mean_dev'] for r in rs]):.0f} s,"
              f" worst-flight mean {np.mean([r['max_dev'] for r in rs]):.0f} s")
        if i > 0:
            d = on - np.array([r["on_time"] for r in ref])
            se = d.std(ddof=1) / math.sqrt(n)
            verdict = "significant at 95%" if abs(d.mean()) > 1.96 * se else "inside the noise band"
            print(f"  vs [0], paired: on_time {d.mean():+.3f} (SE {se:.3f}) -> {verdict};"
                  f" better on {int((d > 0).sum())} seeds, worse on {int((d < 0).sum())}")
            # Discordant pairs (McNemar): only seeds where exactly one of the two busts carry
            # information about a difference in bust rate.
            fixed = sum(1 for a, b in zip(ref, rs) if a["sep"] and not b["sep"])
            broke = sum(1 for a, b in zip(ref, rs) if b["sep"] and not a["sep"])
            m = fixed + broke
            z = (fixed - broke) / math.sqrt(m) if m else 0.0
            print(f"  vs [0], separation: fixed on {fixed} seeds, newly lost on {broke}"
                  f" (McNemar z {z:+.2f}{', significant' if abs(z) > 1.96 else ''})")
        print()

    if args.attempts > 0:
        # Best attempt per seed under the lexicographic objective: no loss of separation, then
        # the sum of bracket values, then the fewest clearances.
        key = lambda r: (not r["sep"], r["bracket_score"], -r["clearances"])
        print(f"=== attempts: 1 deterministic + {args.attempts} stochastic per seed")
        for i, path in enumerate(args.models):
            by_seed = {}
            for r in all_rows:
                if r["model_idx"] == i:
                    by_seed.setdefault(r["seed"], []).append(r)
            det = [rs[0] for rs in by_seed.values()]
            sto = [r for rs in by_seed.values() for r in rs if r["attempt"] > 0]
            best = [max(rs, key=key) for rs in by_seed.values()]
            n = len(by_seed)
            solved_any = sum(any(r["all_on_time"] for r in rs) for rs in by_seed.values())
            never = [rs for rs in by_seed.values() if not any(r["all_on_time"] for r in rs)]
            safety_bound = sum(all(r["sep"] for r in rs) for rs in never)
            rescued = sum(1 for rs in by_seed.values() if not rs[0]["all_on_time"] and any(r["all_on_time"] for r in rs))
            print(f"[{i}] {path}")
            print(f"  solved (all on time, no LoS): deterministic {np.mean([r['all_on_time'] for r in det]):.2f}"
                  f" | stochastic pass@1 {np.mean([r['all_on_time'] for r in sto]):.2f}"
                  f" | pass@{1 + args.attempts} {solved_any / n:.2f} ({rescued} seeds rescued by an attempt)")
            print(f"  separation lost: deterministic {np.mean([r['sep'] for r in det]):.2f}"
                  f" | stochastic {np.mean([r['sep'] for r in sto]):.2f} | best attempt {np.mean([r['sep'] for r in best]):.2f}")
            print(f"  on time: deterministic {np.mean([r['on_time'] for r in det]):.3f} | stochastic {np.mean([r['on_time'] for r in sto]):.3f}"
                  f" | best attempt {np.mean([r['on_time'] for r in best]):.3f};"
                  f" clearances/stream det {np.mean([r['clearances'] for r in det]):.0f}, best {np.mean([r['clearances'] for r in best]):.0f}")
            print(f"  never solved: {len(never)} seeds -- safety-bound (every attempt lost separation) {safety_bound},"
                  f" precision-bound {len(never) - safety_bound}")
            hard_any = sum(any(r["hard_solved"] for r in rs) for rs in by_seed.values())
            print(f"  hard-solved (+/-{HARD_SOLVED_DEV_S:.0f} s, no LoS): deterministic {np.mean([r['hard_solved'] for r in det]):.2f}"
                  f" | stochastic pass@1 {np.mean([r['hard_solved'] for r in sto]):.2f} | pass@{1 + args.attempts} {hard_any / n:.2f}")
        print()

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", *all_rows[0].keys()])
            w.writeheader()
            for r in all_rows:
                w.writerow({"model": args.models[r["model_idx"]], **r})
        print(f"per-episode rows -> {args.csv}")
    if args.flights_csv:
        import csv

        with open(args.flights_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", *flight_rows[0].keys()])
            w.writeheader()
            for f in flight_rows:
                w.writerow({"model": args.models[f["model_idx"]], **f})
        print(f"per-flight rows -> {args.flights_csv}")


if __name__ == "__main__":
    main()
