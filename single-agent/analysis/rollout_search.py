"""Receding-horizon rollout search for the windowed policy. At each planning point compare
the incumbent, greedy and sampled candidate continuations on simulator snapshots. Use
explicit model, seed, budget, K, M and output options; preserve separate safety, timing
and clearance metrics."""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import score_windowed as sw  # noqa: E402  (worker init / model loading / env config sync)

UNLANDED_DEV_S = 1000.0  # deviation charged to a flight that never lands, in the |dev| sum
ON_TIME_S = 60.0


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------

def outcome_key(m: dict) -> tuple:
    """Lexicographic outcome of a finished episode from its episode_metrics (larger is better)."""
    n_fl = float(m.get("window_n_flights", 0.0))
    n_ld = float(m.get("window_n_landed", 0.0))
    n_on = round(float(m.get("window_on_time_rate", 0.0)) * n_fl)
    dev_sum = float(m.get("window_mean_abs_landing_dev", 0.0)) * n_ld + UNLANDED_DEV_S * max(0.0, n_fl - n_ld)
    return (not bool(m.get("had_separation", False)), int(n_on), -round(dev_sum, 3),
            -int(round(float(m.get("window_clearances", 0.0)))))


def is_solved(m: dict) -> bool:
    return bool(m.get("success", False)) and not bool(m.get("had_separation", False))


# --------------------------------------------------------------------------------------------
# rollouts
# --------------------------------------------------------------------------------------------

def _valid(obs, action) -> bool:
    """Is a stored action allowed by the observation's masks?"""
    a = np.asarray(action).ravel()
    try:
        ai, ci = int(a[0]), int(a[1])
        if "mask_select" in obs and not obs["mask_select"][ai]:
            return False
        return bool(obs["mask_action_per_ac"][ai, ci])
    except Exception:  # noqa: BLE001
        return False


def _policy_action(model, obs, deterministic: bool) -> np.ndarray:
    a, _ = model.predict(obs, deterministic=deterministic)
    return np.asarray(a).ravel().astype(np.int64).copy()


def rollout(env, model, obs, mode: str, *, torch_seed: int = 0, replay: list | None = None,
            max_steps: int = 10_000, bound_n_on: int | None = None,
            sample_steps: int | None = None) -> tuple[list, dict | None, int]:
    """Roll the env (already restored to the snapshot, `obs` its observation) to the end of the
    episode. mode: "det" | "sample" | "replay" (replay[i] for the first len(replay) steps, then the
    deterministic policy; an invalid stored action also falls back to the policy for that step).
    bound_n_on: exact pruning. If given, the rollout is abandoned (metrics None) as soon as so many
    flights have landed outside +/-60 s that it can no longer reach `bound_n_on` on-time flights;
    such a candidate could neither beat nor tie the best no-LoS candidate (whose n_on that is), so
    the argmax is unchanged.
    Returns (actions taken, episode_metrics or None if pruned, n_fallbacks)."""
    import torch

    if mode == "sample":
        torch.manual_seed(int(torch_seed) % (2**31))
    actions, fallbacks, info = [], 0, {}
    done = False
    while not done and len(actions) < max_steps:
        i = len(actions)
        if mode == "replay" and replay is not None and i < len(replay):
            a = np.asarray(replay[i], dtype=np.int64)
            if not _valid(obs, a):
                a, fallbacks = _policy_action(model, obs, True), fallbacks + 1
        else:
            sampling = mode == "sample" and (sample_steps is None or i < sample_steps)
            a = _policy_action(model, obs, deterministic=not sampling)
        try:
            obs, _r, te, tr, info = env.step(a)
        except Exception:  # noqa: BLE001  a mask-valid clearance the interpreter cannot execute here
            a, fallbacks = _policy_action(model, obs, True), fallbacks + 1
            obs, _r, te, tr, info = env.step(a)
        actions.append(a)
        done = te or tr
        if bound_n_on is not None and not done:
            late = sum(1 for d in env._landing_devs.values() if abs(d) > ON_TIME_S)
            if len(env._eta) - late < bound_n_on:
                return actions, None, fallbacks
    return actions, (info.get("episode_metrics") or {}), fallbacks


def _save_record(path: Path, seed: int, rec_obs: list, rec_act: list, m: dict, rec_det: list | None = None) -> None:
    """One episode's executed (observation, action) pairs and its outcome, for distillation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {f"obs_{k}": np.stack([o[k] for o in rec_obs]) for k in rec_obs[0]} if rec_obs else {}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, actions=np.stack(rec_act).astype(np.int64) if rec_act else np.zeros((0, 3), np.int64),
                        seed=np.int64(seed), solved=np.bool_(is_solved(m)), sep=np.bool_(m.get("had_separation", False)),
                        on_time=np.float32(m.get("window_on_time_rate", 0.0)), outcome=np.array(outcome_key(m), dtype=np.float64),
                        is_det=np.array(rec_det if rec_det is not None else [], dtype=bool),
                        **arrs)
    tmp.rename(path)  # atomic: a crash never leaves a half-written record behind


def search_episode(env, model, seed: int, K: int, M: int, resim_incumbent: bool = False,
                   stop_solved: bool = True, prune: bool = True, record_dir: str | None = None,
                   sample_steps: int | None = None, attempt: int = 0, observer=None) -> dict:
    """One episode of receding-horizon rollout search. Returns the result row.
    attempt: repeat index of the search on the same scenario; it only re-seeds the sampled candidates
    (attempt 0 = the original seeding). observer: optional score_windowed-style episode observer
    (begin/before_step/after_step/end, e.g. evaluation_records.EpisodeRecorder), called on the EXECUTED steps only."""
    import torch

    t0 = time.time()
    torch.manual_seed((int(seed) * 1009) % (2**31))
    np.random.seed(int(seed) % (2**32))  # pins the observation frame, as score_windowed does
    obs, _ = env.reset(seed=int(seed), options={"scenario_seed": int(seed)})
    if observer is not None:
        observer.begin(env, obs, int(seed), int(attempt))
    max_steps = (int(env.max_steps_per_episode) + 2) * max(1, int(getattr(env, "_max_picks", 1)))

    incumbent: list | None = None       # remaining action list of the chosen candidate
    inc_key = None                      # its known outcome key (cached)
    inc_metrics: dict | None = None
    steps, ret, info, done = 0, 0.0, {}, False
    n_rollouts = n_roll_steps = n_plans = n_switch = violations = replay_mismatch = fallbacks = n_pruned = 0
    plan_best_keys = []
    rec_obs: list[dict] = []
    rec_act: list[np.ndarray] = []
    rec_det: list[bool] = []

    def _rec(o, a) -> None:
        if record_dir:
            rec_obs.append({k: np.array(v, copy=True) for k, v in o.items()})
            rec_act.append(np.asarray(a, dtype=np.int64).ravel().copy())
            # is the executed action the deterministic policy's own? (False = a correction by the search)
            rec_det.append(bool(np.array_equal(_policy_action(model, o, True), rec_act[-1])))

    while not done and steps < max_steps:
        snap, snap_obs = env.snapshot(), obs
        cands = []  # (k, kind, actions, metrics, key)
        # k = 0: the incumbent (cached outcome, or replayed when resimulating / at the first point)
        if incumbent is None:
            acts, met, fb = rollout(env, model, snap_obs, "det", max_steps=max_steps)
            env.restore(snap)
            n_rollouts += 1; n_roll_steps += len(acts); fallbacks += fb
            cands.append((0, "det0", acts, met, outcome_key(met)))
        else:
            if resim_incumbent:
                acts, met, fb = rollout(env, model, snap_obs, "replay", replay=incumbent, max_steps=max_steps)
                env.restore(snap)
                n_rollouts += 1; n_roll_steps += len(acts); fallbacks += fb
                if outcome_key(met) != inc_key or [a.tolist() for a in acts] != [a.tolist() for a in incumbent]:
                    replay_mismatch += 1
                cands.append((0, "inc", acts, met, outcome_key(met)))
            else:
                cands.append((0, "inc", incumbent, inc_metrics, inc_key))
        # k >= 1: the deterministic policy (not at the first point: it IS candidate 0 there), then samples
        best = cands[0]
        for k in range(1, K):
            if stop_solved and is_solved(best[3]):
                break  # a solved episode cannot be improved on the main criterion: stop searching
            if k == 1 and incumbent is not None:
                kind, tseed = "det", 0
            else:
                kind, tseed = "sample", (int(seed) * 1009 + n_plans * 7919 + k * 104729 + int(attempt) * 15485863)
            bound = int(best[4][1]) if (prune and best[4][0]) else None  # best no-LoS candidate's on-time count
            acts, met, fb = rollout(env, model, snap_obs, "det" if kind == "det" else "sample",
                                    torch_seed=tseed, max_steps=max_steps, bound_n_on=bound,
                                    sample_steps=sample_steps)
            env.restore(snap)
            n_rollouts += 1; n_roll_steps += len(acts); fallbacks += fb
            if met is None:
                n_pruned += 1
                continue
            c = (k, kind, acts, met, outcome_key(met))
            cands.append(c)
            if c[4] > best[4]:  # strict: ties go to the lowest k, i.e. the incumbent
                best = c
        if inc_key is not None and best[4] < inc_key:
            violations += 1
            print(f"  [seed {seed}] MONOTONICITY VIOLATION at plan {n_plans}: best {best[4]} < incumbent {inc_key}",
                  flush=True)
        n_switch += int(best[0] != 0 and incumbent is not None)
        plan_best_keys.append(best[4])
        incumbent, inc_key, inc_metrics = best[2], best[4], best[3]
        n_plans += 1

        # Execute the first M actions of the chosen candidate for real.
        env.restore(snap)
        obs = snap_obs
        take = min(M, len(incumbent))
        for i in range(take):
            a = incumbent[i]
            if not _valid(obs, a):
                a, fallbacks = _policy_action(model, obs, True), fallbacks + 1
                replay_mismatch += 1
                incumbent, inc_key, inc_metrics = None, None, None  # the cached outcome no longer holds
                _rec(obs, a)
                if observer is not None:
                    observer.before_step(env, obs, a)
                obs, r, te, tr, info = env.step(a)
                if observer is not None:
                    observer.after_step(env, info)
                ret += float(r); steps += 1
                done = te or tr
                break
            _rec(obs, a)
            if observer is not None:
                observer.before_step(env, obs, a)
            obs, r, te, tr, info = env.step(np.asarray(a, dtype=np.int64))
            if observer is not None:
                observer.after_step(env, info)
            ret += float(r); steps += 1
            done = te or tr
            if done:
                break
        if incumbent is not None and not done:
            incumbent = incumbent[take:]
        if done and incumbent is not None and outcome_key(info.get("episode_metrics") or {}) != inc_key:
            replay_mismatch += 1

    m = info.get("episode_metrics") or {}
    if observer is not None:
        observer.end(env, info)
    if record_dir:
        _save_record(Path(record_dir) / f"{int(seed)}.npz", seed, rec_obs, rec_act, m, rec_det)
    final_key = outcome_key(m)
    if plan_best_keys and final_key < max(plan_best_keys) and replay_mismatch == 0:
        violations += 1
    nf, nl = float(m.get("window_n_flights", 0.0)), float(m.get("window_n_landed", 0.0))
    return {
        "seed": int(seed),
        "solved": is_solved(m),
        "hard_solved": bool(not m.get("had_separation", False) and nf > 0 and nl >= nf
                            and float(m.get("window_max_abs_landing_dev", 0.0)) <= sw.HARD_SOLVED_DEV_S),
        "all_on_time": bool(m.get("success", False)),
        "sep": bool(m.get("had_separation", False)),
        "on_time": float(m.get("window_on_time_rate", 0.0)),
        "landed": nl,
        "n_flights": nf,
        "mean_dev": float(m.get("window_mean_abs_landing_dev", 0.0)),
        "max_dev": float(m.get("window_max_abs_landing_dev", 0.0)),
        "end_reason": str(m.get("end_reason")),
        "steps": steps,
        "ep_return": ret,
        "clearances": float(m.get("window_clearances", 0.0)),
        "bracket_score": float(m.get("window_bracket_score", 0.0)),
        "n_plans": n_plans,
        "n_rollouts": n_rollouts,
        "n_pruned": n_pruned,
        "rollout_steps": n_roll_steps,
        "n_switches": n_switch,
        "fallbacks": fallbacks,
        "violations": violations,
        "replay_mismatch": replay_mismatch,
        "first_plan_solved": bool(plan_best_keys and plan_best_keys[0][0] and plan_best_keys[0][1] == nf),
        "wall_s": round(time.time() - t0, 1),
    }


def _work(args) -> dict:
    seed, K, M, resim, stop_solved, prune, record_dir, sample_steps = args
    row = search_episode(sw._ENV, sw._MODELS[0], int(seed), int(K), int(M), bool(resim), bool(stop_solved), bool(prune),
                         record_dir, sample_steps)
    row.update({"K": K, "M": M})
    return row


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------

def summarise(rows: list[dict], label: str) -> str:
    lines = [f"=== {label}: {len(rows)} seeds"]
    if not rows:
        return "\n".join(lines)
    lines.append(f"solved {sum(r['solved'] for r in rows)}/{len(rows)}, "
                 f"loss of separation {sum(r['sep'] for r in rows)}, "
                 f"on-time fraction {np.mean([r['on_time'] for r in rows]):.3f}")
    lines.append(f"cost: {np.mean([r['n_rollouts'] for r in rows]):.0f} rollouts/episode "
                 f"({np.mean([r['n_pruned'] for r in rows]):.0f} pruned), "
                 f"{np.mean([r['rollout_steps'] for r in rows]):.0f} simulated steps/episode "
                 f"(vs {np.mean([r['steps'] for r in rows]):.0f} executed), wall {np.mean([r['wall_s'] for r in rows]):.0f} s/episode "
                 f"(max {max(r['wall_s'] for r in rows):.0f})")
    lines.append(f"incumbent switches: {np.mean([r['n_switches'] for r in rows]):.2f}/episode over "
                 f"{np.mean([r['n_plans'] for r in rows]):.1f} plans; seeds solved at the very first plan: "
                 f"{sum(r['first_plan_solved'] for r in rows)}")
    lines.append(f"checks: monotonicity violations {sum(r['violations'] for r in rows)}, replay mismatches "
                 f"{sum(r['replay_mismatch'] for r in rows)}, policy fallbacks {sum(r['fallbacks'] for r in rows)}")
    return "\n".join(lines)


def _load_rows(*csvs: str) -> list[dict]:
    """Rows of one or more per-episode CSVs (types restored); the first row of a seed wins."""
    import csv

    def conv(v: str):
        if v in ("True", "False"):
            return v == "True"
        try:
            f = float(v)
            return int(f) if re.fullmatch(r"-?\d+", v) else f
        except ValueError:
            return v

    out: dict[int, dict] = {}
    for c in csvs:
        if not Path(c).exists():
            continue
        with open(c) as fh:
            for r in csv.DictReader(fh):
                row = {k: conv(v) for k, v in r.items() if k != "model"}
                out.setdefault(int(row["seed"]), row)
    return list(out.values())


def _append_row(path: str, model: str, row: dict) -> None:
    import csv

    new = not Path(path).exists() or Path(path).stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", *row.keys()])
        if new:
            w.writeheader()
        w.writerow({"model": model, **row})


def main() -> None:
    from config.config import Config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs=1, default=None, help="the checkpoint (one)")
    ap.add_argument("--summarize-only", nargs="+", default=None, metavar="CSV",
                    help="no search: merge these per-episode CSVs (first row of a seed wins), write --csv if given, "
                         "and print their evaluation summary")
    ap.add_argument("--reverse-seeds", action="store_true",
                    help="process the seeds last-to-first (a second process can help a running one; merge with --summarize-only)")
    ap.add_argument("--resume", action="store_true", help="skip seeds already in --csv (rows are appended as they finish)")
    ap.add_argument("--seeds", type=int, default=100, help="first N of Config.SCENARIO_EVAL_SEEDS (or of --seed-file)")
    ap.add_argument("--seed-file", default=None)
    ap.add_argument("--flights", type=int, default=20)
    ap.add_argument("--use-case", type=int, choices=(1, 2), default=None)
    ap.add_argument("--K", type=int, default=8, help="candidates per planning point (incl. the incumbent)")
    ap.add_argument("--M", type=int, default=10, help="policy steps (env.step calls) between replans")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--resim-incumbent", action="store_true",
                    help="re-simulate the incumbent by replay at every plan and cross-check the cached outcome")
    ap.add_argument("--no-stop-solved", action="store_true",
                    help="keep generating candidates even when the incumbent already solves the episode "
                         "(default: stop; a solved episode cannot be improved on the main criterion)")
    ap.add_argument("--no-prune", action="store_true",
                    help="disable the exact on-time-count pruning of rollouts that cannot beat the best candidate")
    ap.add_argument("--sample-steps", type=int, default=None, metavar="J",
                    help="sampled candidates sample only their first J actions, then follow the deterministic policy")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="save every executed (observation, action) pair to DIR/<seed>.npz, for distillation")
    args = ap.parse_args()

    if args.summarize_only:
        rows = sorted(_load_rows(*args.summarize_only), key=lambda r: r["seed"])
        if args.csv:
            Path(args.csv).unlink(missing_ok=True)
            for r in rows:
                _append_row(args.csv, "merged", r)
        print(summarise(rows,
                        f"MPC K={rows[0]['K']} M={rows[0]['M']} (merged {len(args.summarize_only)} files)"))
        return
    if not args.models:
        sys.exit("--models is required (or --summarize-only)")
    path = args.models[0]
    if not Path(path).exists():
        sys.exit(f"missing checkpoint: {path}")
    from actions.action_set import sync_obs_width_to_checkpoint

    sync_obs_width_to_checkpoint(path)  # obs width, sequence obs, max picks, mask frame (also set in os.environ)
    if args.seed_file:
        seeds = [int(x) for x in Path(args.seed_file).read_text().split() if x.strip()][: args.seeds]
    else:
        seeds = [int(s) for s in Config.SCENARIO_EVAL_SEEDS][: args.seeds]
    print(f"Rollout search: {path}\n  {len(seeds)} seeds, K={args.K}, M={args.M}, {args.workers} workers, "
          f"max_picks={Config.MAX_PICKS}, use_case={args.use_case}", flush=True)

    if args.reverse_seeds:
        seeds = seeds[::-1]
    prior = _load_rows(args.csv) if (args.resume and args.csv) else []
    todo = [x for x in seeds if x not in {r["seed"] for r in prior}]
    rows = list(prior)
    t0 = time.time()
    with Pool(args.workers, initializer=sw._init_worker,
              initargs=(args.models, args.flights, False, 1, (120.0, 900.0), 0, 0, args.use_case)) as pool:
        for i, row in enumerate(pool.imap_unordered(_work, [(x, args.K, args.M, args.resim_incumbent, not args.no_stop_solved, not args.no_prune, args.record,
                                                      args.sample_steps)
                                                     for x in todo],
                                                    chunksize=1), 1):
            rows.append(row)
            if args.csv:
                _append_row(args.csv, path, row)  # incremental: a crash or a helper process loses nothing
            print(f"[{i}/{len(todo)}] seed {row['seed']} solved={row['solved']} sep={row['sep']} "
                  f"on_time={row['on_time']:.2f} clr={row['clearances']:.0f} rollouts={row['n_rollouts']} "
                  f"wall={row['wall_s']}s elapsed={time.time() - t0:.0f}s", flush=True)
    order = {x: i for i, x in enumerate(seeds)}
    rows.sort(key=lambda r: order.get(r["seed"], 0))
    if args.csv:
        print(f"per-episode rows -> {args.csv}")
    label = f"MPC K={args.K} M={args.M} {Path(path).parent.name}"
    print(summarise(rows, label))
    print(f"total wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
