"""Record episode, aircraft and action metrics from windowed policy evaluation.

The observer records executed decisions without changing policy actions. Reports retain
procedure, seed, attempt, safety, timing coverage and per-action counts. Initial timing
uses a do-nothing forecast; final timing distinguishes landed, predicted and unresolved
flights. Optional pick records include zero-time reselection decisions.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import score_windowed as sw  # noqa: E402  (scenario/frame pinning, loading, attempts)

# Stable CSV field names for episode and aircraft records.
EPISODE_HEADER = [
    "Agent Type (Multi|Single)",
    "Procedure type (PMS|TMB)",
    "Seed number (100 seeds)",
    "Type of policy (Deterministicx1|Stochasticx10) 0-10",
    "End reason (success|delayed|timeout|airspace_exit|Loss of separation) (LoS within 3NM and 500ft)",
    "Total number of clearances",
    "Total intial predicted deviation (seconds)",
    "Total final deviation (seconds)",
    "Nb. actions NOOP",
    "Nb. actions SPDN",
    "Nb. actions heading (turns)",
    "Nb. actions SPDUP",
    "Nb. actions lengthen trombone ",
    "Nb. actions shorten trombone",
    "Nb. actions skip waypoint",
    "Nb. actions direct to ILS (trombone only)",
    "Nb. actions direct to merge point (pms only)",
]
AIRCRAFT_HEADER = [
    "Agent Type (Multi|Single)",
    "Procedure type (PMS|TMB)",
    "Seed number (100 seeds)",
    "Type of policy (Deterministicx1|Stochasticx10) 0-10",
    "Aircraft Index (0-19)",
    "Initial time deviation (seconds)",
    "Final time deviation (seconds)",
    "Number of clearances",
]

# action-name -> template column key
CATEGORIES = ["noop", "spdn", "heading", "spdup", "lengthen", "shorten", "skip", "ils", "merge"]


def category_of(name: str) -> str:
    """Template action column for a clearance enum NAME; raises on a name no column covers."""
    if name == "DO_NOTHING":
        return "noop"
    if name.startswith("SLOW_DOWN"):
        return "spdn"
    if name.startswith("TURN_"):
        return "heading"
    if name.startswith("SPEED_UP"):
        return "spdup"
    if name == "LENGTHEN_TROMBONE":
        return "lengthen"
    if name == "SHORTEN_TROMBONE":
        return "shorten"
    if name.startswith("SKIP_"):
        return "skip"
    if name == "VECTOR_TO_ILS":
        return "ils"
    if name == "DIRECT_TO_MERGE_POINT":
        return "merge"
    raise SystemExit(f"clearance {name!r} of the active action set has no column in the paper template; "
                     "add it to analysis/evaluation_records.py category_of")


def do_nothing_deviations(env, steps_after_last_eta: int = 40) -> dict[str, float]:
    """Signed landing deviation of every flight if nothing more is ever commanded: the whole
    stream flown do-nothing to its end (analysis/long_stream_seeds.py _screen). Returns a NEW world
    from the simulator, so the env's live state is untouched. A flight that does not land in the
    rollout is absent from the result."""
    from actions.rewards import get_aircraft_time_deviations
    from config.config import Config

    world = env.simulator.world
    last_eta = max(float(t.eta_s) for t in world.arrival_targets)
    remaining = max(0.0, last_eta - float(world.age_in_seconds))
    steps = math.ceil(remaining / float(Config.TIME_BETWEEN_ACTIONS)) + steps_after_last_eta
    pred = env.simulator.advance_multiple_steps(steps=steps, world=world)
    devs = get_aircraft_time_deviations(pred)
    done = {ac.call_sign for ac in pred.aircraft if ac.mission.completed}
    return {cs: float(d) for cs, d in devs.items() if cs in done}


def aman_ttl_ttg(env) -> dict[str, float]:
    """Every flight's AMAN time to lose (+) or gain (-), as the generator set it: the sheet's "TTL or TTG value"
    column, the multi-agent team's definition. Not the do-nothing deviation (``do_nothing_deviations``)."""
    return {ac.call_sign: float(ac.aman.time_to_lose or 0.0) - float(ac.aman.time_to_gain or 0.0)
            for ac in env.simulator.world.aircraft}


def warning_pairs(world) -> int:
    """Pairs that came inside the 5 NM / 1000 ft safezone without a loss of separation (0 < severity < 1; severity
    1.0 is <= 3 NM and <= 500 ft). This is the multi-agent sheet's "Infringements (not conflicts)" (their
    evaluation/forms/landing.py warning_separation_pairs, same radii). A loss of separation is the end reason."""
    from flight_simulator import Infringement

    from utils.infringement_utils import pair_severity

    pairs = set()
    for infr in world.infringements:
        if isinstance(infr, Infringement.LOSS_OF_SEPARATION):
            sev = pair_severity(float(infr.horizontal_separation_nm), float(infr.vertical_separation_ft))
            if 0.0 < sev < 1.0:
                pairs.add(tuple(sorted(str(x) for x in infr.aircraft_pair)))
    return len(pairs)


class EpisodeRecorder:
    """score_windowed._OBSERVER: records the policy's picks and the per-flight outcome of every
    episode. Read-only (the do-nothing rollouts run on new worlds)."""

    def __init__(self, use_case: int):
        from actions.action_set import Action

        self.use_case = int(use_case)
        self.do_nothing = int(Action.DO_NOTHING)
        self.names = {int(a): a.name for a in Action}
        self.cat = {int(a): category_of(a.name) for a in Action}
        self.records: list[dict] = []
        self._initial: dict[int, dict[str, float]] = {}  # per scenario seed, shared by its attempts
        self._aman: dict[int, dict[str, float]] = {}  # AMAN TTL/TTG per scenario seed
        self._cur: dict | None = None
        self._ctx: dict | None = None

    # -- hooks ---------------------------------------------------------------------------------
    def begin(self, env, obs, seed: int, attempt: int) -> None:
        assert str(env.difficulty).endswith(f"USE_CASE_{self.use_case}"), (
            f"env difficulty {env.difficulty} is not use case {self.use_case}")
        if seed not in self._initial:
            self._initial.clear()
            self._aman.clear()
            self._initial[seed] = do_nothing_deviations(env)
            self._aman[seed] = aman_ttl_ttg(env)
        self._cur = {
            "seed": seed, "attempt": attempt,
            "picks": {n: 0 for n in self.names.values()},  # every pick by clearance name
            "per_ac": {},                                   # callsign -> non-DO_NOTHING picks
            "noop": 0, "noop_forced": 0, "noop_closing": 0,
            "n_picks": 0,
        }

    def before_step(self, env, obs, action) -> None:
        flat = np.asarray(action).ravel()
        a_air = int(flat[0])
        mask = np.asarray(obs["mask_action_per_ac"][a_air]).astype(bool)
        self._ctx = {
            "callsign": env._prev_aircraft_order[a_air],
            "alternatives": bool(mask.sum() - int(mask[self.do_nothing]) > 0),
            "pending": len(getattr(env, "_picked", [])) > 0,
            "pick_in_interval": len(getattr(env, "_picked", [])) + 1,  # 2 = a reselection (zero-time) pick
            "again": int(flat[2]) if flat.size > 2 else 0,
            "t_s": float(env.simulator.world.age_in_seconds),
        }

    def after_step(self, env, info) -> None:
        cur, ctx = self._cur, self._ctx
        assert info["callsign"] == ctx["callsign"]
        clr = int(info["clearance"])  # the clearance the env actually applied
        cur["n_picks"] += 1
        cur["picks"][self.names[clr]] += 1
        if clr == self.do_nothing:
            if ctx["pending"]:
                kind = "noop_closing"
            elif not ctx["alternatives"]:
                kind = "noop_forced"
            else:
                kind = "noop"
            cur[kind] += 1
        else:
            kind = "clearance"
            cur["per_ac"][info["callsign"]] = cur["per_ac"].get(info["callsign"], 0) + 1
        # one row per pick (written only by callers that ask for it, e.g. run_search.py --picks)
        cur.setdefault("pick_log", []).append({
            "pick": cur["n_picks"] - 1, "t_s": round(ctx["t_s"], 1), "pick_in_interval": ctx["pick_in_interval"],
            "again_requested": ctx["again"], "callsign": info["callsign"], "clearance": self.names[clr], "kind": kind})

    def end(self, env, info) -> None:
        cur = self._cur
        initial = self._initial[cur["seed"]]
        aman = self._aman[cur["seed"]]
        queue = sorted(env._eta, key=env._eta.get)  # AMAN order
        landed = dict(env._landing_devs)
        pred = do_nothing_deviations(env) if len(landed) < len(queue) else {}
        flights = []
        for pos, cs in enumerate(queue):
            if cs in landed:
                final, src = float(landed[cs]), "landed"
            elif cs in pred:
                final, src = float(pred[cs]), "predicted"
            else:
                final, src = 0.0, "unresolved"
            flights.append({"index": pos, "callsign": cs, "eta_s": float(env._eta[cs]), "initial": float(initial.get(cs, 0.0)),
                            "initial_resolved": cs in initial, "aman_ttl": aman[cs], "final": final, "final_source": src,
                            "clearances": int((env._action_counts or {}).get(cs, 0))})
        m = info.get("episode_metrics") or {}
        # the policy picks and the env's own tally of issued clearances must agree
        n_clr = sum(f["clearances"] for f in flights)
        assert n_clr == sum(cur["per_ac"].values()) == int(round(float(m.get("window_clearances", n_clr)))), (
            cur["seed"], cur["attempt"], n_clr, cur["per_ac"], m.get("window_clearances"))
        cur.update(flights=flights, end_reason=str(m.get("end_reason", info.get("end_reason"))),
                   n_clearances=n_clr, metrics=m, warning_pairs=warning_pairs(env.simulator.world))
        self.records.append(cur)
        self._cur = self._ctx = None


_OBS: EpisodeRecorder | None = None


def _init_worker(model_paths, n_flights, use_case_override, use_case, baseline=None, veto_k=None) -> None:
    global _OBS
    sw._init_worker(model_paths, n_flights, False, 1, (120.0, 900.0), 0, 0, use_case_override)
    if veto_k is not None:
        # the multi-agent team's k-step window shield ("rollout k"), on the policy's own picks
        from analysis.forecast_veto import ForecastVeto

        sw._VETO = ForecastVeto(sw._MODELS[0], int(veto_k))
    if baseline:
        # A non-learning baseline (analysis/baselines.py) replaces the policy; the checkpoint only
        # fixes the MDP (run_meta.json) and the observation width, so the baseline flies the agents' env.
        from analysis.baselines import make

        sw._MODELS[0] = make(baseline)
    _OBS = EpisodeRecorder(use_case)
    sw._OBSERVER = _OBS


def _run_seed(args: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    seed, attempts = args
    sw._ATTEMPTS = int(attempts)
    _OBS.records = []
    rows, _ = sw._score_seed(seed)
    recs = _OBS.records
    assert len(rows) == len(recs)
    return rows, recs


def resolve_use_case(model_path: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    from analysis.lookahead import env_config_for

    return int(env_config_for(model_path).get("use_case", 1))


def main() -> None:
    from config.config import Config

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint .zip (one model)")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_per_episode.csv, _per_aircraft.csv, _extra.csv, _extra_aircraft.csv")
    ap.add_argument("--seeds", type=int, default=100, help="first N of Config.SCENARIO_EVAL_SEEDS")
    ap.add_argument("--seed-file", default=None, help="score these scenario seeds instead (one per line); --seeds caps how many")
    ap.add_argument("--attempts", type=int, default=10, help="stochastic (sampled) attempts per seed, after the deterministic one (type of policy 0)")
    ap.add_argument("--flights", type=int, default=20, help="flights per scenario")
    ap.add_argument("--use-case", type=int, choices=(1, 2), default=None,
                    help="scenario, overriding the run's: 1 = MXP trombone (TMB), 2 = BGY point merge (PMS)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--veto-k", type=int, default=None,
                    help="fly the policy inside the k-step forecast veto (analysis/forecast_veto.py, the multi-agent "
                         "team's 'rollout k' window shield)")
    ap.add_argument("--picks", action="store_true",
                    help="also write <prefix>_picks.csv: one row per executed pick of every attempt")
    ap.add_argument("--baseline", choices=("random", "greedy"), default=None,
                    help="fly a non-learning baseline (analysis/baselines.py) instead of the policy, in --model's MDP")
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"missing checkpoint: {args.model}")
    from actions.action_set import sync_obs_width_to_checkpoint

    sync_obs_width_to_checkpoint(args.model)
    use_case = resolve_use_case(args.model, args.use_case)
    procedure = "TMB" if use_case == 1 else "PMS"
    if args.seed_file:
        seeds = [int(x) for x in Path(args.seed_file).read_text().split() if x.strip()][: args.seeds]
    else:
        seeds = [int(s) for s in Config.SCENARIO_EVAL_SEEDS][: args.seeds]
    from actions.action_set import Action

    print(f"{args.model}: use case {use_case} ({procedure}), {len(seeds)} seeds x (1 deterministic + {args.attempts} "
          f"sampled), {args.flights}-flight streams, TADA_ACTION_SET={Config.ACTION_SET} "
          f"max_picks={int(getattr(Config, 'MAX_PICKS', 1))}, {args.workers} workers", flush=True)
    for a in Action:
        category_of(a.name)  # fail before the run on a clearance with no column

    with Pool(args.workers, initializer=_init_worker,
              initargs=([args.model], args.flights, args.use_case, use_case, args.baseline, args.veto_k)) as pool:
        per_seed = pool.map(_run_seed, [(s, args.attempts) for s in seeds], chunksize=1)

    ep_rows, ac_rows, extra, extra_ac, picks = [], [], [], [], []
    names = [a.name for a in Action]
    for rows, recs in per_seed:
        for r, rec in zip(rows, recs):
            assert (r["seed"], r["attempt"]) == (rec["seed"], rec["attempt"])
            seed, attempt = rec["seed"], rec["attempt"]
            reason = {"separation": "Loss of separation"}.get(rec["end_reason"], rec["end_reason"])
            fl = rec["flights"]
            counts = {c: 0 for c in CATEGORIES}
            for n, k in rec["picks"].items():
                if n != "DO_NOTHING":
                    counts[category_of(n)] += k
            counts["noop"] = rec["noop"]
            ep_rows.append([
                "Single", procedure, seed, attempt, reason, rec["n_clearances"],
                round(sum(abs(f["initial"]) for f in fl), 2), round(sum(abs(f["final"]) for f in fl), 2),
                *[counts[c] for c in CATEGORIES],
            ])
            for f in fl:
                ac_rows.append(["Single", procedure, seed, attempt, f["index"], round(f["initial"], 2),
                                round(f["final"], 2), f["clearances"]])
                extra_ac.append({"seed": seed, "attempt": attempt, "aircraft_index": f["index"], "callsign": f["callsign"],
                                 "initial_dev_s": round(f["initial"], 2), "initial_resolved": f["initial_resolved"],
                                 "aman_ttl_s": round(f["aman_ttl"], 3),
                                 "final_dev_s": round(f["final"], 2), "final_source": f["final_source"],
                                 "clearances": f["clearances"], "eta_s": round(f["eta_s"], 2)})
            srcs = [f["final_source"] for f in fl]
            extra.append({
                "seed": seed, "attempt": attempt, "env_end_reason": rec["end_reason"], "steps": r["steps"],
                "n_flights": len(fl), "n_landed": srcs.count("landed"), "n_final_predicted": srcs.count("predicted"),
                "n_final_unresolved": srcs.count("unresolved"),
                "n_initial_unresolved": sum(not f["initial_resolved"] for f in fl),
                "solved": r["solved"], "hard_solved": r["hard_solved"], "all_on_time": r["all_on_time"],
                "on_time_rate": round(r["on_time"], 4), "sep": r["sep"], "mean_abs_landing_dev_s": round(r["mean_dev"], 2),
                "max_abs_landing_dev_s": round(r["max_dev"], 2), "bracket_score": r["bracket_score"],
                "horizon_capped": r["horizon_capped"], "ep_return": round(r["ep_return"], 4),
                "n_picks": rec["n_picks"], "again_picks": int(rec["metrics"].get("window_again_picks", 0)),
                "multi_pick_steps": int(rec["metrics"].get("window_multi_pick_steps", 0)),
                "noop": rec["noop"], "noop_forced": rec["noop_forced"], "noop_closing": rec["noop_closing"],
                "warning_pairs": rec["warning_pairs"], "veto_overrides": r.get("veto_overrides", 0),
                **{f"n_{n}": rec["picks"][n] for n in names},
            })
            idx = {f["callsign"]: f["index"] for f in fl}
            picks.extend({"seed": seed, "attempt": attempt, "aircraft_index": idx.get(p_["callsign"], -1), **p_}
                         for p_ in rec.get("pick_log", []))

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(suffix: str, header, rows) -> Path:
        p = Path(f"{out}_{suffix}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        return p

    p1 = write("per_episode", EPISODE_HEADER, ep_rows)
    p2 = write("per_aircraft", AIRCRAFT_HEADER, ac_rows)
    p3 = write("extra", list(extra[0]), [list(e.values()) for e in extra])
    p4 = write("extra_aircraft", list(extra_ac[0]), [list(e.values()) for e in extra_ac])
    if args.picks:
        write("picks", list(picks[0]), [list(p_.values()) for p_ in picks])
    det = [e for e in extra if e["attempt"] == 0]
    print(f"deterministic: {sum(e['solved'] for e in det)}/{len(det)} solved, "
          f"{sum(e['sep'] for e in det)} loss of separation; all attempts: {sum(e['solved'] for e in extra)}/{len(extra)} solved")
    for p in (p1, p2, p3, p4):
        print(f"-> {p}")


if __name__ == "__main__":
    main()
