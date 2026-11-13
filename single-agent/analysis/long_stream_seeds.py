"""Seeds for long streams: K stitched 20-flight segments, screened for feasibility.

For each candidate stream seed (a fixed generator, never one of the 100 validation seeds) the full
stitched stream is generated and flown do-nothing to its end. A stream is FEASIBLE when no flight is
predicted more than --max-early seconds early (default 650 s: the scenario generator's own cap on the
delay a flight may need). Over-capacity streams, with flights thousands of seconds early, are the
impossible cases the over-capacity finding describes; feasible streams are the "easier" long streams.

Writes <out>_all.csv (seed, max_early_s, n_flights, feasible, then per-stream difficulty columns from the
same do-nothing flight: max_late_s, max_abs_dev_s, total_abs_dev_s, seq_switches, entry_inversions, los_pairs_t0)
and <out>_feasible.txt / <out>_all.txt
(the first --n seeds of each kind, one per line, for score_windowed.py --seed-file).

Deviation = landing time - AMAN target eta (negative = early; MISS_TIMED_LANDING is only raised beyond 10 s, so
smaller deviations read as 0). seq_switches = pairwise inversions between the do-nothing landing order (eta + dev)
and the AMAN (eta) order; entry_inversions = same for the spawn order (start_time_s) vs the AMAN order.
--max-abs-dev additionally bounds max_abs_dev_s in `feasible` (early AND late); `--max-early none --max-abs-dev 900`
uses that rule alone. The columns early_le_650 / abs_le_900 are always written, independent of the flags. --seed-file screens the given seeds instead of
drawing candidates (no holdout exclusion).

    python analysis/long_stream_seeds.py --segments 5 --gap 600 900 --candidates 1500 --n 40 \\
        --out analysis/<dir>/k5 --workers 40
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from config.config import Config

_ENV = None
EXTRA = ["max_late_s", "max_abs_dev_s", "total_abs_dev_s", "seq_switches", "entry_inversions", "los_pairs_t0"]


def _init(segments: int, gap: tuple, flights: int, use_case=None):
    global _ENV
    from atc_env.windowed_env import WindowedATCEnv

    _ENV = WindowedATCEnv({"evaluation_mode": True, "scenario_aircraft_count": flights, "stitch_segments": segments,
                           "stitch_gap_s": gap, "max_episode_steps_cap": 10_000,
                           **({"use_case": int(use_case)} if use_case else {})})


def _screen(seed: int):
    from actions.rewards import get_aircraft_time_deviations
    from flight_simulator import Infringement
    from utils.infringement_utils import is_realised_los, pair_severity

    env = _ENV
    np.random.seed(seed % 2**32)
    env.reset(seed=seed, options={"scenario_seed": seed})
    world = env.simulator.world
    last_eta = max(float(t.eta_s) for t in world.arrival_targets)
    steps = math.ceil(last_eta / float(Config.TIME_BETWEEN_ACTIONS)) + 40  # the whole stream, do-nothing
    pred = env.simulator.advance_multiple_steps(steps=steps, world=world)
    devs = get_aircraft_time_deviations(pred)
    early = max(0.0, -min(devs.values())) if devs else 0.0
    late = max(0.0, max(devs.values())) if devs else 0.0
    absd = [abs(d) for d in devs.values()]
    eta = {str(t.ac_callsign): float(t.eta_s) for t in world.arrival_targets}
    landing = {cs: eta[cs] + devs.get(cs, 0.0) for cs in eta}
    start = {str(a.call_sign): float(a.start_time) for a in world.aircraft}
    los = {tuple(sorted(str(x) for x in i.aircraft_pair)) for i in pred.infringements
           if isinstance(i, Infringement.LOSS_OF_SEPARATION)
           and is_realised_los(pair_severity(i.horizontal_separation_nm, i.vertical_separation_ft))}
    extra = {"max_late_s": round(late, 1), "max_abs_dev_s": round(max(absd), 1) if absd else 0.0,
             "total_abs_dev_s": round(sum(absd), 1),
             "seq_switches": _inversions(eta, landing), "entry_inversions": _inversions(eta, start),
             "los_pairs_t0": len(los)}
    return seed, round(early, 1), len(env._eta), extra


def _early_arg(v: str):
    f = None if v.strip().lower() in ("none", "off") else float(v)
    return None if f is None or f < 0 else f


def _rule(args) -> str:
    parts = [] if args.max_early is None else [f"no flight > {args.max_early:.0f} s early"]
    if args.max_abs_dev is not None:
        parts.append(f"max |dev| <= {args.max_abs_dev:.0f} s")
    return ", ".join(parts) or "no criterion"


def _inversions(aman: dict, other: dict) -> int:
    """Pairs of flights ordered differently by `other` than by the AMAN targets (Kendall tau distance)."""
    cs = list(aman)
    return sum(1 for i in range(len(cs)) for j in range(i + 1, len(cs))
               if (aman[cs[i]] - aman[cs[j]]) * (other[cs[i]] - other[cs[j]]) < 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--segments", type=int, default=5)
    ap.add_argument("--flights", type=int, default=20, help="flights per segment")
    ap.add_argument("--gap", type=float, nargs=2, default=(600.0, 900.0), help="cooling gap range between segments (s)")
    ap.add_argument("--candidates", type=int, default=1000)
    ap.add_argument("--n", type=int, default=40, help="seeds to keep of each kind")
    ap.add_argument("--max-early", type=_early_arg, default=650.0,
                    help="`feasible` requires max_early_s <= this (default 650); 'none' (or a negative value) disables the early criterion")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-abs-dev", type=float, default=None,
                    help="if set, `feasible` also requires max_abs_dev_s <= this (default: off)")
    ap.add_argument("--seed-file", default=None, help="screen the seeds in this file (one per line) instead of drawing candidates")
    ap.add_argument("--rng", type=int, default=2027, help="candidate generator seed")
    ap.add_argument("--use-case", type=int, choices=(1, 2), default=None, help="scenario: 1 = MXP (default), 2 = BGY point merge")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    if args.seed_file:
        cands = [int(x) for x in Path(args.seed_file).read_text().split()]
    else:
        holdout = {int(s) for s in Config.SCENARIO_EVAL_SEEDS}
        rng = np.random.default_rng(args.rng)
        cands = []
        seen = set()
        while len(cands) < args.candidates:
            s = int(rng.integers(0, 2**31 - 1))
            if s not in holdout and s not in seen:
                cands.append(s)
                seen.add(s)

    def _ok(early, ex):
        return (args.max_early is None or early <= args.max_early) and (args.max_abs_dev is None or ex["max_abs_dev_s"] <= args.max_abs_dev)

    with Pool(args.workers, initializer=_init, initargs=(args.segments, tuple(args.gap), args.flights, args.use_case)) as pool:
        rows = pool.map(_screen, cands, chunksize=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "max_early_s", "n_flights", "feasible"] + EXTRA + ["early_le_650", "abs_le_900"])
        for seed, early, n, ex in rows:
            w.writerow([seed, early, n, int(_ok(early, ex))] + [ex[k] for k in EXTRA]
                       + [int(early <= 650.0), int(ex["max_abs_dev_s"] <= 900.0)])
    feasible = [seed for seed, early, _, ex in rows if _ok(early, ex)]
    Path(f"{out}_feasible.txt").write_text("\n".join(map(str, feasible[: args.n])) + "\n")
    Path(f"{out}_all.txt").write_text("\n".join(str(r[0]) for r in rows[: args.n]) + "\n")
    earlies = np.array([r[1] for r in rows])
    print(f"{len(rows)} candidate {args.segments}x{args.flights}-flight streams (gaps {args.gap[0]:.0f}-{args.gap[1]:.0f} s): "
          f"{len(feasible)} feasible ({len(feasible) / len(rows):.1%}; {_rule(args)}); "
          f"largest early arrival median {np.median(earlies):.0f} s, 90th pct {np.percentile(earlies, 90):.0f} s, max {earlies.max():.0f} s")
    print(f"kept {min(args.n, len(feasible))} feasible -> {out}_feasible.txt; first {args.n} unfiltered -> {out}_all.txt")


if __name__ == "__main__":
    main()
