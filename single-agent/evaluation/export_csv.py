"""Convert selected raw episode and aircraft records into the per-episode/per-aircraft
sheet layout. Keep procedure, seed, attempt, timing, sequence and per-action values
separate. Supply explicit input and output directories and validate complete expected
keys before using exported sheets as final evidence. Retain initial-resolution and
final-source flags with the underlying records; warning-pair presence is distinct from
realized loss of separation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

EPISODE_COLS = [
    "Agent Type (Multi|Single)",
    "Procedure type (PMS|TMB)",
    "Seed number (100 seeds)",
    "Type of policy (Deterministicx1|Stochasticx20) 0-20",
    "End reason (success|delayed|timeout|airspace_exit|Loss of separation)",
    "Total initial absolute time deviation (seconds)",
    "Total final absolute time deviation (seconds)",
    "AMAN sequence met? (Yes|No)",
    "Infringements (not conflicts) (Yes|No)",
]
# sheet group -> (column label, raw clearance names counted in it)
ACTIONS = {
    "TMB": [("Do nothing", None), ("Slow down", "SLOW_DOWN"), ("Speed up", "SPEED_UP"),
            ("Turn and return to route", "TURN_"), ("Extend trombone ", "LENGTHEN_TROMBONE"),
            ("Retract trombone", "SHORTEN_TROMBONE"), ("Skip waypoints", "SKIP_"), ("Vector to ILS", "VECTOR_TO_ILS")],
    "PMS": [("Do nothing", None), ("Slow down", "SLOW_DOWN"), ("Speed up", "SPEED_UP"),
            ("Turn and return to route", "TURN_"), ("Skip waypoints", "SKIP_"),
            ("Direct to merge point", "DIRECT_TO_MERGE_POINT")],
}
AIRCRAFT_COLS = [
    "Agent Type (Multi|Single)",
    "Procedure type (PMS|TMB)",
    "Seed number (100 seeds)",
    "Type of policy (Deterministicx1|Stochasticx20) 0-20",
    "Aircraft Index (0-19)",
    "TTL or TTG value (seconds) +ve for TTL and -ve for TTG",
    "Initial absolute time deviation (seconds)",
    "Final absolute time deviation (seconds)",
    "Landed according to AMAN sequence? (Yes | No)",
    "Number of non-'do nothing' clearances",
]
END_REASON = {"separation": "Loss of separation"}


def yes(b) -> str:
    return "Yes" if bool(b) else "No"


def sequence_flags(ac: pd.DataFrame) -> tuple[bool, dict[int, bool]]:
    """One episode's flights -> (AMAN sequence met, per aircraft index: landed in its AMAN position)."""
    ac = ac.sort_values("aircraft_index")
    t = ac.eta_s + ac.final_dev_s
    resolved = ac.final_source.ne("unresolved")
    rank = t.rank(method="first").astype(int) - 1
    met = bool(resolved.all() and (rank.values == ac.aircraft_index.values).all())
    per = {int(i): bool(src == "landed" and r == i)
           for i, src, r in zip(ac.aircraft_index, ac.final_source, rank)}
    return met, per


def action_counts(ep: pd.Series, proc: str) -> list:
    out = []
    for group, cols in ACTIONS.items():
        for _label, prefix in cols:
            if group != proc:
                out.append("")
            elif prefix is None:
                out.append(int(ep["noop"]))
            else:
                out.append(int(sum(ep[c] for c in ep.index if c.startswith(f"n_{prefix}"))))
    return out


def load_raw(raw: Path, agent: str, proc: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(episodes, aircraft) of one agent and procedure, in run_search.py's column names."""
    if agent in ("best", "search"):  # run_search.py output (v6: in <raw>/search/)
        d = raw / "search" if agent == "search" else raw
        f_ep, f_ac = d / f"{proc}_episodes.csv", d / f"{proc}_aircraft.csv"
    else:
        f_ep, f_ac = raw / f"{agent}_{proc}_extra.csv", raw / f"{agent}_{proc}_extra_aircraft.csv"
    if not (f_ep.exists() and f_ac.exists()):
        print(f"(no {f_ep.name} / {f_ac.name}, skipped)")
        return None
    eps = pd.read_csv(f_ep).rename(columns={"env_end_reason": "end_reason"})
    acs = pd.read_csv(f_ac)
    return (eps.drop_duplicates(["seed", "attempt"], keep="first"),
            acs.drop_duplicates(["seed", "attempt", "aircraft_index"], keep="first"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="best",
                    help="best | search (run_search output) | policy | greedy | veto1..veto4 (analysis/evaluation_records output)")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-file", type=Path, default=HERE / "seeds.txt",
                    help="row order")
    ap.add_argument("--attempts", type=int, default=20, help="keep attempts 0..N")
    args = ap.parse_args()
    from_seeds = [int(ln.split()[0]) for ln in args.seed_file.read_text().splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")][:100]
    order = {s: i for i, s in enumerate(from_seeds)}

    ep_rows, ac_rows = [], []
    for proc in ("tmb", "pms"):
        loaded = load_raw(args.raw, args.agent, proc)
        if loaded is None:
            continue
        eps, acs = loaded
        P = proc.upper()
        eps = eps[eps.attempt <= args.attempts]
        eps = eps.assign(_o=eps.seed.map(order)).sort_values(["_o", "attempt"])
        by_ep = dict(tuple(acs.groupby(["seed", "attempt"])))
        n_missing = 0
        for _, e in eps.iterrows():
            ac = by_ep.get((e.seed, e.attempt))
            if ac is None or len(ac) != int(e.n_flights):
                n_missing += 1
                continue
            met, per = sequence_flags(ac)
            infr = e.warning_pairs > 0 if "warning_pairs" in e.index else str(e.sep) == "True"
            ep_rows.append(["Single", P, int(e.seed), int(e.attempt), END_REASON.get(e.end_reason, e.end_reason),
                            round(ac.initial_dev_s.abs().sum(), 2), round(ac.final_dev_s.abs().sum(), 2),
                            yes(met), yes(infr), *action_counts(e, P)])
            for _, f in ac.sort_values("aircraft_index").iterrows():
                ttl = f.aman_ttl_s if "aman_ttl_s" in f.index else -f.initial_dev_s
                ac_rows.append(["Single", P, int(e.seed), int(e.attempt), int(f.aircraft_index),
                                round(ttl, 2) + 0.0, round(abs(f.initial_dev_s), 2), round(abs(f.final_dev_s), 2),
                                yes(per[int(f.aircraft_index)]), int(f.clearances)])
        print(f"{P}: {len(eps) - n_missing} episodes ({eps.seed.nunique()} seeds, attempts {eps.attempt.min()}..{eps.attempt.max()})"
              + (f", {n_missing} skipped (incomplete aircraft rows)" if n_missing else ""))

    ep_header = EPISODE_COLS + [f"Number of actions | {g} | {lab.strip()}" for g, cols in ACTIONS.items() for lab, _ in cols]
    args.out.mkdir(parents=True, exist_ok=True)
    for tab, header, rows in (("Per Episode", ep_header, ep_rows), ("Per Aircraft", AIRCRAFT_COLS, ac_rows)):
        for suffix, hdr in (("", True), (" - values", False)):
            p = args.out / f"TADA_single_agent_{args.agent} - {tab}{suffix}.csv"
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                if hdr:
                    w.writerow(header)
                w.writerows(rows)
            print(f"-> {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
