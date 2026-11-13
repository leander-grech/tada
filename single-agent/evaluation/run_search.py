"""Evaluate selected TMB/PMS policies with receding-horizon search (K=8, M=10). Record
executed per-episode, per-aircraft and optional pick data. Attempt zero and subsequent
independently seeded candidates share the configured scenario cohort; input model and
paths are explicit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import tempfile
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import evaluation_records as pr  # noqa: E402
from analysis import rollout_search as rs  # noqa: E402
from analysis import score_windowed as sw  # noqa: E402

_OBS: pr.EpisodeRecorder | None = None
_CFG: dict = {}


def _init(model: str, flights: int, use_case: int, K: int, M: int, picks: bool = False) -> None:
    global _OBS
    sw._init_worker([model], flights, False, 1, (120.0, 900.0), 0, 0, use_case)
    _OBS = pr.EpisodeRecorder(use_case)
    _CFG.update(K=K, M=M, picks=picks)


def _episode(job: tuple[int, int]) -> tuple[dict, list[dict]]:
    seed, attempt = job
    _OBS.records = []
    row = rs.search_episode(sw._ENV, sw._MODELS[0], seed, _CFG["K"], _CFG["M"], attempt=attempt, observer=_OBS)
    (rec,) = _OBS.records
    names = list(rec["picks"])
    ep = {"seed": seed, "attempt": attempt, **{k: row[k] for k in (
        "solved", "hard_solved", "all_on_time", "sep", "on_time", "landed", "n_flights", "mean_dev", "max_dev",
        "end_reason", "steps", "clearances", "bracket_score", "n_plans", "n_rollouts", "rollout_steps",
        "violations", "replay_mismatch", "fallbacks", "wall_s")},
          "n_picks": rec["n_picks"], "noop": rec["noop"], "noop_forced": rec["noop_forced"],
          "noop_closing": rec["noop_closing"], "warning_pairs": rec["warning_pairs"],
          **{f"n_{n}": rec["picks"][n] for n in names}}
    idx = {f["callsign"]: f["index"] for f in rec["flights"]}
    picks = [{"seed": seed, "attempt": attempt, "aircraft_index": idx.get(r["callsign"], -1), **r}
             for r in rec.get("pick_log", [])] if _CFG.get("picks") else []
    ac = [{"seed": seed, "attempt": attempt, "aircraft_index": f["index"], "callsign": f["callsign"],
           "eta_s": round(f["eta_s"], 2), "initial_dev_s": round(f["initial"], 2),
           "initial_resolved": f["initial_resolved"], "aman_ttl_s": round(f["aman_ttl"], 3),
           "final_dev_s": round(f["final"], 2),
           "final_source": f["final_source"], "clearances": f["clearances"]} for f in rec["flights"]]
    return ep, ac, picks


def _append(path: Path, rows: list[dict]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(args, seeds, attempts, config, use_case):
    from analysis.lookahead import vecnormalize_path_for

    import flight_simulator

    native = (importlib.import_module("flight_simulator.flight_simulator")
              if hasattr(flight_simulator, "__path__") else flight_simulator)
    model = Path(args.model).resolve()
    config_root = Path(config.PROCEDURE_CONFIG_ROOTS[use_case] if
                       config.SCENARIO_GENERATOR == "procedure" else config.SIMULATOR_CONFIG_ROOT)
    # Only the selected scenario inputs affect the generated streams.
    scenario = {str(p.relative_to(config_root)): _hash_file(p)
                for p in sorted(config_root.rglob("*")) if p.is_file() and p.suffix in {".toml", ".json"}}
    return {
        "model_sha256": _hash_file(model),
        "normalization_sha256": _hash_file(vecnormalize_path_for(str(model))),
        "metadata_sha256": [_hash_file(d / "run_meta.json") for d in (model.parent, model.parent.parent)],
        "procedure": args.procedure, "seeds": seeds, "attempts": attempts,
        "K": args.K, "M": args.M, "flights": args.flights, "picks": args.picks,
        "generator": config.SCENARIO_GENERATOR, "scenario_inputs": scenario,
        "native_sha256": _hash_file(Path(native.__file__)),
        "sector_inputs": {p.name: _hash_file(p) for p in sorted(
            (ROOT / "static/sector_source/tada_scenario_sectors").glob("*.csv"))},
        "action_set": config.ACTION_SET, "airport_mask": config.AIRPORT_MASK,
        "max_picks": config.MAX_PICKS, "sequence_obs": config.SEQUENCE_OBS,
        "mask_frame_fix": config.MASK_FRAME_FIX, "turn_mask_fix": config.TURN_MASK_FIX,
    }


def _read_table(path):
    if not path.exists() or not path.stat().st_size:
        return [], [], False
    with path.open(newline="") as fh:
        content = fh.read()
    torn = not content.endswith("\n")
    if torn and "\n" not in content:
        return [], [], True  # The header itself was interrupted.
    reader = csv.DictReader(io.StringIO(content, newline=""))
    fields = reader.fieldnames or []
    rows = list(reader)
    if torn and rows:
        rows.pop()
    return fields, rows, torn


def _key(row):
    try:
        if None in row or any(v is None for v in row.values()):
            return None
        return int(row["seed"]), int(row["attempt"])
    except (KeyError, TypeError, ValueError):
        return None


def _rewrite(path, fields, rows):
    if not path.exists():
        return
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            if fields:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _complete_rows(paths, flights, picks_requested, expected):
    tables = [_read_table(path) for path in paths]
    required = [{"seed", "attempt", "n_flights", "n_picks"},
                {"seed", "attempt", "aircraft_index"}, {"seed", "attempt", "pick"}]
    for path, (fields, _, torn), columns in zip(paths, tables, required):
        if (fields and not columns.issubset(fields)) or (
                not fields and not torn and path.exists() and path.stat().st_size):
            sys.exit("Existing results contain a malformed CSV header")
    grouped = []
    for _, rows, _ in tables:
        groups = {}
        for row in rows:
            groups.setdefault(_key(row), []).append(row)
        grouped.append(groups)
    if any(key is not None and key not in expected for groups in grouped for key in groups):
        sys.exit("Existing results contain seed/attempt pairs outside this run")
    done = set()
    for key, episodes in grouped[0].items():
        if key is None or len(episodes) != 1:
            continue
        try:
            ep = episodes[0]
            indices = [int(r["aircraft_index"]) for r in grouped[1].get(key, [])]
            if float(ep["n_flights"]) != flights or sorted(indices) != list(range(flights)):
                continue
            if picks_requested:
                picks = [int(r["pick"]) for r in grouped[2].get(key, [])]
                if sorted(picks) != list(range(int(ep["n_picks"]))):
                    continue
            done.add(key)
        except (KeyError, TypeError, ValueError):
            continue
    # Remove incomplete pairs from every table before recomputation. Rewrite episodes first,
    # so another interruption cannot leave a completed marker for partially repaired rows.
    for path, (fields, rows, torn) in zip(paths, tables):
        kept = [row for row in rows if _key(row) in done]
        if torn or len(kept) != len(rows):
            _rewrite(path, fields, kept)
    return done


def main() -> None:
    from actions.action_set import Action, sync_obs_width_to_checkpoint
    from config.config import Config

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="compatible policy model .zip")
    ap.add_argument("--procedure", choices=("tmb", "pms"), required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/search"))
    ap.add_argument("--seeds", type=int, default=100, help="first N of Config.SCENARIO_EVAL_SEEDS (the shared 100 seeds)")
    ap.add_argument("--seed-file", default=None,
                    help="fly these scenario seeds instead (one per line, e.g. seeds.txt); --seeds caps how many")
    ap.add_argument("--attempts", default="0-20", help="attempt indices, e.g. 0-20 or 0 or 1-5")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--flights", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", action="store_true", help="resume compatible runs, skipping only complete episode outputs")
    ap.add_argument("--picks", action="store_true",
                    help="also write <out>/<proc>_picks.csv: one row per executed pick (time, pick 1 or 2 of the "
                         "decision interval, aircraft, clearance, genuine / forced / step-closing do-nothing)")
    args = ap.parse_args()

    use_case = 1 if args.procedure == "tmb" else 2
    sync_obs_width_to_checkpoint(args.model)
    for a in Action:
        pr.category_of(a.name)
    lo, _, hi = args.attempts.partition("-")
    attempts = list(range(int(lo), int(hi or lo) + 1))
    if args.seed_file:
        seeds = [int(x) for x in Path(args.seed_file).read_text().split()][: args.seeds]
    else:
        seeds = [int(s) for s in Config.SCENARIO_EVAL_SEEDS][: args.seeds]
    f_ep, f_ac = args.out / f"{args.procedure}_episodes.csv", args.out / f"{args.procedure}_aircraft.csv"
    f_picks = args.out / f"{args.procedure}_picks.csv"
    f_settings = args.out / f"{args.procedure}_run_settings.json"
    paths = [f_ep, f_ac, f_picks]
    settings = _settings(args, seeds, attempts, Config, use_case)
    if args.resume:
        if f_settings.exists():
            try:
                previous = json.loads(f_settings.read_text())
            except (ValueError, OSError) as exc:
                sys.exit(f"Cannot read resume settings: {exc}")
            if previous != settings:
                sys.exit("Resume settings do not match the existing run")
        elif any(path.exists() for path in paths):
            sys.exit("Cannot resume existing results without run settings")
    elif f_settings.exists() or any(path.exists() for path in paths):
        sys.exit(f"{f_ep} exists: pass --resume to continue it, or move it away")
    args.out.mkdir(parents=True, exist_ok=True)
    if not f_settings.exists():
        fd, name = tempfile.mkstemp(dir=args.out, prefix=f".{f_settings.name}.")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(settings, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(name, f_settings)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    done = _complete_rows(paths, args.flights, args.picks, {(s, a) for s in seeds for a in attempts}) if args.resume else set()
    # attempt-major: every seed's attempt 0 first, so a partial run already covers all seeds
    jobs = [(s, a) for a in attempts for s in seeds if (s, a) not in done]
    print(f"best agent ({args.procedure.upper()}): {args.model}\n  {len(seeds)} seeds x attempts {attempts[0]}..{attempts[-1]}"
          f" = {len(jobs)} episodes to fly ({len(done)} already done), K={args.K} M={args.M}, {args.workers} workers,"
          f" TADA_ACTION_SET={Config.ACTION_SET} max_picks={Config.MAX_PICKS} TADA_GENERATOR={Config.SCENARIO_GENERATOR}",
          flush=True)
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.model, args.flights, use_case, args.K, args.M, args.picks)) as pool:
        for i, (ep, ac, picks) in enumerate(pool.imap_unordered(_episode, jobs, chunksize=1), 1):
            _append(f_ac, ac)
            if picks:
                _append(f_picks, picks)
            _append(f_ep, [ep])
            print(f"[{i}/{len(jobs)}] seed {ep['seed']} attempt {ep['attempt']} solved={ep['solved']} sep={ep['sep']} "
                  f"on_time={ep['on_time']:.2f} wall={ep['wall_s']}s elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"-> {f_ep}\n-> {f_ac}")


if __name__ == "__main__":
    main()
