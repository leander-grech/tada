"""Construct training seed pools from scored stream inputs with explicit feasibility
thresholds and retained fractions. Supply the input CSV and output prefix; generated
pools are training inputs rather than evaluation evidence."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank in [0, 1]."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # ties share their mean rank
        idx = x == v
        ranks[idx] = ranks[idx].mean()
    return ranks / max(1, len(x) - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", help="<prefix>_all.csv from long_stream_seeds.py")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--max-abs-dev", type=float, default=900.0)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--switch-col", default="entry_inversions", help="the sequence-switch column")
    args = ap.parse_args()
    assert all(0 < f <= 1 for f in args.fractions) and args.fractions == sorted(args.fractions)

    from config.config import Config
    holdout = {int(s) for s in Config.SCENARIO_EVAL_SEEDS}
    rows = [r for r in csv.DictReader(open(args.csv))
            if float(r["max_abs_dev_s"]) <= args.max_abs_dev and int(r["seed"]) not in holdout]
    if not rows:
        raise SystemExit("no feasible streams in the pool")
    seeds = np.array([int(r["seed"]) for r in rows])
    sw = np.array([float(r[args.switch_col]) for r in rows])
    dev = np.array([float(r["max_abs_dev_s"]) for r in rows])
    score = 0.5 * _pct_rank(sw) + 0.5 * _pct_rank(dev)
    order = np.argsort(score, kind="stable")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(1 for _ in open(args.csv)) - 1
    print(f"{len(rows)} of {total} streams feasible (|dev| <= {args.max_abs_dev:.0f} s), validation seeds excluded")
    for k, f in enumerate(args.fractions):
        keep = order[: max(1, int(round(f * len(rows))))]
        Path(f"{out}_tier{k}.txt").write_text("\n".join(map(str, seeds[keep])) + "\n")
        print(f"  tier{k}: easiest {f:.0%} = {len(keep)} streams; {args.switch_col} median {np.median(sw[keep]):.0f} "
              f"(max {sw[keep].max():.0f}), max|dev| median {np.median(dev[keep]):.0f} s (max {dev[keep].max():.0f} s)")
    Path(f"{out}_feasible.txt").write_text("\n".join(map(str, seeds)) + "\n")


if __name__ == "__main__":
    main()
