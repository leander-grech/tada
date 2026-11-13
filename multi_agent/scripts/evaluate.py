"""Offline evaluation: load a checkpoint, run episodes, print summary metrics."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import ray

from atc_env.timing_reporting import summarize_timing_coverage
from common.runtime_log import (
    colorize_runtime_label,
    force_runtime_log_color,
    runtime_print,
)
from common.seeding import seed_all
from config.config import Config
from evaluation.attempts import (
    SeedOutcome,
)
from evaluation.loading import build_eval_algo
from evaluation.run import (
    METRIC_COLS,
    OUTCOME_ORDER,
    OUTCOME_SORT_ORDER,
    EvalState,
    build_summary,
    coverage_shortfall,
    load_seed_manifest,
    resolve_attempt_workers,
    resolve_solve_threshold,
    run_attempts_mode,
    seed_sort_key,
)
from evaluation.seeds import build_evaluation_seed_manifest, operational_seeds


def _print_stats_block(label: str, episodes: list[dict[str, Any]]) -> None:
    """Print mean/std/min/max stats for a group of episodes."""
    if not episodes:
        return
    print(f"\n  {colorize_runtime_label(label)} (n={len(episodes)})")
    print(f"  {'Metric':<16} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-' * 55}")
    coverage = summarize_timing_coverage(ep.get("final_timing") for ep in episodes)
    print(
        f"  Final timing targets: exact={coverage['exact_count']}/{coverage['target_count']}, interval={coverage['interval_count']}, unknown={coverage['unknown_count']}"
    )
    for key, col_label, _, _, _ in METRIC_COLS[:-1]:
        arr = np.array([ep[key] for ep in episodes if ep.get(key) is not None])
        col_label = f"{col_label} ({len(arr)}/{len(episodes)})"
        if not len(arr):
            print(f"    {col_label:<14} unknown")
            continue
        print(
            f"    {col_label:<14} {arr.mean():>8.2f}  {arr.std():>8.2f}  "
            f"{arr.min():>8.2f}  {arr.max():>8.2f}"
        )


def _print_summary(
    scenarios: list[dict[str, Any]],
    *,
    sub_episode_count: int,
    solve_threshold_s: float,
) -> None:
    n = len(scenarios)
    if n == 0:
        print("\nNo full episodes completed.")
        if sub_episode_count > 0:
            print(
                f"  ({sub_episode_count} sub-episode callbacks fired but no environment"
                " episode terminated naturally — timestep budget may be too small)"
            )
        else:
            print("  (no on_episode_end callbacks fired — evaluation may not have run)")
        return

    scenarios = sorted(
        scenarios,
        key=lambda ep: (
            OUTCOME_SORT_ORDER.get(ep.get("outcome"), len(OUTCOME_SORT_ORDER)),
            float(ep["init_dev"]) if ep.get("init_dev") is not None else float("inf"),
            *seed_sort_key(ep.get("seed")),
        ),
    )
    outcomes = Counter(ep["outcome"] for ep in scenarios)
    width = 72

    print(f"\n{'=' * width}")
    print(f"  Evaluation Summary ({n} scenarios, solve threshold {solve_threshold_s:g}s)")
    print(f"{'=' * width}")

    # Per-scenario table
    metric_hdr = "".join(f"  {hdr:>{w + len(sfx)}}" for _, hdr, w, _, sfx in METRIC_COLS)
    print(f"\n  {'Seed':>12}  {'Outcome':<14}{metric_hdr}")
    print(f"  {'-' * 76}")
    for ep in scenarios:
        seed_str = str(ep["seed"]) if ep["seed"] is not None else "?"
        cells = "".join(
            (
                f"  {ep[key]:>{w}{fmt}}{sfx}"
                if ep.get(key) is not None
                else f"  {'unknown':>{w}}{sfx}"
            )
            for key, _, w, fmt, sfx in METRIC_COLS
        )
        outcome = colorize_runtime_label(str(ep["outcome"]))
        print(f"  {seed_str:>12}  {outcome:<14}{cells}")

    # Outcome distribution
    print("\n  Outcomes")
    print(f"  {'-' * 40}")
    ordered_outcomes = list(OUTCOME_ORDER)
    ordered_outcomes.extend(
        sorted(reason for reason in outcomes if reason not in OUTCOME_SORT_ORDER)
    )
    for reason in ordered_outcomes:
        ct = outcomes.get(reason, 0)
        outcome = colorize_runtime_label(reason)
        print(f"    {outcome:<20} {ct:>3} / {n}  ({100.0 * ct / n:5.1f}%)")

    # Aggregated stats (all metric columns except the last, which is AC)
    _print_stats_block("All", scenarios)

    for reason in ordered_outcomes:
        group = [ep for ep in scenarios if ep["outcome"] == reason]
        if group:
            _print_stats_block(reason, group)

    print(f"{'=' * width}\n")


def _print_attempts_report(report: dict[str, Any], outcomes: list[SeedOutcome]) -> None:
    """Print the pass@k rollup as a plain table."""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  Attempts to Solve ({report['scenarios']} scenarios, cap {report['attempt_cap']})")
    print(f"{'=' * width}\n")

    det_rate = report["deterministic_solved_rate"]
    print(f"  {'episodes run':<28}{report['episodes_run']:>10}")
    print(f"  {'solve threshold':<28}{report['solve_threshold_s']:>9.4g}s")
    if det_rate is not None:
        print(f"  {'deterministic solved':<28}{det_rate:>9.1%}")
    print(f"  {'pass@1 (stochastic)':<28}{report['pass_at_1']:>9.1%}")
    print(f"  {f'pass@{report["attempt_cap"]}':<28}{report['pass_at_cap']:>9.1%}")
    print(f"  {'never solved':<28}{report['censored']:>10}")
    mean_attempts = report["mean_attempts_to_success"]
    mean_attempts_text = "n/a" if mean_attempts is None else f"{mean_attempts:.2f}"
    print(f"  {'mean attempts to success':<28}{mean_attempts_text:>10}")

    print(f"\n  {'k':>4}  {'pass@k':>8}")
    print(f"  {'-' * 16}")
    for k, value in enumerate(report["pass_at_k"], start=1):
        print(f"  {k:>4}  {value:>7.1%}")

    unsolved = sorted(o.seed for o in outcomes if not o.solved)
    if unsolved:
        preview = ", ".join(str(s) for s in unsolved[:12])
        suffix = " ..." if len(unsolved) > 12 else ""
        print(f"\n  never solved: {preview}{suffix}")
    print(f"\n{'=' * width}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline evaluation of a trained PPO checkpoint.",
    )
    parser.add_argument(
        "-d",
        "--model-directory",
        dest="model_dir",
        required=True,
        help="Path to RLlib PPO checkpoint directory.",
    )
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=None,
        help="Episodes to run (default: seed manifest length, otherwise configured eval seeds).",
    )
    parser.add_argument(
        "-a",
        "--max-aircraft",
        type=int,
        default=None,
        help=f"Max aircraft count (default: {Config.MAX_AIRCRAFT_COUNT_EVAL}).",
    )
    parser.add_argument(
        "-m",
        "--max-steps",
        type=int,
        default=None,
        help=f"Max steps per episode (default: {Config.EPISODE_STEPS}).",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=None,
        help=(
            "Seed manifest for paired evaluation: JSON list, line-delimited text, "
            "or a CSV with a 'seed' column (e.g. a prior attempts_per_seed.csv)."
        ),
    )
    parser.add_argument(
        "--write-summary-json",
        type=Path,
        default=None,
        help="Write a machine-readable evaluation summary JSON to this path.",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Sample actions from the policy distribution instead of the greedy argmax.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help=(
            "Stochastic attempts per scenario (default: 1). Values above 1 imply "
            "--explore and switch to the retry-until-solved (pass@k) report."
        ),
    )
    parser.add_argument(
        "--attempts-out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the attempts report: attempts_raw.csv, "
            "attempts_per_seed.csv, attempts_summary.json and attempts_to_solve.png."
        ),
    )
    parser.add_argument(
        "--no-deterministic-baseline",
        action="store_true",
        help="Skip the extra greedy pass that anchors the pass@k chart.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Remote env runners to fan each attempts round across (default: "
            f"min({Config.EVAL_ENV_RUNNERS}, scenario count); 1 keeps sampling "
            "in the driver process). Only valid in attempts mode. Each runner "
            "takes a disjoint slice of the scenario manifest; evaluation is "
            "single-threaded per process, so this is the way to use more than "
            "one core."
        ),
    )
    parser.add_argument(
        "--solve-threshold",
        type=float,
        default=Config.OPERATIONAL_SUCCESS_S,
        metavar="SECONDS",
        help=(
            "Absolute time-deviation window inside which every arrival target "
            f"counts as solved (default: {Config.OPERATIONAL_SUCCESS_S:g} s, the "
            "operational window). Observations, reward coefficients and "
            "candidate selection keep the trained window, but this bound is also "
            "the episode-stop rule and picks the terminal reward branch, so "
            "episode returns and step counts only compare across runs graded "
            "with the same threshold. An episode that already infringed ends as "
            "an infringement at any threshold."
        ),
    )

    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        help="Authenticated development inputs; heldout/replacements are denied",
    )
    parser.add_argument(
        "--benchmark-output", type=Path, help="New immutable benchmark result directory"
    )
    return parser


def main() -> None:
    force_runtime_log_color()
    parser = build_parser()
    args = parser.parse_args()

    if args.benchmark_manifest is not None:
        from evaluation.benchmark.evaluate import run_development

        if args.benchmark_output is None:
            parser.error("--benchmark-manifest requires --benchmark-output")
        run_development(args.model_dir, args.benchmark_manifest, args.benchmark_output)
        return

    checkpoint = Path(args.model_dir).absolute()
    if not checkpoint.is_dir():
        runtime_print("Evaluate", "error", message=f"No such directory: {checkpoint}")
        sys.exit(1)

    from evaluation.checkpoint_adapter import inspect_inference_checkpoint

    saved_run = inspect_inference_checkpoint(checkpoint)
    if args.max_aircraft is None:
        args.max_aircraft = saved_run.scenario["aircraft_count"]
    if args.max_steps is None:
        args.max_steps = saved_run.scenario["episode_steps"]

    default_seeds = operational_seeds(saved_run, args.num_episodes or 10)
    try:
        eval_seeds = (
            load_seed_manifest(args.seed_manifest)
            if args.seed_manifest is not None
            else default_seeds
        )
    except ValueError as exc:
        runtime_print("Evaluate", "error", message=str(exc))
        sys.exit(1)

    num_episodes = int(args.num_episodes) if args.num_episodes is not None else len(eval_seeds)
    if num_episodes <= 0:
        runtime_print("Evaluate", "error", message="num-episodes must be positive")
        sys.exit(1)

    attempt_cap = max(1, int(args.attempts))
    attempts_mode = attempt_cap > 1 or args.attempts_out_dir is not None
    if attempts_mode and args.write_summary_json is not None:
        runtime_print(
            "Evaluate",
            "error",
            message="--write-summary-json is not produced in attempts mode; use --attempts-out-dir",
        )
        sys.exit(1)
    if args.workers is not None and not attempts_mode:
        runtime_print(
            "Evaluate",
            "error",
            message="--workers only applies to attempts mode (--attempts, --attempts-out-dir)",
        )
        sys.exit(1)

    try:
        solve_threshold_s = resolve_solve_threshold(args.solve_threshold)
    except ValueError as exc:
        runtime_print("Evaluate", "error", message=str(exc))
        sys.exit(1)

    workers: int | None = None

    if attempts_mode:
        # The env walks `evaluation_seeds` and wraps modulo its *own* length, so
        # the manifest must be exactly the scenario set each round replays.  The
        # randomize quota the plain path uses cannot serve here: it draws fresh
        # seeds as it goes, so round 2 would face new scenarios instead of
        # retrying round 1's.
        if args.seed_manifest is not None:
            # An explicit manifest supplies the ordered scenario set.  An
            # explicit --num-episodes may truncate it; --workers only shards
            # this resolved set and does not change its seeds.
            eval_seeds = eval_seeds[:num_episodes]
            num_episodes = len(eval_seeds)
        else:
            # Materializing the draws the env would have made keeps the scenario
            # set identical to a plain run of the same size.
            eval_seeds = build_evaluation_seed_manifest(
                eval_seeds,
                num_episodes,
                rng_seed=Config.SEED,
            )

        try:
            workers = resolve_attempt_workers(
                args.workers,
                num_episodes,
                configured_workers=Config.EVAL_ENV_RUNNERS,
            )
        except ValueError as exc:
            runtime_print("Evaluate", "error", message=str(exc))
            sys.exit(1)

    eval_seeds = eval_seeds[:num_episodes]
    if len(eval_seeds) != num_episodes:
        raise ValueError("num-episodes exceeds the authenticated seed manifest")
    seed = Config.SEED
    seed_all(seed, deterministic=True)

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray\.")
    ray.init(ignore_reinit_error=True)
    runtime_print(
        "Evaluate",
        "start",
        checkpoint=checkpoint,
        episodes=num_episodes,
        max_aircraft=args.max_aircraft,
        max_steps=args.max_steps,
        solve_threshold_s=solve_threshold_s,
    )

    eval_env_config = {
        "evaluation_mode": True,
        "max_aircraft_count": args.max_aircraft,
        "max_steps_per_episode": args.max_steps,
        "evaluation_seeds": eval_seeds,
        "evaluation_randomize_scenario_count": 0,
        "success_time_dev_s": solve_threshold_s,
    }

    if attempts_mode:
        assert workers is not None
        run_attempts_mode(
            checkpoint,
            eval_env_config=eval_env_config,
            seed=seed,
            scenarios=eval_seeds,
            cap=attempt_cap,
            run_baseline=not args.no_deterministic_baseline,
            out_dir=args.attempts_out_dir,
            max_aircraft=args.max_aircraft,
            workers=workers,
            solve_threshold_s=solve_threshold_s,
            show_report=_print_attempts_report,
        )
        return
    timestep_budget = int(args.max_steps * num_episodes * 1.1)

    state = EvalState()
    algo = build_eval_algo(
        checkpoint,
        eval_env_config=eval_env_config,
        callbacks=[],
        timestep_budget=timestep_budget,
        seed=seed,
        explore=args.explore,
    )
    # Plain mode walks the manifest in order and stops at `num_episodes`
    # scenarios, so its expected set is the manifest prefix of that size.
    expected_seeds = list(eval_seeds[:num_episodes])
    from evaluation.attempts import completed_summary_row
    from evaluation.sampling import sample_completed_round

    try:
        for result in sample_completed_round(algo, seeds=expected_seeds, rng_seed=seed):
            state.collected[result.attempt_id] = completed_summary_row(result)
    finally:
        algo.stop()
        ray.shutdown()

    _print_summary(
        list(state.collected.values()),
        sub_episode_count=state.sub_episode_count,
        solve_threshold_s=solve_threshold_s,
    )

    missing = coverage_shortfall((row["seed"] for row in state.collected.values()), expected_seeds)
    if missing or len(state.collected) < num_episodes:
        runtime_print(
            "Evaluate",
            "error",
            message=(
                f"incomplete coverage: {len(state.collected)}/{num_episodes} scenarios; "
                f"missing={missing}"
            ),
        )
        sys.exit(1)

    if args.write_summary_json is not None:
        summary = build_summary(
            list(state.collected.values()),
            sub_episode_count=state.sub_episode_count,
            checkpoint=checkpoint,
            solve_threshold_s=solve_threshold_s,
            seed_manifest=eval_seeds,
        )
        args.write_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        runtime_print("Evaluate", "summary", path=args.write_summary_json)


if __name__ == "__main__":
    main()
