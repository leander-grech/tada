"""Offline checkpoint evaluation library behind ``scripts/evaluate.py``.

Seed manifests, completed rounds, pass@k attempts and the plain-run summary;
the CLI keeps argument parsing and console tables.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from operator import attrgetter
from pathlib import Path
from typing import Any

import numpy as np
import ray

from atc_env.timing_reporting import summarize_timing_coverage
from common.runtime_log import (
    runtime_print,
)
from evaluation.attempts import (
    AttemptRecord,
    SeedOutcome,
    write_attempts_per_seed,
    write_attempts_raw,
)
from evaluation.loading import build_eval_algo
from evaluation.stats import build_report, render_attempts_plot, summarize_seed
from outcomes import vocabulary

# Distinct RNG offset per attempt round, so no two rounds replay the same
# action-sampling stream.  Each shard re-seeds its own process, so a round
# repeats only for the same ``--workers`` split, and only as far as torch's
# kernels are reproducible: remote runners are seeded without the driver's
# ``deterministic=True`` pinning, which would re-exec the actor.
_ATTEMPT_SEED_STRIDE = 100_003


@dataclasses.dataclass
class EvalState:
    """Mutable evaluation state shared between the collector callback and summary."""

    # Keyed by seed: first occurrence wins (deterministic eval — repeats are identical).
    collected: dict[Any, dict[str, Any]] = dataclasses.field(default_factory=dict)
    sub_episode_count: int = 0  # filtered sub-episode callbacks (diagnostics)


# Numeric columns for the per-scenario table and aggregated stats.
# (dict_key, header, num_width, format_spec, suffix)
# Column width = num_width + len(suffix). Stats use all except the last (AC).
METRIC_COLS = [
    ("init_dev", "d_init/ac", 8, ".1f", "s"),
    ("final_dev", "d_exact/ac", 10, ".1f", "s"),
    ("forecast_infringement_severity", "FcstInfr", 7, ".2f", ""),
    ("steps", "Steps", 5, "d", ""),
    ("nz_actions", "NZAct", 5, "d", ""),
    ("nz_ex_trb", "ExTrb", 5, "d", ""),
    ("spd", "Spd", 5, "d", ""),
    ("aircraft", "AC", 3, "d", ""),
]


OUTCOME_ORDER = (
    vocabulary.SUCCESS,
    vocabulary.INFRINGEMENT,
    vocabulary.DELAYED,
    vocabulary.TIMEOUT,
    vocabulary.UNRESOLVED,
)


OUTCOME_SORT_ORDER = {outcome: idx for idx, outcome in enumerate(OUTCOME_ORDER)}


def seed_sort_key(seed: Any) -> tuple[bool, str]:
    return (seed is None, "" if seed is None else str(seed))


def load_seed_manifest(path: Path) -> list[int]:
    """Load paired evaluation seeds from a JSON list, CSV, or line-delimited text.

    CSV input needs a ``seed`` column, which makes any prior
    ``attempts_per_seed.csv`` directly reusable as the manifest for a re-run.
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Seed manifest is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        lines = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            raise ValueError(f"Seed manifest has no seeds: {path}")

        try:
            rows = list(csv.reader(lines))
        except csv.Error as exc:
            raise ValueError(f"Invalid seed manifest CSV: {path}") from exc

        header = [value.strip() for value in rows[0]]
        if "seed" in header:
            column = header.index("seed")
            try:
                seeds = [int(row[column].strip()) for row in rows[1:]]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid seed manifest CSV: {path}") from exc
        else:
            try:
                seeds = [int(row[0].strip()) for row in rows]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid seed manifest: {path}") from exc
    else:
        if not isinstance(parsed, list):
            raise ValueError("Seed manifest JSON must be a list")
        seeds = [int(seed) for seed in parsed]

    if not seeds:
        raise ValueError(f"Seed manifest has no seeds: {path}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seed manifest contains duplicate seeds: {path}")
    return seeds


def coverage_shortfall(collected: Iterable[int], eval_seeds: Sequence[int]) -> list[int]:
    """Manifest seeds absent from *collected*, in manifest order."""
    have = set(collected)
    return [seed for seed in eval_seeds if seed not in have]


def mean_metric(episodes: list[dict[str, Any]], key: str) -> float | None:
    if not episodes:
        return None
    known = [ep[key] for ep in episodes if ep.get(key) is not None]
    return float(np.mean(known)) if known else None


def build_summary(
    scenarios: list[dict[str, Any]],
    *,
    sub_episode_count: int,
    checkpoint: str | Path,
    seed_manifest: list[int],
    solve_threshold_s: float,
) -> dict[str, Any]:
    """Build machine-readable evaluation summary for paired checkpoint comparison."""
    scenarios = sorted(scenarios, key=lambda ep: seed_sort_key(ep.get("seed")))
    n = len(scenarios)
    outcomes = Counter(ep["outcome"] for ep in scenarios)
    outcome_summary = {
        reason: {
            "count": int(outcomes.get(reason, 0)),
            "rate": (float(outcomes.get(reason, 0)) / n) if n > 0 else 0.0,
        }
        for reason in OUTCOME_ORDER
    }

    return {
        "checkpoint": str(checkpoint),
        "solve_threshold_s": float(solve_threshold_s),
        "num_scenarios": n,
        "sub_episode_count": int(sub_episode_count),
        "seed_manifest": [int(seed) for seed in seed_manifest],
        "scenario_seeds": [int(ep["seed"]) for ep in scenarios if ep.get("seed") is not None],
        "outcomes": outcome_summary,
        "selection_axes": {
            "success_rate": outcome_summary["success"]["rate"],
            "infringement_rate": outcome_summary["infringement"]["rate"],
            "delayed_rate": outcome_summary["delayed"]["rate"],
            "final_deviation_mean": mean_metric(scenarios, "final_dev"),
            "learned_command_mean": mean_metric(
                [row.get("action_counts", {}) for row in scenarios], "learned_command_count"
            ),
            "speed_action_mean": mean_metric(scenarios, "spd"),
        },
        "non_dominated_axes": [
            "success_rate",
            "infringement_rate",
            "delayed_rate",
            "learned_command_mean",
            "speed_action_mean",
        ],
        "diagnostics": {
            "forecast_infringement_severity_mean": mean_metric(
                scenarios, "forecast_infringement_severity"
            ),
            "non_trombone_command_mean": mean_metric(scenarios, "nz_ex_trb"),
        },
        "final_timing_target_coverage": summarize_timing_coverage(
            ep.get("final_timing") for ep in scenarios
        ),
        "deviation_mean_semantics": "episode mean of exact-known target absolute errors; see target coverage",
        "metric_known_counts": {
            key: sum(ep.get(key) is not None for ep in scenarios) for key, *_ in METRIC_COLS
        },
        "scenarios": scenarios,
    }


def sample_round(
    algo: Any,
    *,
    scenarios: Sequence[int],
    attempt: int,
    rng_seed: int,
) -> list[AttemptRecord]:
    """Run one evaluation pass over the manifest and return its attempt records.

    Each runner is driven directly instead of through ``algo.evaluate()``.
    ``evaluate()`` accounts for its duration across the whole runner group, so a
    runner that returns fewer episodes than asked has the shortfall handed to a
    *different* runner, which replays its own already-covered slice while the
    short slice stays incomplete.  Its async fetch also drops in-flight results
    (``timeout_seconds=0.0``) and can hand them to a later call, which would
    file one round's episodes under another's attempt index.

    Records are shaped inside each owning runner process; the attempt index is
    stamped here because a runner does not need to know the driver's round.

    Raises:
        RuntimeError: If the round did not cover exactly *scenarios*.  pass@k is
            a fraction of the scenario set, so an incomplete round would quietly
            report the missing scenarios as unsolved.
    """
    from evaluation.attempts import completed_attempt
    from evaluation.sampling import sample_completed_round

    records = sample_completed_round(
        algo,
        seeds=scenarios,
        rng_seed=rng_seed,
        transport=completed_attempt,
        seed_of=attrgetter("seed"),
    )
    return [dataclasses.replace(record, attempt=attempt) for record in records]


def run_attempts(
    algo: Any,
    *,
    scenarios: Sequence[int],
    cap: int,
    base_seed: int,
    baselines: dict[int, AttemptRecord],
) -> tuple[list[AttemptRecord], list[SeedOutcome]]:
    """Run *cap* stochastic rounds over every scenario and summarize them.

    Every scenario gets the full attempt budget rather than stopping at its
    first success: pass@k is recovered offline from the first-success index,
    and keeping the rounds balanced stops the per-attempt distributions from
    being skewed toward the scenarios that needed the most retries.
    """
    attempts: list[AttemptRecord] = []
    by_seed: dict[int, list[AttemptRecord]] = {}
    for attempt in range(1, cap + 1):
        round_records = sample_round(
            algo,
            scenarios=scenarios,
            attempt=attempt,
            rng_seed=base_seed + attempt * _ATTEMPT_SEED_STRIDE,
        )
        attempts.extend(round_records)
        for record in round_records:
            by_seed.setdefault(record.seed, []).append(record)
        solved = sum(1 for records in by_seed.values() if any(r.solved for r in records))
        runtime_print(
            "Attempts",
            "round",
            attempt=f"{attempt}/{cap}",
            solved_so_far=f"{solved}/{len(by_seed)}",
        )

    outcomes = [
        summarize_seed(records, cap=cap, deterministic=baselines.get(seed))
        for seed, records in sorted(by_seed.items())
    ]
    return attempts, outcomes


def resolve_attempt_workers(
    requested: int | None,
    scenario_count: int,
    *,
    configured_workers: int,
) -> int:
    """Resolve and validate the attempts-mode runner count."""
    if requested is None:
        return min(configured_workers, scenario_count)
    if requested < 1:
        raise ValueError("--workers must be positive")
    if requested > scenario_count:
        raise ValueError(f"--workers {requested} exceeds the {scenario_count} scenarios")
    return requested


def resolve_solve_threshold(requested: float) -> float:
    """Resolve and validate the evaluation success window in seconds."""
    threshold_s = float(requested)
    if not math.isfinite(threshold_s) or threshold_s <= 0.0:
        raise ValueError("--solve-threshold must be a positive finite number of seconds")
    return threshold_s


def run_attempts_mode(
    checkpoint: Path,
    *,
    eval_env_config: dict[str, Any],
    seed: int,
    scenarios: Sequence[int],
    cap: int,
    run_baseline: bool,
    out_dir: Path | None,
    max_aircraft: int,
    workers: int,
    solve_threshold_s: float,
    show_report: Callable[[dict[str, Any], list[SeedOutcome]], None],
) -> None:
    """Run the baseline and stochastic phases, then emit the report.

    The baseline and stochastic phases intentionally use separate algorithm
    builds for phase isolation.  Sharing a runner would carry the environment's
    scenario-seed cursor, Ray's sampled-lifetime exploration counter, and
    connector/module state from one phase into the other, changing trajectories.

    ``workers`` above 1 fans each round across that many remote env runners,
    each holding a disjoint slice of the manifest.

    ``solve_threshold_s`` is the deviation window the environment graded against.
    It is recorded in the report and drawn on the plot, so a stored run never has
    to have its grading window inferred from the config in force later.
    """
    callbacks = []
    sharded_env_config = {**eval_env_config, "evaluation_seed_shards": workers}

    def _build(*, explore: bool) -> Any:
        return build_eval_algo(
            checkpoint,
            eval_env_config=sharded_env_config,
            callbacks=callbacks,
            timestep_budget=None,
            seed=seed,
            explore=explore,
            num_env_runners=0 if workers <= 1 else workers,
        )

    try:
        baselines: dict[int, AttemptRecord] = {}
        if run_baseline:
            runtime_print("Attempts", "baseline", scenarios=len(scenarios), explore=False)
            algo = _build(explore=False)
            try:
                for record in sample_round(
                    algo,
                    scenarios=scenarios,
                    attempt=0,
                    rng_seed=seed,
                ):
                    baselines[record.seed] = record
            finally:
                algo.stop()

        algo = _build(explore=True)
        try:
            attempts, outcomes = run_attempts(
                algo,
                scenarios=scenarios,
                cap=cap,
                base_seed=seed,
                baselines=baselines,
            )
        finally:
            algo.stop()
    finally:
        ray.shutdown()

    report = build_report(outcomes, attempts, cap=cap, solve_threshold_s=solve_threshold_s)
    show_report(report, outcomes)

    if out_dir is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "attempts_raw.csv"
    per_seed_path = out_dir / "attempts_per_seed.csv"
    summary_path = out_dir / "attempts_summary.json"
    plot_path = out_dir / "attempts_to_solve.png"

    write_attempts_raw(raw_path, attempts)
    write_attempts_per_seed(per_seed_path, outcomes)
    summary_path.write_text(
        json.dumps(
            {**report, "checkpoint": str(checkpoint), "max_aircraft": max_aircraft},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    render_attempts_plot(
        plot_path,
        outcomes,
        attempts,
        cap=cap,
        title=(
            f"Attempts to solve - {checkpoint.name} - "
            f"{max_aircraft} aircraft, {len(scenarios)} scenarios"
        ),
        solve_threshold_s=solve_threshold_s,
    )
    for path in (raw_path, per_seed_path, summary_path, plot_path):
        runtime_print("Attempts", "wrote", path=path)
