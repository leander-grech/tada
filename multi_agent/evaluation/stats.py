"""Retry-until-solved ("attempts to solve") analysis for stochastic offline evaluation.

A greedy policy gives one answer per scenario. Sampling from the policy
distribution instead turns evaluation into a *pass@k* question: given ``k``
independent attempts at the same scenario, how often is at least one solved?

This module holds the pure statistics and plotting (records and CSV round trips
live in ``evaluation.attempts``) — so the sampling loop in ``evaluation.run`` stays small and
these helpers stay unit-testable without Ray or a checkpoint.

Every field emitted here is a native metric of this environment
(``last_end_reason`` plus the deviation / infringement figures used by
``atc_env.episode_metrics``). No external grading rubric is reconstructed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from atc_env.timing_reporting import summarize_timing_coverage
from evaluation.attempts import AttemptRecord, SeedOutcome, canonical_attempt, decode_attempts
from outcomes.selection import best_candidate

# A scenario counts as solved when the environment itself reports success.
# An attempt can end without a terminal result when the sampling budget delays it.

# The outcome classifier gives irreversible live LOS/exit priority. This label
# counts failed attempts, not unique events. infr_severity is a forecast
# diagnostic (including warning bands and symmetric aircraft contributions),
# never a realized event count or a candidate safety eligibility signal.

# Inclusive upper bounds for the "attempts to first success" histogram; labels
# are derived from these and clamped to the run's cap.
_ATTEMPT_BUCKET_EDGES: tuple[int, ...] = (1, 3, 7, 15)


def summarize_seed(
    attempts: list[AttemptRecord],
    *,
    cap: int,
    deterministic: AttemptRecord | None = None,
) -> SeedOutcome:
    """Collapse one seed's attempt list into a single outcome row."""
    if not attempts:
        raise ValueError("summarize_seed requires at least one attempt")

    decoded = decode_attempts(attempts)
    current = [candidate is not None for _, candidate in decoded]
    if any(current) and not all(current):
        raise ValueError("canonical and noncanonical attempts require separate seed summaries")
    attempts = [a for a, _ in decoded]
    if deterministic is not None:
        deterministic, baseline_candidate = canonical_attempt(deterministic)
        if (baseline_candidate is not None) != all(current):
            raise ValueError("canonical and noncanonical baselines require separate summaries")
    seed = attempts[0].seed
    success = min((a for a in attempts if a.solved), key=lambda a: a.attempt, default=None)
    if all(current):
        selected = best_candidate([candidate for _, candidate in decoded])
        best = next(
            (
                a
                for a in attempts
                if selected is not None and a.attempt_id == selected["attempt_id"]
            ),
            None,
        )
    else:
        # Rows without eligibility evidence are diagnostic only.
        best = min(attempts, key=lambda a: (a.max_dev is None, a.max_dev or 0.0))

    return SeedOutcome(
        seed=seed,
        solved=success is not None,
        attempts_used=len(attempts),
        first_success_attempt=success.attempt if success else None,
        censored=success is None and len(attempts) >= cap,
        best_max_dev=best.max_dev if best else None,
        best_end_reason=best.end_reason if best else None,
        deterministic_solved=None if deterministic is None else deterministic.solved,
        deterministic_end_reason=None if deterministic is None else deterministic.end_reason,
        deterministic_max_dev=None if deterministic is None else deterministic.max_dev,
    )


def pass_at_k(outcomes: list[SeedOutcome], cap: int) -> list[float]:
    """Fraction of scenarios solved within k attempts, for k = 1..cap.

    A scenario counts at every k at or above its first success, so the curve is
    monotonically non-decreasing.
    """
    total = len(outcomes)
    if total == 0:
        return [0.0] * cap
    firsts = [o.first_success_attempt for o in outcomes if o.first_success_attempt is not None]
    return [sum(1 for f in firsts if f <= k) / total for k in range(1, cap + 1)]


def attempt_buckets(outcomes: list[SeedOutcome], cap: int) -> list[tuple[str, int, bool]]:
    """Histogram of attempts-to-first-success as ``(label, count, is_unsolved)``.

    Bucket labels are clamped to *cap* so a range never advertises attempts the
    run could not have made.
    """
    buckets: list[tuple[str, int, bool]] = []
    lower = 1
    for upper in _ATTEMPT_BUCKET_EDGES:
        if lower > cap:
            break
        top = min(upper, cap)
        count = sum(
            1
            for o in outcomes
            if o.first_success_attempt is not None and lower <= o.first_success_attempt <= top
        )
        buckets.append((str(lower) if lower == top else f"{lower}-{top}", count, False))
        lower = upper + 1

    if lower <= cap:
        count = sum(
            1
            for o in outcomes
            if o.first_success_attempt is not None and o.first_success_attempt >= lower
        )
        buckets.append((str(lower) if lower == cap else f"{lower}-{cap}", count, False))

    unsolved = sum(1 for o in outcomes if o.first_success_attempt is None)
    buckets.append((f"not solved\nin {cap}", unsolved, True))
    return buckets


def _has_timing_coverage(attempt: AttemptRecord) -> bool:
    counts = (
        attempt.final_timing_exact_count,
        attempt.final_timing_interval_count,
        attempt.final_timing_unknown_count,
        attempt.final_timing_target_count,
    )
    return (
        all(type(value) is int and value >= 0 for value in counts) and sum(counts[:3]) == counts[3]
    )


def _has_exact_mean(attempt: AttemptRecord) -> bool:
    return (
        _has_timing_coverage(attempt)
        and (
            attempt.final_timing_source in {"owned_forecast", "unowned_forecast_world"}
            or (
                attempt.completed_result_json is not None
                and attempt.final_timing_source in {"live_completed", "verified_completed"}
            )
        )
        and attempt.final_timing_exact_count is not None
        and attempt.final_timing_exact_count > 0
        and attempt.final_dev_per_ac is not None
    )


def _final_deviation_summary(attempts: Sequence[AttemptRecord]) -> dict:
    reported = [a for a in attempts if _has_timing_coverage(a)]
    exact = [a for a in attempts if _has_exact_mean(a)]
    unverified = [a for a in attempts if a.final_dev_per_ac is not None and not _has_exact_mean(a)]
    coverage = summarize_timing_coverage(
        {
            "exact_count": a.final_timing_exact_count,
            "target_count": a.final_timing_target_count,
            "interval_count": a.final_timing_interval_count,
            "unknown_count": a.final_timing_unknown_count,
        }
        for a in reported
    )
    if not reported:
        # No report is unavailable, not evidence of zero targets.
        coverage.update(
            {
                key: None
                for key in ("exact_count", "target_count", "interval_count", "unknown_count")
            }
        )
    coverage["episodes_missing_coverage"] = len(attempts) - len(reported)
    return {
        "final_deviation_mean_semantics": "episode-weighted reported means; exact-known and legacy/unverified separated; not canonical eligibility",
        "exact_known_deviation_mean": _known_mean(a.final_dev_per_ac for a in exact),
        "exact_known_deviation_episode_count": len(exact),
        "legacy_unverified_deviation_mean": _known_mean(a.final_dev_per_ac for a in unverified),
        "legacy_unverified_deviation_episode_count": len(unverified),
        "final_timing_target_coverage": coverage,
    }


def summarize_attempt_metrics(attempts: list[AttemptRecord]) -> dict[str, Any]:
    """Mean per-episode behaviour across every stochastic attempt of a run.

    The pass@k rollup answers "how often does retrying help"; this answers
    "what does a single attempt look like", which is what checkpoint selection
    trades off - infringement share first, then timing.
    """
    decoded = decode_attempts(attempts)
    attempts = [a for a, _ in decoded]
    forecast = [
        a.infr_severity for a in attempts if a.infringement_source != "forecast_unavailable"
    ]
    n = len(attempts)
    if n == 0:
        return {"episodes": 0, "end_reason_share": {}}
    reason_counts: dict[str, int] = {}
    for record in attempts:
        reason = record.end_reason or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    shares = {reason: count / n for reason, count in reason_counts.items()}
    return {
        "episodes": n,
        "end_reason_share": dict(sorted(shares.items(), key=lambda kv: -kv[1])),
        "infringement_severity_mean": _known_mean(forecast),
        "forecast_infringement_severity_mean": _known_mean(forecast),
        "forecast_infringement_severity_known_count": len(forecast),
        "canonical_episode_count": sum(candidate is not None for _, candidate in decoded),
        "historical_diagnostic_episode_count": sum(candidate is None for _, candidate in decoded),
        "infringement_metric_role": "legacy_forecast_diagnostic_not_realized_safety",
        "max_dev_mean": _known_mean(a.max_dev for a in attempts),
        "max_dev_known_count": sum(a.max_dev is not None for a in attempts),
        "initial_deviation_known_count": sum(a.init_dev_per_ac is not None for a in attempts),
        "final_deviation_known_count": sum(a.final_dev_per_ac is not None for a in attempts),
        **_final_deviation_summary(attempts),
        "steps_mean": sum(a.steps for a in attempts) / n,
        "nz_actions_mean": sum(a.nz_actions for a in attempts) / n,
        "nz_ex_trb_mean": sum(a.nz_ex_trb for a in attempts) / n,
    }


def build_report(
    outcomes: list[SeedOutcome],
    attempts: list[AttemptRecord],
    *,
    cap: int,
    solve_threshold_s: float,
) -> dict[str, Any]:
    """Machine-readable rollup of the attempts run.

    ``solve_threshold_s`` is the deviation window the environment graded against,
    carried through so a stored report stays interpretable on its own.
    """
    total = len(outcomes)
    curve = pass_at_k(outcomes, cap)
    det = [o.deterministic_solved for o in outcomes if o.deterministic_solved is not None]
    first_successes = [
        o.first_success_attempt for o in outcomes if o.first_success_attempt is not None
    ]
    return {
        "scenarios": total,
        "attempt_cap": cap,
        "solve_threshold_s": float(solve_threshold_s),
        "episodes_run": len(attempts),
        "solved_within_cap": sum(1 for o in outcomes if o.solved),
        "censored": sum(1 for o in outcomes if o.censored),
        "pass_at_1": curve[0] if curve else 0.0,
        "pass_at_cap": curve[-1] if curve else 0.0,
        "pass_at_k": curve,
        "deterministic_solved_rate": (sum(det) / len(det)) if det else None,
        "mean_attempts_to_success": (
            sum(first_successes) / len(first_successes) if first_successes else None
        ),
    }


def _known_mean(values):
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def render_attempts_plot(
    path: Path,
    outcomes: list[SeedOutcome],
    attempts: list[AttemptRecord],
    *,
    cap: int,
    title: str,
    solve_threshold_s: float,
) -> None:
    """Render the three-panel attempts-to-solve figure.

    Panels: attempts needed, pass@k against the greedy baseline, and the
    distribution of worst-aircraft deviation across every attempt.  The third
    panel separates episode-weighted exact-known and unverified means
    and reports target coverage. Neither the plotted means nor coverage certify
    canonical evaluation eligibility.
    """
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_hist, ax_pass, ax_dev) = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")

    blue, orange, green, red, ink = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#333333"

    # --- attempts needed -------------------------------------------------
    buckets = attempt_buckets(outcomes, cap)
    labels = [b[0] for b in buckets]
    counts = [b[1] for b in buckets]
    colors = [red if b[2] else blue for b in buckets]
    ax_hist.bar(labels, counts, color=colors)
    ax_hist.set_title("Attempts needed", loc="left", fontweight="bold")
    ax_hist.set_xlabel("attempts to first success")
    ax_hist.set_ylabel("scenarios")
    for idx, (label, count, is_unsolved) in enumerate(buckets):
        if count and is_unsolved:
            ax_hist.text(
                idx, count, str(count), ha="center", va="bottom", color=red, fontweight="bold"
            )

    # --- pass@k ----------------------------------------------------------
    curve = pass_at_k(outcomes, cap)
    ks = list(range(1, cap + 1))
    ax_pass.plot(ks, curve, marker="o", color=blue, markersize=4)
    det = [o.deterministic_solved for o in outcomes if o.deterministic_solved is not None]
    if det:
        det_rate = sum(det) / len(det)
        ax_pass.axhline(det_rate, linestyle="--", color=ink, linewidth=1)
        # Below the line and mid-axis: the curve endpoint annotations sit above
        # it at both ends, so anything on that side collides.
        ax_pass.text(
            (1 + cap) / 2,
            det_rate - 0.03,
            f"deterministic {det_rate:.0%}",
            ha="center",
            va="top",
            color=ink,
        )
    ax_pass.set_xscale("log")
    ax_pass.set_xticks(ks if cap <= 10 else [1, 2, 3, 5, 10, cap])
    ax_pass.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_pass.set_ylim(0.0, 1.0)
    ax_pass.set_title("pass@k", loc="left", fontweight="bold")
    ax_pass.set_xlabel("attempts allowed (k)")
    ax_pass.set_ylabel("fraction of scenarios solved")
    if curve:
        ax_pass.annotate(
            f"{curve[0]:.2f}",
            (1, curve[0]),
            textcoords="offset points",
            xytext=(0, 10),
            fontweight="bold",
        )
        ax_pass.annotate(
            f"{curve[-1]:.2f}",
            (cap, curve[-1]),
            textcoords="offset points",
            xytext=(-6, 10),
            ha="right",
            fontweight="bold",
        )

    # --- deviation across attempts ---------------------------------------
    # Everything on this panel is named in one legend box rather than in inline
    # captions: the labels differ in length per run and per aircraft count, so
    # any hand-placed text eventually collides with its neighbour.
    devs = [a.max_dev for a in attempts if a.max_dev is not None]
    if devs:
        ax_dev.hist(
            devs,
            bins=30,
            color=blue,
            label=f"worst aircraft, known {len(devs)}/{len(attempts)} attempts",
        )
    ax_dev.axvline(
        solve_threshold_s,
        linestyle="--",
        color=ink,
        linewidth=1.4,
        label=f"solve threshold {solve_threshold_s:.0f}s",
    )
    summary = _final_deviation_summary(attempts)
    scenario = _known_mean(a.init_dev_per_ac for a in attempts)
    for mean, color, name in (
        (scenario, orange, "scenario episode mean/ac"),
        (
            summary["exact_known_deviation_mean"],
            green,
            f"episode-weighted exact-known mean ({summary['exact_known_deviation_episode_count']} episodes)",
        ),
        (
            summary["legacy_unverified_deviation_mean"],
            red,
            f"legacy/unverified episode mean ({summary['legacy_unverified_deviation_episode_count']} episodes)",
        ),
    ):
        if mean is not None:
            ax_dev.axvline(
                mean, linestyle="--", color=color, linewidth=1.8, label=f"{name} {mean:.0f}s"
            )
    coverage = summary["final_timing_target_coverage"]

    def count(name):
        value = coverage[name + "_count"]
        return "unavailable" if value is None else str(value)

    ax_dev.text(
        0,
        -0.23,
        f"Reported target coverage: exact {count('exact')} / targets {count('target')}\n"
        f"interval {count('interval')}; unknown {count('unknown')}; "
        f"missing coverage {coverage['episodes_missing_coverage']}/{len(attempts)} episodes\n"
        "Means weight episodes equally, not pooled targets; no eligibility claim.",
        transform=ax_dev.transAxes,
        va="top",
        fontsize=8,
    )
    ax_dev.set_title("Performance per attempt", loc="left", fontweight="bold")
    ax_dev.set_xlabel("deviation (s)")
    ax_dev.set_ylabel("attempts")
    ax_dev.legend(loc="upper right", fontsize=8, framealpha=0.95)

    for axis in (ax_hist, ax_pass, ax_dev):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
