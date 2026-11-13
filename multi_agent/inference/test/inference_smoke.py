"""Multi-run smoke test and micro-benchmark for the InferenceEngine."""

# pylint: disable=wrong-import-position,duplicate-code

import argparse
import statistics
import sys
import time

import ray
from flight_simulator import World

from actions.actions import ActionRecord, count_nonzero_actions
from common.runtime_log import runtime_print
from inference.engine import EpisodeResult, InferenceEngine
from inference.service import (
    InferenceCandidate,
    NoValidSolutionError,
    _candidate_score,
    _cumulative_speed_change,
    select_best_candidate,
)
from inference.test.quiet_runtime import (
    configure_default_quiet_env,
    quiet_logs_enabled,
    set_ray_logger_levels,
    suppress_stdio,
)
from outcomes.selection import is_selectable

_COMPONENT = "Benchmark"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory path.")
    parser.add_argument("--world-json", required=True, help="World JSON matching the checkpoint scenario contract.")
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of inference runs (default: 5)."
    )
    parser.add_argument(
        "--print-actions",
        type=int,
        default=12,
        help="How many actions to print from run #1 (default: 12).",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Enable stochastic action sampling. Run #1 is always greedy; remaining use seeds.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------
def _write(line: str = "") -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _print_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    align: str | None = None,
) -> None:
    """Print right-aligned table with auto-computed column widths.

    align: string of 'l'/'r' per column (default all right-aligned).
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    if align is None:
        align = "r" * len(headers)

    def fmt(cells: list[str]) -> str:
        parts = [c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(cells, widths, align)]
        return "  " + "  ".join(parts)

    _write(fmt(headers))
    _write("  " + "  ".join("─" * w for w in widths))
    for row in rows:
        _write(fmt(row))


def main():
    configure_default_quiet_env()
    args = _parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    with open(args.world_json, "r", encoding="utf-8") as f:
        world_json = f.read()

    world_preview = World.from_json(world_json)
    runtime_print(
        _COMPONENT,
        "loaded",
        aircraft=len(world_preview.aircraft),
        age=f"{world_preview.age_in_seconds}s",
    )

    set_ray_logger_levels()
    quiet_logs = quiet_logs_enabled()
    ray_started = False
    engine = None

    durations: list[float] = []
    action_counts: list[float] = []
    rewards: list[float] = []
    signatures: list[tuple[tuple[int, str, int], ...]] = []
    first_actions: list[ActionRecord] = []
    run_rows: list[list[str]] = []
    episode_results: list[EpisodeResult] = []

    try:
        with suppress_stdio(quiet_logs):
            ray.init(
                ignore_reinit_error=True, runtime_env={}, logging_level="ERROR", log_to_driver=False
            )
            ray_started = True
            engine = InferenceEngine(args.checkpoint, deterministic=True, deterministic_seed=0)

        runtime_print(
            _COMPONENT,
            "start",
            runs=args.runs,
            explore=args.explore,
            checkpoint=args.checkpoint,
        )

        for idx in range(args.runs):
            world = World.from_json(world_json)
            t0 = time.perf_counter()
            explore = args.explore and idx > 0
            seed = idx if explore else None
            result = engine.run_episode_with_metrics(world, seed=seed, explore=explore)
            actions = result["actions"]
            elapsed = time.perf_counter() - t0

            total_reward = sum(float(a["reward"]) for a in actions)
            signature = tuple(
                (int(a["step"]), str(a["call_sign"]), int(a["action"])) for a in actions
            )

            if idx == 0:
                first_actions = actions

            durations.append(elapsed)
            action_counts.append(float(len(actions)))
            rewards.append(total_reward)
            signatures.append(signature)
            episode_results.append(result)

            cells = [
                str(idx + 1),
                f"{elapsed:.3f}s",
                str(len(actions)),
                f"{total_reward:+.3f}",
            ]
            if args.explore:
                cells.append(str(seed) if seed is not None else "-")
            run_rows.append(cells)

        # --- Runs table ---
        run_hdrs = ["Run", "Time", "Actions", "Reward"]
        if args.explore:
            run_hdrs.append("Seed")
        _write()
        _print_table(run_hdrs, run_rows)

        # --- Selection scoring ---
        # Shows how the inference pipeline would rank these runs as candidates.
        candidates = [InferenceCandidate(run_index=i, **ep) for i, ep in enumerate(episode_results)]
        try:
            best_idx = select_best_candidate(candidates, "diagnostic")["run_index"]
        except NoValidSolutionError:
            best_idx = None
        score_rows: list[list[str]] = []

        for i, candidate in enumerate(candidates):
            ep = candidate
            acts = ep["actions"]
            _, nz_ex_trb, _ = count_nonzero_actions(acts)
            eff = float(ep["cumulative_reward"]) / max(nz_ex_trb, 1)
            dev = ep["total_deviation"]
            spd = _cumulative_speed_change(acts)
            valid = is_selectable(candidate)
            score = _candidate_score(candidate)

            score_rows.append(
                [
                    str(i + 1),
                    str(ep["end_reason"] or "?"),
                    str(nz_ex_trb),
                    f"{eff:.2f}",
                    str(dev),
                    f"{spd:.1f}",
                    str(score),
                    "← best" if i == best_idx else "" if valid else "(filtered)",
                ]
            )

        _write()
        runtime_print(
            _COMPONENT,
            "scoring",
            note="lower wins",
            ranking="eligible safe success, conservative timing upper bound, all-command cost, stable run order",
        )
        _print_table(
            ["Run", "End", "ExTrb", "Eff", "Dev", "Spd", "Score", ""],
            score_rows,
            align="rlrrrrrl",
        )

        # --- Summary ---
        deterministic_ok = all(sig == signatures[0] for sig in signatures[1:])
        unique_signatures = len(set(signatures))

        sum_hdrs = ["", "mean", "std", "min", "max"]
        sum_rows: list[list[str]] = []

        def _add_summary(label: str, values: list[float], precision: int = 3) -> None:
            fmt = f".{precision}f"
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            sum_rows.append(
                [
                    label,
                    f"{mean:{fmt}}",
                    f"{std:{fmt}}",
                    f"{min(values):{fmt}}",
                    f"{max(values):{fmt}}",
                ]
            )

        _add_summary("Time", durations)
        _add_summary("Actions", action_counts, precision=2)
        _add_summary("Reward", rewards)

        _write()
        runtime_print(_COMPONENT, "summary")
        _print_table(sum_hdrs, sum_rows, align="lrrrr")

        _write()
        runtime_print(
            _COMPONENT,
            "signatures",
            deterministic_ok=deterministic_ok,
            unique=f"{unique_signatures}/{args.runs}",
        )

        # --- Actions detail ---
        showing = min(args.print_actions, len(first_actions))
        if showing > 0:
            act_hdrs = ["Step", "Callsign", "Act", "Position", "Reward"]
            act_rows = [
                [
                    str(a["step"]),
                    a["call_sign"],
                    str(a["action"]),
                    f"({a['x']:6.1f}, {a['y']:6.1f})",
                    f"{a['reward']:+.3f}",
                ]
                for a in first_actions[:showing]
            ]
            _write()
            runtime_print(_COMPONENT, "actions", run=1, showing=showing)
            _print_table(act_hdrs, act_rows, align="rlrlr")
    finally:
        if engine is not None:
            engine.shutdown()
        if ray_started and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
