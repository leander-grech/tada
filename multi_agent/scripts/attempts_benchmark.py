"""Run the retry-until-solved (pass@k) benchmark over a checkpoint x aircraft matrix.

One `evaluate --attempts` run answers one question: how does *this* checkpoint do
at *this* aircraft count.  Checkpoint selection needs the grid, so this driver
runs it, keeps every run's artifacts, and emits one comparison table over them.

Each cell runs in its own `evaluate` process.  That is deliberate: evaluation
pins BLAS and torch to a single thread by re-executing the interpreter, which a
same-process loop could not survive, and a fresh process guarantees Ray is torn
down between cells.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common.jsonio import read_json
from common.runtime_log import force_runtime_log_color, runtime_print
from config.config import Config
from evaluation.attempts import (
    DELAYED_END_REASON,
    INFRINGEMENT_END_REASON,
    SOLVED_END_REASON,
    read_attempts_raw,
)
from evaluation.stats import summarize_attempt_metrics

# The pinned scenario set: benchmark numbers are only comparable across runs
# that replay the same seeds.
DEFAULT_SEED_MANIFEST = Path("config/evaluation_seed_manifest.csv")

# pass@k columns worth printing; each is clamped to the run's attempt cap.
_PASS_AT_K_COLUMNS: tuple[int, ...] = (1, 2, 4, 8, 20)


@dataclasses.dataclass(frozen=True)
class BenchmarkCell:
    """One `evaluate --attempts` run: a checkpoint at one aircraft count."""

    checkpoint: Path
    max_aircraft: int
    out_dir: Path

    @property
    def label(self) -> str:
        return f"{self.out_dir.name}"


def checkpoint_slugs(checkpoints: Sequence[Path]) -> list[str]:
    """Name each checkpoint for its output directory.

    Directory names alone collide across runs (``checkpoint_000060`` exists in
    every trial), so fall back to including the parent directory for the whole
    set rather than only for the colliding pair.  If that qualification is
    still ambiguous, fail before any subprocess can overwrite a cell's
    artifacts.
    """
    names = [path.name for path in checkpoints]
    if len(set(names)) == len(names):
        return names

    slugs = [f"{path.parent.name}_{path.name}" for path in checkpoints]
    paths_by_slug: dict[str, list[Path]] = {}
    for path, slug in zip(checkpoints, slugs, strict=True):
        paths_by_slug.setdefault(slug, []).append(path)
    collisions = {slug: paths for slug, paths in paths_by_slug.items() if len(paths) > 1}
    if collisions:
        details = "; ".join(
            f"{slug}: {', '.join(str(path) for path in paths)}"
            for slug, paths in collisions.items()
        )
        raise ValueError(f"checkpoint output slug collision(s): {details}")
    return slugs


def build_cells(
    checkpoints: Sequence[Path],
    aircraft_counts: Sequence[int],
    *,
    out_root: Path,
) -> list[BenchmarkCell]:
    """Expand the checkpoint x aircraft grid into runs, aircraft count outermost.

    Ordering matters for a long benchmark: finishing every checkpoint at one
    aircraft count before starting the next means an interrupted run still
    leaves a complete, comparable set.
    """
    slugs = checkpoint_slugs(checkpoints)
    return [
        BenchmarkCell(
            checkpoint=checkpoint,
            max_aircraft=count,
            out_dir=out_root / f"a{count}_{slug}",
        )
        for count in aircraft_counts
        for checkpoint, slug in zip(checkpoints, slugs, strict=True)
    ]


def evaluate_argv(
    cell: BenchmarkCell,
    *,
    attempts: int,
    solve_threshold_s: float,
    seed_manifest: Path,
    max_steps: int,
    workers: int | None,
) -> list[str]:
    """Build the `evaluate` command line for one cell."""
    argv = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "-d",
        str(cell.checkpoint),
        "--max-aircraft",
        str(cell.max_aircraft),
        "--max-steps",
        str(max_steps),
        "--attempts",
        str(attempts),
        "--solve-threshold",
        str(solve_threshold_s),
        "--seed-manifest",
        str(seed_manifest),
        "--attempts-out-dir",
        str(cell.out_dir),
    ]
    if workers is not None:
        argv += ["--workers", str(workers)]
    return argv


class _CellSummaryError(RuntimeError):
    """A zero-exit cell did not produce a readable attempts summary."""


def collect_cell(cell: BenchmarkCell) -> dict[str, Any]:
    """Read one finished cell's artifacts into a comparison row."""
    summary_path = cell.out_dir / "attempts_summary.json"
    try:
        summary = read_json(summary_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _CellSummaryError(
            f"{type(exc).__name__}: unable to read {summary_path}: {exc}"
        ) from exc
    if not isinstance(summary, dict):
        raise _CellSummaryError(f"summary is not a JSON object: {summary_path}")
    attempts = read_attempts_raw(cell.out_dir / "attempts_raw.csv")
    return {
        "checkpoint": str(cell.checkpoint),
        "label": cell.label,
        "max_aircraft": cell.max_aircraft,
        "out_dir": str(cell.out_dir),
        "report": summary,
        "attempt_metrics": summarize_attempt_metrics(attempts),
    }


def pass_at(report: dict[str, Any], k: int) -> float | None:
    """pass@k from a report, or None when the run's cap never reached k."""
    curve = report["pass_at_k"]
    if k > len(curve):
        return None
    return float(curve[k - 1])


def _incomplete_grid_notice(
    rows: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    *,
    total_cells: int | None,
) -> list[str]:
    if not failures:
        return []
    attempted = len(rows) + len(failures) if total_cells is None else total_cells
    return [
        f"INCOMPLETE GRID: {len(rows)}/{attempted} cells completed.",
        *[
            f"- `{failure['label']}` failed (exit code {failure['exit_code']})."
            for failure in failures
        ],
        "",
    ]


def render_comparison(
    rows: list[dict[str, Any]],
    *,
    failures: Sequence[dict[str, Any]] = (),
    total_cells: int | None = None,
) -> str:
    """Render the cross-run comparison as a Markdown table."""
    failure_list = list(failures)
    notice = _incomplete_grid_notice(rows, failure_list, total_cells=total_cells)
    if not rows:
        if not notice:
            return "No completed benchmark runs.\n"
        return "\n".join([*notice, "No completed benchmark runs."]) + "\n"
    caps = {int(row["report"]["attempt_cap"]) for row in rows}
    ks = [k for k in _PASS_AT_K_COLUMNS if k <= max(caps)]
    header = (
        ["Checkpoint", "AC", "Det"]
        + [f"pass@{k}" for k in ks]
        + [
            "Never",
            "Mean att",
            "Success",
            "Infr",
            "Delayed",
            "Infr sev",
            "Max dev",
            "NZAct",
            "ExTrb",
        ]
    )
    align = ["---", "---:", "---:"] + ["---:"] * (len(header) - 3)
    lines = [
        *notice,
        f"| {' | '.join(header)} |",
        f"| {' | '.join(align)} |",
    ]
    for row in rows:
        report = row["report"]
        metrics = row["attempt_metrics"]
        shares = metrics["end_reason_share"]
        det = report["deterministic_solved_rate"]
        mean_attempts = report["mean_attempts_to_success"]
        cells = [
            f"`{row['label']}`",
            str(row["max_aircraft"]),
            "-" if det is None else f"{det:.1%}",
        ]
        for k in ks:
            value = pass_at(report, k)
            cells.append("-" if value is None else f"{value:.1%}")
        cells += [
            str(report["censored"]),
            "n/a" if mean_attempts is None else f"{mean_attempts:.2f}",
            f"{shares.get(SOLVED_END_REASON, 0.0):.1%}",
            f"{shares.get(INFRINGEMENT_END_REASON, 0.0):.1%}",
            f"{shares.get(DELAYED_END_REASON, 0.0):.1%}",
            f"{metrics['infringement_severity_mean']:.2f}",
            f"{metrics['max_dev_mean']:.1f}",
            f"{metrics['nz_actions_mean']:.2f}",
            f"{metrics['nz_ex_trb_mean']:.2f}",
        ]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def run_cell(argv: list[str], *, log_path: Path) -> int:
    """Run one cell, tee-ing its output to a log file, and return its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(  # noqa: S603 - argv is built here, never shell
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pass@k attempts benchmark over a checkpoint x aircraft grid."
    )
    parser.add_argument(
        "-d",
        "--checkpoint",
        dest="checkpoints",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="Checkpoint directory to benchmark; repeat for several checkpoints.",
    )
    parser.add_argument(
        "-a",
        "--max-aircraft",
        dest="aircraft_counts",
        type=int,
        nargs="+",
        default=[Config.MAX_AIRCRAFT_COUNT_EVAL],
        metavar="N",
        help=(
            "Aircraft counts to benchmark each checkpoint at "
            f"(default: {Config.MAX_AIRCRAFT_COUNT_EVAL})."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=20,
        metavar="K",
        help="Stochastic attempts per scenario (default: 20).",
    )
    parser.add_argument(
        "--solve-threshold",
        type=float,
        default=Config.OPERATIONAL_SUCCESS_S,
        metavar="SECONDS",
        help=(
            "Success window passed to every run (default: "
            f"{Config.OPERATIONAL_SUCCESS_S:g} s, the operational window)."
        ),
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=DEFAULT_SEED_MANIFEST,
        metavar="PATH",
        help=f"Pinned scenario seeds replayed by every run (default: {DEFAULT_SEED_MANIFEST}).",
    )
    parser.add_argument(
        "-m",
        "--max-steps",
        type=int,
        default=Config.EPISODE_STEPS,
        metavar="N",
        help=f"Maximum steps per episode (default: {Config.EPISODE_STEPS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Remote env runners per run (default: the evaluate default).",
    )
    parser.add_argument(
        "-o",
        "--out-root",
        type=Path,
        required=True,
        metavar="PATH",
        help="Directory that receives one subdirectory of artifacts per run.",
    )
    return parser


def main() -> None:
    force_runtime_log_color()
    args = build_parser().parse_args()

    checkpoints = [path.absolute() for path in args.checkpoints]
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        runtime_print("Benchmark", "error", message=f"checkpoint not found: {missing[0]}")
        sys.exit(1)
    if not args.seed_manifest.exists():
        runtime_print(
            "Benchmark", "error", message=f"seed manifest not found: {args.seed_manifest}"
        )
        sys.exit(1)

    cells = build_cells(checkpoints, args.aircraft_counts, out_root=args.out_root.absolute())
    runtime_print(
        "Benchmark",
        "start",
        runs=len(cells),
        aircraft=args.aircraft_counts,
        attempts=args.attempts,
        solve_threshold=args.solve_threshold,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        argv = evaluate_argv(
            cell,
            attempts=args.attempts,
            solve_threshold_s=args.solve_threshold,
            seed_manifest=args.seed_manifest,
            max_steps=args.max_steps,
            workers=args.workers,
        )
        runtime_print("Benchmark", "run", progress=f"{index}/{len(cells)}", cell=cell.label)
        code = run_cell(argv, log_path=cell.out_dir / "run.log")
        if code != 0:
            failures.append(
                {
                    "checkpoint": str(cell.checkpoint),
                    "exit_code": code,
                    "label": cell.label,
                    "max_aircraft": cell.max_aircraft,
                }
            )
            runtime_print(
                "Benchmark",
                "error",
                message=f"{cell.label} exited {code}",
                log=str(cell.out_dir / "run.log"),
            )
            continue
        try:
            row = collect_cell(cell)
        except _CellSummaryError as exc:
            failures.append(
                {
                    "checkpoint": str(cell.checkpoint),
                    "error": str(exc),
                    "exit_code": 0,
                    "label": cell.label,
                    "max_aircraft": cell.max_aircraft,
                }
            )
            runtime_print(
                "Benchmark",
                "error",
                message=f"{cell.label} produced an unreadable summary: {exc}",
                log=str(cell.out_dir / "run.log"),
            )
            continue
        rows.append(row)
        report = row["report"]
        runtime_print(
            "Benchmark",
            "done",
            cell=cell.label,
            pass_at_cap=f"{report['pass_at_cap']:.1%}",
            never_solved=report["censored"],
        )

    table = render_comparison(rows, failures=failures, total_cells=len(cells))
    out_root = args.out_root.absolute()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "benchmark_comparison.md").write_text(table, encoding="utf-8")
    (out_root / "benchmark_comparison.json").write_text(
        json.dumps(
            {
                "cells_attempted": len(cells),
                "cells_completed": len(rows),
                "failures": failures,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sys.stdout.write(f"\n{table}")
    runtime_print("Benchmark", "wrote", path=str(out_root / "benchmark_comparison.md"))

    if failures:
        runtime_print(
            "Benchmark",
            "error",
            message=f"failed runs: {', '.join(failure['label'] for failure in failures)}",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
