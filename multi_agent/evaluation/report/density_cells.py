"""Benchmark cells read back from stored ``attempts-benchmark`` artifacts.

A cell is one checkpoint at one aircraft count; this module loads cells and
derives the per-cell curves the density figures draw.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from common.jsonio import read_json
from evaluation.attempts import (
    INFRINGEMENT_END_REASON,
    AttemptRecord,
    read_attempts_per_seed,
    read_attempts_raw,
)

# One colour per density, in ascending aircraft order: cool for light traffic,
# warm for heavy.  Ordered by traffic, not by cell discovery order.
DENSITY_COLORS: tuple[str, ...] = ("#1f77b4", "#ff7f0e", "#d62728", "#6a3d9a", "#2ca02c")

# Cell directories written by ``scripts/attempts_benchmark.py`` are named
# ``a<aircraft>_<checkpoint slug>``.
_CELL_NAME = re.compile(r"^a(\d+)_(.+)$")


@dataclasses.dataclass(frozen=True)
class DensityCell:
    """One checkpoint evaluated at one aircraft count."""

    aircraft: int
    checkpoint: str
    cap: int
    scenarios: int
    """Scenarios in the run, as recorded by :func:`evaluation.stats.build_report`.

    Never re-derived from the attempt rows: a producer that stops retrying a
    scenario once it is solved writes fewer than ``scenarios * cap`` of them.
    """
    solve_threshold_s: float
    """Success window, in seconds, used to grade every attempt in the cell."""
    curve: list[float]
    """pass@k for k = 1..cap."""
    deterministic_rate: float | None
    attempts: list[AttemptRecord]
    deterministic_los_free_rate: float | None = None
    """Share of greedy baselines that did not terminate on lost separation."""

    @property
    def nz_actions_mean(self) -> float:
        return sum(a.nz_actions for a in self.attempts) / len(self.attempts)


def parse_cell_name(name: str) -> tuple[int, str]:
    """Split a benchmark cell directory name into (aircraft, checkpoint slug)."""
    match = _CELL_NAME.match(name)
    if match is None:
        raise ValueError(f"not a benchmark cell directory name: {name!r}")
    return int(match.group(1)), match.group(2)


def load_cell(directory: Path) -> DensityCell:
    """Load one benchmark cell from its output directory."""
    summary = read_json(directory / "attempts_summary.json")
    aircraft, checkpoint = parse_cell_name(directory.name)
    attempts = read_attempts_raw(directory / "attempts_raw.csv")
    per_seed_path = directory / "attempts_per_seed.csv"
    deterministic_los_free_rate: float | None = None
    if per_seed_path.is_file():
        outcomes = read_attempts_per_seed(per_seed_path)
        if len(outcomes) == int(summary["scenarios"]) and all(
            outcome.deterministic_end_reason is not None for outcome in outcomes
        ):
            deterministic_los_free_rate = sum(
                outcome.deterministic_end_reason != INFRINGEMENT_END_REASON for outcome in outcomes
            ) / len(outcomes)
    return DensityCell(
        aircraft=aircraft,
        checkpoint=checkpoint,
        cap=int(summary["attempt_cap"]),
        scenarios=int(summary["scenarios"]),
        solve_threshold_s=float(summary["solve_threshold_s"]),
        curve=[float(v) for v in summary["pass_at_k"]],
        deterministic_rate=(
            None
            if summary.get("deterministic_solved_rate") is None
            else float(summary["deterministic_solved_rate"])
        ),
        attempts=attempts,
        deterministic_los_free_rate=deterministic_los_free_rate,
    )


def discover_cells(roots: Sequence[Path], *, checkpoint: str) -> list[DensityCell]:
    """Find every cell of one checkpoint under ``roots``, ordered by density.

    ``roots`` are ``attempts-benchmark`` output roots; one root holding the
    whole grid and one root per aircraft count are both supported.  The match
    must resolve to exactly one checkpoint: a figure whose densities came from
    different checkpoints would be labelled with whichever one sorted first.
    """
    found: dict[int, DensityCell] = {}
    for root in roots:
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not _CELL_NAME.match(directory.name):
                continue
            aircraft, slug = parse_cell_name(directory.name)
            if checkpoint not in slug:
                continue
            if aircraft in found:
                raise ValueError(
                    f"two cells for {aircraft} aircraft match {checkpoint!r}: "
                    f"{found[aircraft].checkpoint} and {slug}"
                )
            found[aircraft] = load_cell(directory)
    if not found:
        raise ValueError(f"no benchmark cells matching {checkpoint!r} under {list(roots)}")
    slugs = {cell.checkpoint for cell in found.values()}
    if len(slugs) > 1:
        raise ValueError(
            f"{checkpoint!r} matches several checkpoints, so the densities would be "
            f"mixed in one figure: {sorted(slugs)}"
        )
    resolve_solve_threshold(tuple(found.values()), None)
    if len(found) > len(DENSITY_COLORS):
        raise ValueError(
            f"{len(found)} densities exceed the {len(DENSITY_COLORS)}-colour palette, "
            f"so one would be drawn as a blank panel: {sorted(found)}"
        )
    return [found[key] for key in sorted(found)]


def shared_solve_threshold(thresholds: Iterable[float], what: str) -> float:
    """The one grading window shared by ``what``; mixed windows are refused."""
    distinct = {float(threshold) for threshold in thresholds}
    if len(distinct) != 1:
        raise ValueError(f"{what} must share a solve threshold, got {sorted(distinct)}")
    return next(iter(distinct))


def resolve_solve_threshold(cells: Sequence[DensityCell], override: float | None) -> float:
    """Resolve one grading window and reject mixed windows in a figure."""
    shared = shared_solve_threshold(
        (cell.solve_threshold_s for cell in cells), "cells in one density figure"
    )
    return shared if override is None else override


def crossing_k(passk: Sequence[float], los_free: Sequence[float]) -> int | None:
    """First ``k`` where retrying has bought more solves than it cost in safety.

    ``pass@k`` only rises and ``LoS-free@k`` only falls, so their difference is
    monotone and this one index marks where the gap opens and never closes.  A
    tie is not a crossing: the two shares have to actually separate before the
    figure claims either one leads.
    """
    for k, (solved, safe) in enumerate(zip(passk, los_free, strict=True), start=1):
        if solved > safe:
            return k
    return None


def forced_los_curve(cell: DensityCell) -> list[float]:
    """Share of scenarios where every one of the first k attempts lost separation.

    The dual of ``pass@k``: instead of "at least one attempt solved it", this is
    "no attempt avoided losing separation", so retrying cannot deliver a clean
    episode.  It falls with k, because one clean attempt retires a scenario for
    every larger budget.

    This is the operational counterpart to :func:`exposure_curves`' descending
    ``LoS_free@k``.  That one asks whether *any* of k attempts infringed, so it
    necessarily rises with k - it counts exposure of the sampling process, not
    risk carried by a retry loop that keeps a clean attempt when it finds one.

    Selection is safety-first, matching the repository's priority: a clean
    attempt is preferred over an infringing one even when the infringing one has
    the better timing.  Ranking by deviation instead lets a fast-but-unsafe
    episode win, which makes the curve rise again at some k.
    """
    clean_at: dict[int, int] = {}
    for record in cell.attempts:
        if record.end_reason != INFRINGEMENT_END_REASON:
            clean_at[record.seed] = min(record.attempt, clean_at.get(record.seed, record.attempt))
    first_clean = Counter(clean_at.values())

    scenarios = float(cell.scenarios)
    curve: list[float] = []
    avoided = 0
    for k in range(1, cell.cap + 1):
        avoided += first_clean[k]
        curve.append(1.0 - avoided / scenarios)
    return curve


def action_count_bins(cells: Sequence[DensityCell], *, width: int = 4) -> list[float]:
    """Shared histogram edges over every cell's nonzero action counts.

    Densities are only comparable on identical bins, so the edges span the
    pooled range at a fixed width instead of being fitted per cell.
    """
    counts = [a.nz_actions for cell in cells for a in cell.attempts]
    if not counts:
        raise ValueError("no attempts to bin")
    if width <= 0:
        raise ValueError(f"bin width must be positive, got {width}")
    low = (min(counts) // width) * width
    high = ((max(counts) // width) + 1) * width
    return [float(edge) for edge in range(low, high + width, width)]


def exposure_curves(cell: DensityCell) -> tuple[list[float], list[float]]:
    """Return ``(pass@k, LoS_free@k)`` over k = 1..cap for one density.

    Both are fractions of the cell's recorded scenario count, so they are
    directly comparable on one axis:

    - ``pass@k`` - at least one of the first k attempts solved the scenario.
      Recomputed here rather than read from ``cell.curve`` so both families
      share one window definition; it reproduces the stored curve exactly.
    - ``LoS_free@k`` - *none* of the first k attempts lost separation.  It falls
      with k: every extra retry is another exposure, which is the cost the
      ascending pass@k curve hides.

    Separation loss is deliberately binary here: an attempt either terminated on
    it or it did not.  ``infr_severity`` is not a substitute for that end reason
    and is never consulted - it accumulates severity above
    ``Config.NEAR_CONFLICT_THRESHOLD``, well below the ``1.0`` that terminates
    an episode, so solved episodes routinely carry nonzero severity.  Reading a
    breach off that field would count near-misses as lost separation.  How close
    a near-miss came, and how severe a breach was, are out of scope.

    A producer that stops retrying a solved scenario simply has no rows for the
    attempts it skipped; those unrun attempts count as neither a solve nor an
    exposure.
    """
    solved_at: dict[int, int] = {}
    lost_at: dict[int, int] = {}
    for record in cell.attempts:
        if record.solved:
            solved_at[record.seed] = min(record.attempt, solved_at.get(record.seed, record.attempt))
        if record.end_reason == INFRINGEMENT_END_REASON:
            lost_at[record.seed] = min(record.attempt, lost_at.get(record.seed, record.attempt))
    first_solve = Counter(solved_at.values())
    first_loss = Counter(lost_at.values())

    scenarios = float(cell.scenarios)
    passk: list[float] = []
    los_free: list[float] = []
    solved = lost = 0
    for k in range(1, cell.cap + 1):
        solved += first_solve[k]
        lost += first_loss[k]
        passk.append(solved / scenarios)
        los_free.append(1.0 - lost / scenarios)
    return passk, los_free
