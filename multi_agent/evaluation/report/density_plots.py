"""Cross-density figures for retry-until-solved (pass@k) benchmark runs.

``evaluation.stats`` renders one figure per benchmark *cell* - one
checkpoint at one aircraft count.  Comparing traffic densities means putting
several of those cells on shared axes, which is what this module does:

- :func:`render_passk_density` - one pass@k curve per aircraft count, each with
  its greedy (deterministic) level as a dashed baseline.
- :func:`render_exposure_density` - one panel per aircraft count, pairing the
  benefit of retries (pass@k) with their binary loss-of-separation exposure
  (LoS-free@k).
- :func:`render_action_count_density` - the distribution of nonzero actions per
  episode, one filled outline per aircraft count, each with its mean as a
  dashed line.

Both read only the artifacts a finished ``attempts-benchmark`` run leaves
behind (``attempts_summary.json`` + ``attempts_raw.csv``), so figures can be
rebuilt long after the episodes are gone, and neither needs Ray or a
checkpoint.  Cells carry different episode counts when scenario sets or caps
differ, so the histogram plots *share of episodes* rather than raw counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evaluation.report.density_cells import (
    DENSITY_COLORS,
    DensityCell,
    action_count_bins,
    crossing_k,
    exposure_curves,
    forced_los_curve,
    resolve_solve_threshold,
)

INK = "#333333"

# Total fill opacity to spend across the overlaid densities; each body gets an
# equal share so three overlapping fills stay legible instead of turning brown.
_FILL_ALPHA_BUDGET = 0.48


def greedy_note(cell: DensityCell) -> str:
    """Read the panel's dashed levels out as text, in metric order.

    Both greedy levels share one colour and one dash rhythm because greedy
    LoS-free is >= greedy solved by construction, so the upper line is always
    the safety level.  That convention breaks down for a cell holding only one
    of them, which is why the numbers are printed rather than left to texture.
    Empty when the run had no greedy pass at all and neither line is drawn.
    """
    levels = (cell.deterministic_rate, cell.deterministic_los_free_rate)
    if all(level is None for level in levels):
        return ""
    shown = " / ".join("-" if level is None else f"{level:.0%}" for level in levels)
    return f"greedy {shown}"


def _figure(width: float, height: float, title: str, subtitle: str) -> tuple:
    """Open a single-axes figure with the repository's title layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    # Bold run identity on the figure, plain protocol note on the axes: two
    # separate slots, so neither can overprint the other.
    fig.suptitle(title, x=0.015, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK, pad=10)
    # Only the compared axis is gridded: vertical rules add a fourth kind of
    # vertical line next to bin risers, mean lines and the spine.
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax, plt


def _configure_k_axis(ax, cap: int) -> None:
    """Use the shared log-k scale of every retry figure."""
    import matplotlib.ticker

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, cap] if cap > 10 else list(range(1, cap + 1)))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.95, cap * 1.02)


def _span(values: set[int]) -> str:
    """Format one shared protocol value or an honest inclusive range."""
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low}-{high}"


def render_passk_density(
    path: Path,
    cells: Sequence[DensityCell],
    *,
    checkpoint: str,
    solve_threshold_s: float | None = None,
) -> None:
    """Render pass@k curves for several densities on shared axes."""
    cap = max(cell.cap for cell in cells)
    solve_threshold_s = resolve_solve_threshold(cells, solve_threshold_s)
    scenario_span = _span({cell.scenarios for cell in cells})
    cap_span = _span({cell.cap for cell in cells})
    greedy_clause = (
        "; dashed line = greedy (deterministic) solve rate"
        if any(cell.deterministic_rate is not None for cell in cells)
        else ""
    )
    fig, ax, plt = _figure(
        11.0,
        6.4,
        f"pass@k vs traffic density - {checkpoint}",
        f"{scenario_span} pinned scenarios, {cap_span} attempts, "
        f"{solve_threshold_s:g} s success window{greedy_clause}",
    )

    for color, cell in zip(DENSITY_COLORS, cells, strict=False):
        ks = list(range(1, cell.cap + 1))
        ax.plot(
            ks,
            cell.curve,
            marker="o",
            color=color,
            markersize=4.5,
            linewidth=2.0,
            label=f"{cell.aircraft} aircraft - {cell.nz_actions_mean:.1f} nonzero actions/episode",
        )
        # Right of the plot, two label lanes: reached level first, then the
        # greedy level its dashed line marks.  Annotating on the curve itself
        # collides with the other densities.
        ax.annotate(
            f"pass@{cell.cap} {cell.curve[-1]:.0%}",
            (cell.cap, cell.curve[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            fontweight="bold",
            color=color,
            fontsize=9,
            annotation_clip=False,
        )
        if cell.deterministic_rate is None:
            continue
        ax.axhline(cell.deterministic_rate, linestyle="--", color=color, linewidth=1)
        ax.annotate(
            f"deterministic {cell.deterministic_rate:.0%}",
            (cell.cap, cell.deterministic_rate),
            textcoords="offset points",
            xytext=(88, 0),
            ha="left",
            va="center",
            color=color,
            fontsize=9,
            annotation_clip=False,
        )

    _configure_k_axis(ax, cap)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("attempts allowed (k)")
    ax.set_ylabel("fraction of scenarios solved")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def render_forced_los_density(
    path: Path,
    cells: Sequence[DensityCell],
    *,
    checkpoint: str,
    solve_threshold_s: float | None = None,
) -> None:
    """Render the separation-loss risk a retry budget cannot remove."""
    import matplotlib.ticker

    cap = max(cell.cap for cell in cells)
    solve_threshold_s = resolve_solve_threshold(cells, solve_threshold_s)
    scenario_span = _span({cell.scenarios for cell in cells})
    cap_span = _span({cell.cap for cell in cells})
    greedy_clause = (
        "; dashed line = greedy (deterministic) rate"
        if any(cell.deterministic_los_free_rate is not None for cell in cells)
        else ""
    )
    fig, ax, plt = _figure(
        11.0,
        6.4,
        f"Separation loss vs traffic density - {checkpoint}",
        f"{scenario_span} pinned scenarios, {cap_span} attempts, "
        f"{solve_threshold_s:g} s success window; share of scenarios where every "
        f"attempt lost separation{greedy_clause}",
    )

    curves = [forced_los_curve(cell) for cell in cells]
    for color, cell, curve in zip(DENSITY_COLORS, cells, curves, strict=False):
        ks = list(range(1, cell.cap + 1))
        ax.plot(
            ks,
            curve,
            marker="o",
            color=color,
            markersize=4.5,
            linewidth=2.0,
            label=f"{cell.aircraft} aircraft - {curve[0]:.0%} on one attempt",
        )
        ax.annotate(
            f"LoS-forced@{cell.cap} {curve[-1]:.0%}",
            (cell.cap, curve[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            fontweight="bold",
            color=color,
            fontsize=9,
            annotation_clip=False,
        )
        if cell.deterministic_los_free_rate is None:
            continue
        greedy = 1.0 - cell.deterministic_los_free_rate
        ax.axhline(greedy, linestyle="--", color=color, linewidth=1)
        ax.annotate(
            f"greedy {greedy:.0%}",
            (cell.cap, greedy),
            textcoords="offset points",
            xytext=(112, 0),
            ha="left",
            va="center",
            color=color,
            fontsize=9,
            annotation_clip=False,
        )

    _configure_k_axis(ax, cap)
    # Risk lives in the bottom decile at low density; a 0-1 axis would flatten
    # every curve into the baseline.  The floor stays at zero so the shares are
    # still read against "no separation loss at all".
    ceiling = max(
        [
            *(c[0] for c in curves),
            *(
                1.0 - cell.deterministic_los_free_rate
                for cell in cells
                if cell.deterministic_los_free_rate is not None
            ),
        ]
    )
    ax.set_ylim(0.0, max(0.05, ceiling * 1.18))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("attempts allowed (k)")
    ax.set_ylabel("share of scenarios forced into separation loss")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def render_exposure_density(
    path: Path,
    cells: Sequence[DensityCell],
    *,
    checkpoint: str,
    solve_threshold_s: float | None = None,
) -> None:
    """Pair retry benefit and binary separation-loss exposure by density."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.colors import to_rgba

    count = len(cells)
    solve_threshold_s = resolve_solve_threshold(cells, solve_threshold_s)
    # Four stacked caption lines per panel sit between the subtitle and the
    # axes; the reserved band is sized for them, not for the tallest curve.
    height = 5.5
    fig, raw_axes = plt.subplots(1, count, figsize=(11.0, height), sharex=True, sharey=True)
    axes = [raw_axes] if count == 1 else list(raw_axes)
    name_size, note_size = (10.0, 8.5) if count <= 3 else (9.0, 7.5)
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=1.0 - 1.85 / height,
        bottom=0.62 / height,
        wspace=0.10,
    )
    fig.suptitle(
        f"Retry exposure vs traffic density - {checkpoint}",
        x=0.015,
        y=1.0 - 0.30 / height,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    scenario_values = {cell.scenarios for cell in cells}
    scenario_span = _span(scenario_values)
    cap = max(cell.cap for cell in cells)
    cap_span = _span({cell.cap for cell in cells})
    greedy_clause = (
        "; dashed = greedy"
        if any(
            cell.deterministic_rate is not None or cell.deterministic_los_free_rate is not None
            for cell in cells
        )
        else ""
    )
    fig.text(
        0.015,
        1.0 - 0.55 / height,
        f"{scenario_span} pinned scenarios, {cap_span} attempts, "
        f"{solve_threshold_s:g} s success window; solid = pass@k "
        "(solved at least once), dotted = LoS-free@k "
        f"(never lost separation){greedy_clause}",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
    )

    for color, cell, ax in zip(DENSITY_COLORS, cells, axes, strict=False):
        ks = list(range(1, cell.cap + 1))
        passk, los_free = exposure_curves(cell)
        crossing = crossing_k(passk, los_free)
        # Monotone curves: past the crossing the gap never closes again, so one
        # index governs both the rule and the band.
        leads = [crossing is not None and k >= crossing for k in ks]
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.fill_between(
            ks,
            passk,
            los_free,
            where=leads,
            interpolate=False,
            facecolor=to_rgba(color, 0.15),
            edgecolor="none",
            zorder=2,
        )
        if crossing is not None:
            ax.axvline(crossing, color=INK, linewidth=0.9, alpha=0.35, zorder=1)
        for baseline in (cell.deterministic_rate, cell.deterministic_los_free_rate):
            if baseline is not None:
                ax.axhline(
                    baseline,
                    linestyle=(0, (5, 2)),
                    color=color,
                    linewidth=1.0,
                    zorder=3,
                )
        ax.plot(
            ks,
            los_free,
            marker="s",
            color=color,
            markersize=3.8,
            linewidth=1.8,
            linestyle=(0, (1, 1.6)),
            markerfacecolor="white",
            markeredgewidth=1.1,
            zorder=4,
        )
        ax.plot(
            ks,
            passk,
            marker="o",
            color=color,
            markersize=3.8,
            linewidth=2.0,
            zorder=5,
        )

        crossing_text = (
            f"pass@k leads from k={crossing}"
            if crossing is not None
            else f"pass@k never leads up to k={cell.cap}"
        )
        lines = [
            (f"{cell.aircraft} aircraft", color, name_size, "bold"),
            (
                f"pass@{cell.cap} {passk[-1]:.0%} / LoS-free@{cell.cap} {los_free[-1]:.0%}",
                INK,
                note_size,
                "normal",
            ),
        ]
        greedy = greedy_note(cell)
        if greedy:
            lines.append((greedy, INK, note_size, "normal"))
        lines.append((crossing_text, INK, note_size, "normal"))
        # Stacked upward from the axes: narrow panels of a five-density grid
        # cannot hold these end-state numbers on one line, and a cell without
        # greedy levels then simply has one line fewer instead of a hole.
        for index, (text, text_color, size, weight) in enumerate(reversed(lines)):
            ax.annotate(
                text,
                (0.0, 1.0),
                xycoords="axes fraction",
                textcoords="offset points",
                xytext=(0, 4 + 18 * index),
                ha="left",
                va="bottom",
                color=text_color,
                fontsize=size,
                fontweight=weight,
                annotation_clip=False,
            )

        _configure_k_axis(ax, cap)
        ax.set_ylim(0.0, 1.0)

    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].set_ylabel(
        f"share of the {scenario_span} scenarios"
        if len(scenario_values) == 1
        else "share of scenarios"
    )
    fig.supxlabel("attempts allowed (k)", fontsize=10, y=0.12 / height)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def density_caption(cells: Sequence[DensityCell], *, bin_width: int) -> str:
    """Spell out what one bar of the histogram is counting.

    ``n=2000 episodes`` alone is ambiguous - it is a scenario set replayed under
    a per-scenario attempt budget, and both numbers matter when comparing runs.
    Cells of one grid normally share both, so a range is printed only when they
    genuinely differ rather than implying a precision the figure does not have.
    """

    scenarios = _span({cell.scenarios for cell in cells})
    cap = _span({cell.cap for cell in cells})
    return (
        f"{scenarios} pinned scenarios x up to {cap} stochastic attempts each "
        f"(episodes per density in the legend); {bin_width}-action bins; dashed line = mean"
    )


def render_action_count_density(
    path: Path,
    cells: Sequence[DensityCell],
    *,
    checkpoint: str,
    bin_width: int = 4,
) -> None:
    """Render the nonzero-action distribution for several densities."""
    import matplotlib.ticker
    from matplotlib.colors import to_rgba

    edges = action_count_bins(cells, width=bin_width)
    fig, ax, plt = _figure(
        11.0,
        6.4,
        f"Nonzero actions per episode - {checkpoint}",
        density_caption(cells, bin_width=bin_width),
    )

    # Denser grids muddy the overlaps, so the body tint thins as densities are added.
    fill_alpha = _FILL_ALPHA_BUDGET / max(3, len(cells))
    tallest = 0.0
    for index, (color, cell) in enumerate(zip(DENSITY_COLORS, cells, strict=False)):
        counts = [a.nz_actions for a in cell.attempts]
        weights = [1.0 / len(counts)] * len(counts)
        # Two passes per density: a translucent body that says which side of the
        # outline the density owns, then the outline.  Every outline is drawn
        # above every fill, so no crossing is ever hidden - which is why this is
        # not one ``stepfilled`` call sharing a zorder between face and edge.
        ax.hist(
            counts,
            bins=edges,
            weights=weights,
            histtype="stepfilled",
            facecolor=to_rgba(color, fill_alpha),
            edgecolor="none",
            linewidth=0.0,
            zorder=2 + 0.01 * index,
        )
        shares, _, _ = ax.hist(
            counts,
            bins=edges,
            weights=weights,
            histtype="step",
            edgecolor=color,
            linewidth=1.8,
            zorder=4 + 0.01 * index,
            label=(
                f"{cell.aircraft} aircraft - mean {cell.nz_actions_mean:.1f} - "
                f"n={len(counts)} episodes"
            ),
        )
        tallest = max(tallest, float(max(shares)))
        # The mean is reported in the legend, not beside its line: no offset rule
        # keeps in-plot labels clear of their own line, of each other and of the
        # legend for arbitrary means.  A marker on the top edge keeps the line
        # findable where it runs inside a fill.
        mean = cell.nz_actions_mean
        ax.axvline(mean, linestyle=(0, (4, 3)), color=color, linewidth=1.2, zorder=3)
        ax.plot(
            [mean],
            [1.0],
            transform=ax.get_xaxis_transform(),
            marker="v",
            markersize=5,
            color=color,
            clip_on=False,
            zorder=6,
        )

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, tallest * 1.15)
    # Ticks land on bin edges, so a bin can be read off the axis exactly.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2 * bin_width))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("nonzero actions in the episode")
    ax.set_ylabel("share of episodes")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
